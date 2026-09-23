# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of `/app/data` matching the pattern `sub-*/sub-*.nwb`. It uses `h5py` directly (not `pynwb`) to read only the needed datasets from each file. A `session_files()` function uses `glob.glob` to find all 152 NWB files. Files are processed either sequentially or in parallel using `ProcessPoolExecutor` with 8 workers.

ii.
```python
def session_files():
    import glob
    return sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')))

def load_session(path):
    """Read everything needed from one NWB file."""
    with h5py.File(path, 'r') as f:
        ident = f['identifier'][()].decode()
        scene = ident.split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        day = f['general/session_id'][()].decode()
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:, 0] > 0
        ...
```

iii. The AI chose h5py over pynwb for efficiency, reading only needed datasets. The glob pattern finds all NWB files across all subject directories. CONVERSION_NOTES confirms 152 files across 11 subjects were found, matching the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subject IDs are collected across all sessions and sorted numerically.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({i['subject'] for i in infos}, key=lambda x: int(x[1:]))
subject_idx = np.array([subjects.index(i['subject']) for i in infos], dtype=np.int64)
```

iii. Subject IDs are read from the NWB metadata rather than parsing directory names. This is more robust as it uses the canonical identifier stored in the file.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session day is extracted from `general/session_id` in the NWB file.

ii.
```python
day = f['general/session_id'][()].decode()
```

iii. The one-file-per-session structure matches the DANDI dataset organization. 152 sessions across 11 subjects were found.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` and `teleport` behavior time series. Trial start indices are where `trial_start > 0`, and trial end indices are where `teleport > 0`. The trial window used is `[start-1, stop-1)`, matching the reference code's `preprocessing.dff` convention.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)          # reference trial window
    ev = events[:, sl]
```

iii. The AI documented that the reference code uses `[start-1:stop-1)` windows in `preprocessing.dff`. The CONVERSION_NOTES Step 1 notes this convention and explains the -1 offset.

## 1-e. How are trials filtered based on quality controls?

i. Four trial filters are applied: (1) lick sensor error trials (>30% of samples with cumulative lick count > 2) are dropped; (2) trials shorter than 2 bins are dropped; (3) trials with NaN neural events are dropped; (4) trials with position < -100 (pre-TTL-sync) are dropped.

ii.
```python
LICK_ERR_FRAC = 0.30
...
lick_error[i] = (np.sum(lick_tr > 2) / max(1, len(lick_tr))) > LICK_ERR_FRAC
...
if lick_error[i]:
    drop['lick_error'] += 1
    continue
if ev.shape[1] < 2:
    drop['too_short'] += 1
    continue
if np.any(np.isnan(ev)):
    drop['nan_events'] += 1
    continue
if np.any(pos < -100):
    drop['pre_sync'] += 1
    continue
```

iii. The lick error filter matches the paper's criterion (0.65% of trials). The other filters are defensive checks for edge cases. CONVERSION_NOTES reports 81 lick-error trials dropped, matching the paper's 81/12,376.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces, NOT the stored `Deconvolved` field in the NWB. The AI explicitly identifies that the NWB's `Deconvolved` is suite2p's default deconvolution, not the paper's custom processing.

ii.
```python
fl = f['processing/ophys/Fluorescence']
planes = sorted(fl.keys())
Fl, Fnl = [], []
for p in planes:
    sel = iscell[plane_idx == int(p.replace('plane', ''))]
    Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
    Fnl.append(f[f'processing/ophys/Neuropil/{p}/data'][:, :].T[sel])
F = np.concatenate(Fl, axis=0)
Fneu = np.concatenate(Fnl, axis=0)
```

iii. CONVERSION_NOTES Step 4 confirms: "Recompute dF/F (maximin, neuropil 0.7) + OASIS from NWB F/Fneu, exactly as `preprocessing.dff`".

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's `preprocessing.dff` pipeline: per-trial windows `[start-1, stop-1)`, neuropil subtraction (coefficient 0.7) with per-trial neuropil mean added back, maximin baseline (Gaussian smooth sigma=15, then min filter 300, then max filter 300), dF/F = (F - baseline)/|baseline|, 2-frame Gaussian smoothing, then OASIS deconvolution (tau=0.7, frame_rate). However, the **final delivered dataset stores dF/F rather than the deconvolved events**, based on a controlled experiment showing dF/F decodes better.

ii.
```python
def compute_events(F, Fneu, starts, stops, frame_rate):
    f_ = np.full(F.shape, np.nan, dtype=np.float64)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float64)
    for start, stop in zip(starts, stops):
        f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
        fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]
    # neuropil subtraction
    f_ -= NEU_COEF * fneu_
    ...
    # maximin baseline
    flow[:, sl] = nansmooth(f_[:, sl], BASELINE_SIGMA)
    flow[:, sl] = ndimage.minimum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
    flow[:, sl] = ndimage.maximum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
    ...
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl], dtype=np.float32),
                               2000, TAU, frame_rate)
    return dff, events
```
Then in `process_session`:
```python
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```
Default `--signal dff` is used for the delivered dataset.

iii. The AI ran a controlled experiment (Step 12) comparing events vs dF/F and found dF/F outperforms events for every decoder output. The dF/F is still computed with the paper's full pipeline, just stopping before the OASIS deconvolution step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` flag from suite2p manual curation (only ROIs with `iscell[:,0] > 0`); (2) putative interneurons excluded (dF/F vs speed Pearson r > 0.5). The interneuron filter is computed via vectorized matrix operations.

ii.
```python
iscell = seg['iscell'][:, 0] > 0
...
sel = iscell[plane_idx == int(p.replace('plane', ''))]
Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
...
# putative interneurons
Dz = D - D.mean(axis=1, keepdims=True)
spz = sp - sp.mean()
speed_corr = (Dz @ spz) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
```

iii. CONVERSION_NOTES: "0.42 +- 0.85% of cells excluded" matching the paper. 409 putative interneurons removed from 138,678 iscell ROIs (0.30%).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since neural and behavioral data share the same sampling (already aligned in the NWB), slicing by the trial window `[start-1, stop-1)` automatically aligns to trial start.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
```

iii. The NWB files already contain neural and behavioral data aligned to the same timebase at ~15.5 Hz. No additional alignment is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate of ~15.5 Hz is preserved (64.48 ms per bin). No temporal rebinning is applied. The time bin size is computed from the median behavior timestamp intervals.

ii.
```python
frame_rate = 1.0 / np.median(np.diff(time))
...
frame_rates = np.array([i['frame_rate'] for i in infos])
bin_ms = float(1000.0 / np.median(frame_rates))
```

iii. Matches "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavioral time series.

ii.
```python
time = b['position/timestamps'][:]
...
tt = S['time'][sl] - S['time'][s - 1]
```

iii. Timestamps are the same for all behavioral variables, so using position timestamps is equivalent to any other.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the first sample of the trial (`s - 1`) is subtracted from all timestamps within the trial window.

ii.
```python
tt = S['time'][sl] - S['time'][s - 1]
```

iii. Simple subtraction to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same timebase in the NWB file. The same slice `[s-1, e-1)` is used for both, so they are inherently aligned.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
...
tt = S['time'][sl] - S['time'][s - 1]
```

iii. Data rates verified to match at ~15.5 Hz. Any length mismatch is handled by truncating to the common length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series (also called `morph` in the reference code).

ii.
```python
beh = dict(..., env=g('environment'), ...)
...
u = np.unique(beh['env'][sl])
morph[i] = int(u[0])
```

iii. Environment is 0 (ENV1) or 1 (ENV2), constant within each trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique environment value within the trial slice `[start:stop]` is taken. The per-trial value is broadcast to all timepoints.

ii.
```python
morph[i] = int(u[0])
...
inp = np.stack([..., np.full(T, float(morph[i])), ...])
```

iii. The AI uses `unique` and takes the first element, which is valid since environment is constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index (0-based), derived from the loop counter over `starts`/`stops` pairs.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp = np.stack([..., np.full(T, float(i)), ...])
```

iii. The loop index `i` serves as the trial number within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop index. The value is constant across all timepoints within a trial.

ii.
```python
np.full(T, float(i))
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavior time series. Reward delivery events are mapped to the behavior timebase using `searchsorted`, and combined with the reward zone flag to determine `isreward`.

ii.
```python
reward_t = b['Reward/timestamps'][:]
...
reward = np.zeros_like(beh['pos'])
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
beh['reward'] = reward
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. The AI uses both reward delivery AND reward zone entry, matching the reference code's `get_trial_types` which defines isreward as reward AND rzone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial `i`, the previous trial outcome is `isreward[i-1]`. For the first trial (i=0), the value is set to 1 (rewarded), reasoning that each imaging session was preceded by ~30 rewarded warm-up trials.

ii.
```python
inp = np.stack([...,
    np.full(T, float(isreward[i - 1]) if i > 0 else 1.0)]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "first trial of a session set to 1 (the mouse had just run ~30 rewarded warm-up trials on the same zone immediately before imaging)."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. The reward zone location is determined from the scene name (parsed from the NWB `identifier` field) with switches at trial 30, following `behavior.get_reward_zones`.

ii.
```python
def scene_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.match(r'Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.match(r'Env\d_Location([ABC])_to_([ABC])$', scene)
    ...
    n0 = min(change_trial, ntrials)
    return np.array([z0] * n0 + [z1] * (ntrials - n0))
```
```python
REWARD_ZONES = {'A': (80., 130.), 'B': (200., 250.), 'C': (320., 370.)}
```

iii. Reward zone coordinates match the paper (A 80-130, B 200-250, C 320-370 cm). Zone labels derived from scene names with switch at trial 30, following `behavior.get_reward_zones`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone. Zero inside the zone, negative before, positive after.

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
```

iii. Matches the paper's reward-relative distance concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins: <-50, [-50,-10), [-10,0), 0, (0,10], (10,50], >50. The implementation uses conditional assignment rather than `np.digitize`.

ii.
```python
def discretize_distance(d):
    out = np.full(d.shape, 3, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. Bin edges match the task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial window slice is used for both neural and behavioral data — no additional alignment needed.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
pos = beh['pos'][sl]
```

iii. Neural and behavioral data are sample-aligned in the NWB.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['pos'][sl]
```

iii. Position directly records the animal's location in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
discretize_position(pos)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins over the 450 cm track using integer division by 90 and clipping to [0, 4].

ii.
```python
def discretize_position(pos):
    return np.clip((pos // 90).astype(np.int64), 0, 4)
```

iii. 450/5 = 90 cm per bin. The `clip` handles any position values slightly outside [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial window slice — no additional alignment needed.

ii.
```python
sl = slice(s - 1, e - 1)
pos = beh['pos'][sl]
```

iii. Sample-aligned in the NWB.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
beh = dict(..., lick=g('lick'), ...)
...
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. The `lick` variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. Instructions specify binary output (no/yes). The raw values can be >1 (cumulative counts), so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial window slice — no additional alignment needed.

ii.
```python
sl = slice(s - 1, e - 1)
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. Sample-aligned in the NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field which encodes the scene name (e.g., `Env1_LocationB_to_A`). The scene name is parsed to determine the reward zone label(s) and whether there is a switch.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
```

iii. This follows the reference code's `behavior.get_reward_zones` which derives zone labels from the scene name.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed with regex to extract zone labels. For switch scenes (e.g., `LocationA_to_B`), the first 30 trials get zone A and remaining trials get zone B, matching `change_trial=30` from the reference code.

ii.
```python
def scene_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.match(r'Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.match(r'Env\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'Env\d_([ABC])_to_Env\d_([ABC])$', scene)
    if m:
        z0, z1 = m.group(1), m.group(2)
        n0 = min(change_trial, ntrials)
        return np.array([z0] * n0 + [z1] * (ntrials - n0))
```
Zone label mapped to index:
```python
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
np.full(T, ZONE_TO_IDX[zone], dtype=np.int64)
```

iii. CONVERSION_NOTES: "zone label/coords can be derived from scene name", verified against measured reward-zone entry positions (median error < 2 cm).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavior time series.

ii.
```python
reward_t = b['Reward/timestamps'][:]
reward = np.zeros_like(beh['pos'])
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
beh['reward'] = reward
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. Uses the reference code's definition: `isreward = any(reward > 0) AND any(rzone > 0)`, from `behavior.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to the nearest behavior timepoint using `searchsorted`. For each trial, isreward is 1 if a reward was delivered AND the reward zone was entered, else 0. The value is constant across all timepoints.

ii.
```python
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
...
np.full(T, isreward[i], dtype=np.int64)
```

iii. Note: the trial slice used for isreward computation is `[start:stop]` (without the -1 offset), while the behavioral data for the trial uses `[start-1:stop-1]`. This matches the reference code where `get_trial_types` uses `[start:stop]`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: All streams truncated to the common minimum length (10 two-plane sessions affected).
- **Lick sensor errors**: Trials with >30% of samples having cumulative lick > 2 are dropped (81 trials).
- **Pre-TTL-sync samples**: Trials with position < -100 are dropped (0 triggered).
- **Short trials**: Trials with < 2 bins are dropped (0 triggered).
- **NaN neural events**: Trials with NaN in the neural data are dropped (0 triggered).
- **Trial count mismatch**: `ntr = min(len(starts), len(stops))` handles any mismatch between trial starts and stops.

ii.
```python
n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
if F.shape[1] != n or len(time) != n:
    print(f'  truncating streams to {n} samples ...')
F = F[:, :n]; Fneu = Fneu[:, :n]; time = time[:n]
beh = {k: v[:n] for k, v in beh.items()}
```

iii. The length mismatch is documented as the "one frame correction" from the reference code. The lick error filter matches the paper exactly.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with h5py (~0.1-4s per session)
2. **dF/F + OASIS deconvolution** (~0.3-8s per session, depending on cell count)
3. **Pickling the output** (~10 GB file)

ii. N/A

iii. Total estimated: ~13 min serial, ~3 min with 8 workers. Actual total: 147s.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial sequentially for constructing input/output arrays. The `compute_events` function loops over trials for the baseline computation and deconvolution. The interneuron detection is already vectorized (matrix product instead of per-cell loop).

ii. N/A

iii. Variable trial lengths make full vectorization awkward. The per-trial dF/F loop is inherent to the algorithm (per-trial baselines).

## 13-c. What processing does the code repeat multiple times?

i. No significant repeated processing. The code does a single pass through all NWB files, computing dF/F and events in one call to `compute_events`, then extracting per-trial data. Unlike the human reference which has a separate survey step that loads all files twice, the AI processes each file once.

ii. N/A

iii. The AI's approach is more efficient than the reference by avoiding a separate survey pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes both dF/F and OASIS-deconvolved events via `compute_events`, but only uses one of them (dF/F by default). The deconvolved events array is computed but discarded when `--signal dff` is used. Additionally, `dff_kept` is computed for the interneuron filter even when using events, but since dF/F is used as the neural signal, both are needed.

ii.
```python
dff, events = compute_events(S['F'], S['Fneu'], starts, stops, S['frame_rate'])
...
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```

iii. The OASIS deconvolution step is unnecessary when using dF/F but is always computed. This adds ~30% to the neural processing time.
