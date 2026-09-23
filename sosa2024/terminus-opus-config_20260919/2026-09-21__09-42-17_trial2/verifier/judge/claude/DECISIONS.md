# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read NWB files from the `/app/data` directory. All `*.nwb` files under `sub-*` directories are discovered via `glob.glob`. Each NWB file corresponds to one session. Behavior streams, fluorescence, and neuropil data are loaded from the HDF5 groups within each file. The AI uses multiprocessing (spawn Pool with 8 workers) for parallel session loading.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
with h5py.File(fname, 'r') as f:
    scene = f['identifier'][()].decode().split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    ...
    beh = load_behavior(f)
    F, Fneu, plane_idx = load_fluorescence(f)
```

iii. The AI chose h5py over pynwb for speed. All 152 NWB files across 11 subjects are discovered and loaded. The glob pattern ensures all subject directories and session files are found.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the `general/subject/subject_id` field within each NWB file, then unique subjects are collected from all converted sessions and sorted numerically.

ii.
```python
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
```

iii. Subject identity comes directly from the NWB metadata. 11 subjects are found, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by the `general/session_id` field. Results are sorted by (subject, session_id).

ii.
```python
results.sort(key=lambda r: (r['subject'], r['session_id']))
```

iii. The one-file-per-session structure is standard for DANDI NWB datasets.

## 1-d. How are the data split into trials?

i. Trials are defined as the interval from `trial_start > 0` to `teleport > 0`. The AI uses `np.where(beh['trial_start'] > 0)` for start indices and `np.where(beh['teleport'] > 0)` for stop indices.

ii.
```python
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
```

iii. This matches the reference paper's trial definition: trial starts at entry to the track and ends at the teleport event. The teleport frame is excluded because its position is interpolated.

## 1-e. How are trials filtered based on quality controls?

i. Trials with lick-sensor errors are dropped. A trial is flagged as a lick-sensor error if >30% of its frames have a cumulative lick count > 2. This matches the paper's criterion and reproduces exactly 81 dropped trials. Additionally, trials truncated by a behavior/imaging length mismatch are dropped, and sessions with < 2 usable trials would be skipped (never triggered).

ii.
```python
LICK_ERROR_FRAC = 0.30
LICK_ERROR_COUNT = 2
...
lick_bad = np.array([(beh['lick'][s:e] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for s, e in zip(starts, stops)])
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_bad[i]:
        continue
```

iii. The paper states "detected by >30% of the frame samples in the trial containing a cumulative lick count >2" and "n = 81 out of 12,376 trials removed". The AI's threshold of 30% reproduces exactly 81 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) arrays stored in the NWB file under `processing/ophys/Fluorescence/planeN` and `processing/ophys/Neuropil/planeN`. The NWB `Deconvolved` field is explicitly NOT used.

ii.
```python
def load_fluorescence(f):
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0].astype(bool)
    plane_idx = seg['planeIdx'][:]
    Fs, Fns = [], []
    for p in np.unique(plane_idx):
        Fs.append(f[f'processing/ophys/Fluorescence/plane{p}/data'][:].T)
        Fns.append(f[f'processing/ophys/Neuropil/plane{p}/data'][:].T)
    F = np.concatenate(Fs, axis=0).astype(np.float32)
    Fneu = np.concatenate(Fns, axis=0).astype(np.float32)
```

iii. The paper computes its own dF/F from raw F and Fneu; the NWB `Deconvolved` is suite2p's deconvolution of raw F without the paper's neuropil correction or maximin baseline.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's `preprocessing.dff` pipeline: (1) restrict to within-trial samples (NaN outside), (2) subtract 0.7 * Fneu, (3) add back per-trial mean neuropil, (4) Gaussian smoothing with sigma 15 frames, (5) minimum filter over 300 frames, (6) maximum filter over 300 frames to get baseline, (7) dF/F = (F - baseline) / |baseline|, (8) smooth dF/F with 2-sample Gaussian, (9) OASIS deconvolution with tau=0.7 and rate=15.5 Hz. However, the **saved** neural signal is dF/F, not the deconvolved events. The `keep_teleports` metadata is NOT used; the AI always computes baselines within trials only.

ii.
```python
def compute_dff(F, Fneu, starts, stops):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    f_ -= NEU_COEF * fneu_
    ...
    # baseline: nansmooth, min filter, max filter
    ...
    dff[:, mask] = (f_[:, mask] - flow[:, mask]) / np.abs(flow[:, mask])
    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
    return dff

def deconvolve(dff, starts, stops):
    spks = np.zeros(dff.shape, dtype=np.float32)
    for s, e in zip(starts, stops):
        spks[:, s:e] = dcnv.oasis(
            np.ascontiguousarray(dff[:, s:e], dtype=np.float32),
            2000, TAU, FRAME_RATE)
    return spks
...
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. The AI justified the choice of dF/F over events through a controlled comparison showing dF/F decodes better on 5/6 outputs. However, the paper's standard population analyses all use the deconvolved events (`events`), and the reference code saves events. The AI also chose not to use `keep_teleports` metadata, arguing it's irrelevant since only within-trial data is used, but `keep_teleports` affects the baseline window size and thus the dF/F values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) suite2p manual curation (`iscell[:,0]==1`), (2) exclusion of putative interneurons with Pearson r(dF/F, speed) > 0.5. The interneuron correlation is computed vectorially over within-trial samples.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
F[iscell], Fneu[iscell], plane_idx[iscell]
...
# interneuron exclusion
d = dff[:, in_trial]
sp = beh['speed'][in_trial].astype(np.float32)
dz = d - d.mean(axis=1, keepdims=True)
sz = sp - sp.mean()
denom = (np.sqrt((dz ** 2).sum(axis=1)) * np.sqrt((sz ** 2).sum()))
r_speed = (dz @ sz) / denom
keep_cells = r_speed <= INTERNEURON_R_THRESH
```

iii. Both filters match the paper's Methods: "ROIs were manually curated" and "putative interneurons were detected for exclusion ... Pearson correlation of >0.5".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since neural and behavioral data share the same time indices (behavior is already interpolated onto imaging frames in the NWB file), alignment is achieved by simply slicing from trial start to teleport.

ii.
```python
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. The NWB file contains behavior data already interpolated to imaging frames, so no resampling or shifting is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is one imaging frame = 64.48 ms (15.5078125 Hz per plane). No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 15.5078125  # Hz, imaging frame rate per plane
DT = 1.0 / FRAME_RATE   # 64.48 ms
...
'time_bin_size': 1000.0 / FRAME_RATE,  # ms
```

iii. The paper states "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior timestamps array (`beh['t']`), which comes from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
beh['t'] = b['position']['timestamps'][:]
...
tt = beh['t'][s:e] - beh['t'][s]
...
inp[0] = tt
```

iii. The timestamps are the same for all behavior time series (verified by the NWB structure).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps within the trial.

ii.
```python
tt = beh['t'][s:e] - beh['t'][s]
inp[0] = tt
```

iii. Straightforward time-from-start computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (behavior is already interpolated onto imaging frames), so the same `[s:e]` slice is used for both.

ii.
```python
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
...
tt = beh['t'][s:e] - beh['t'][s]
```

iii. The NWB file pre-aligns behavior to imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series, specifically the median value over the trial period.

ii.
```python
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
...
inp[1] = env_stream[i]
```

iii. Environment is constant within a trial (0=ENV1, 1=ENV2), so median is equivalent to any single value.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of the environment values within the trial is taken. Since environment is constant within trials, this effectively just reads the per-trial value. The value is broadcast across all timepoints.

ii.
```python
inp[1] = env_stream[i]
```

iii. No complex processing needed; environment does not change within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-based index of the trial within the session, derived from the loop counter over trials (including lick-error trials that are dropped).

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_bad[i]:
        continue
    ...
    inp[2] = i
```

iii. The trial number uses the original trial index (before filtering), so gaps may appear when lick-error trials are dropped. This preserves the temporal ordering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
inp[2] = i
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` sparse time series (timestamps of reward delivery) and the `reward_zone` behavior time series. A trial is considered rewarded if reward was delivered AND the animal entered the reward zone.

ii.
```python
rew_t = b['Reward']['timestamps'][:]
rew_bin = np.zeros_like(beh['position'])
if len(rew_t):
    idx = np.searchsorted(beh['t'], rew_t)
    idx = np.clip(idx, 0, len(rew_bin) - 1)
    rew_bin[idx] = 1
beh['reward'] = rew_bin
...
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
```

iii. This matches the reference code's `behavior.get_trial_types`: `isreward = any(reward>0) & any(rzone>0)`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the *previous* trial (using the original trial index, before filtering) is used. For the first trial of a session, the value is set to 1 (rewarded). The value is broadcast across all timepoints.

ii.
```python
inp[3] = isreward[i - 1] if i > 0 else 1
```

iii. The AI justified setting the first trial to 1 because "30 warm-up trials with the same reward zone immediately precede each imaging session, and 84.7% of trials are rewarded."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. The reward zone location comes from parsing the VR scene name (from the NWB `identifier` field) using the reference code's convention: `behavior.get_reward_zones` with zones A=[80,130], B=[200,250], C=[320,370] and a switch after 30 trials on switch sessions.

ii.
```python
def parse_scene(scene):
    m = re.match(r'^Env(\d)_Location([ABC])$', scene)
    if m:
        return m.group(2), None, int(m.group(1)) - 1, None
    ...

def zone_per_trial(scene, ntrials):
    z1, z2, e1, e2 = parse_scene(scene)
    if z2 is None:
        return [z1] * ntrials, [e1] * ntrials
    n1 = min(CHANGE_TRIAL, ntrials)
    return ([z1] * n1 + [z2] * (ntrials - n1), ...)
...
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
zone = REWARD_ZONES[zones[i]]
dist = signed_distance_to_zone(pos, zone)
```

iii. This follows the reference code's `behavior.get_reward_zones`, which determines the reward zone from the scene name and applies the 30-trial switch rule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone: 0 inside the zone, negative before, positive after.

ii.
```python
def signed_distance_to_zone(pos, zone):
    lo, hi = zone
    d = np.zeros_like(pos)
    before = pos < lo
    after = pos > hi
    d[before] = pos[before] - lo
    d[after] = pos[after] - hi
    return d
```

iii. This matches the instruction's "distance to any location in the reward zone."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic: `d < -50` -> 0, `[-50, -10)` -> 1, `[-10, 0)` -> 2, `d == 0` -> 3, `(0, 10]` -> 4, `(10, 50]` -> 5, `d > 50` -> 6.

ii.
```python
def bin_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    assert np.all(out >= 0)
    return out
```

iii. The bins match the instructions. The explicit conditions ensure exact 0 maps to bin 3 ("in zone").

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as the neural data within each trial.

ii.
```python
pos = beh['position'][s:e].astype(np.float64)
...
dist = signed_distance_to_zone(pos, zone)
out[0] = bin_distance(dist)
```

iii. All streams use the same `[s:e]` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['position'][s:e].astype(np.float64)
...
out[1] = bin_position(pos)
```

iii. Position directly records the animal's VR corridor position in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Discretized into 5 bins of 90 cm each using `np.floor(pos / 90)` clipped to [0, 4].

ii.
```python
def bin_position(pos):
    return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
```

iii. Five 90 cm bins over the 450 cm track. Clipping handles edge cases where position is slightly outside [0, 450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.floor(pos / 90)` clipped to [0, 4]: bin 0 = [0, 90), bin 1 = [90, 180), bin 2 = [180, 270), bin 3 = [270, 360), bin 4 = [360, 450].

ii.
```python
return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
```

iii. The `np.floor` approach differs slightly from the reference's `np.digitize` at exact boundary values (e.g., pos=90 maps to bin 1 with floor but also bin 1 with digitize since digitize uses right-open intervals by default with the reference's bin edges).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. Same `[s:e]` slice.

iii. Pre-aligned in NWB.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = beh['lick'][s:e].astype(np.float64)
...
out[3] = (licks > 0).astype(np.int64)
```

iii. The `lick` variable records cumulative lick counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (licks > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii. Same `[s:e]` slice.

iii. Pre-aligned in NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the VR scene name stored in the NWB `identifier` field, combined with the 30-trial switch rule from the reference code.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
...
out[4] = ZONE_TO_IDX[zones[i]]
```

iii. This follows `behavior.get_reward_zones` from the reference code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine the initial reward zone and any switch. On switch sessions (e.g., `Env1_LocationA_to_B`), the first 30 trials get zone A and the rest get zone B. Zone A=0, B=1, C=2. The value is broadcast across all timepoints.

ii.
```python
CHANGE_TRIAL = 30
...
def zone_per_trial(scene, ntrials):
    z1, z2, e1, e2 = parse_scene(scene)
    if z2 is None:
        return [z1] * ntrials, [e1] * ntrials
    n1 = min(CHANGE_TRIAL, ntrials)
    return ([z1] * n1 + [z2] * (ntrials - n1), ...)
```

iii. Matches the reference code's convention of switching after trial 30.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from both the `Reward` sparse time series (reward delivery events) and the `reward_zone` behavior time series (reward zone entry flag).

ii.
```python
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
...
out[5] = isreward[i]
```

iii. A trial is rewarded only if the animal both entered the reward zone AND received a reward, matching `behavior.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event occurred AND if the animal entered the reward zone. The value (0 or 1) is broadcast across all timepoints.

ii.
```python
out[5] = isreward[i]
```

iii. Per-trial binary output as specified.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Behavior/imaging length mismatch**: If behavior has more samples than imaging (or vice versa, up to 5 samples difference), both are truncated to the common length. This handles the "one frame correction" from the reference code.
- **Lick-sensor error trials**: Dropped (81 trials matching the paper).
- **Truncated trials**: Trials whose teleport falls beyond the truncation point are dropped (assertion checks).
- **NaN/Inf checks**: All saved arrays are asserted to be finite.

ii.
```python
nframes = min(n_beh, n_ophys)
if n_beh != n_ophys:
    assert abs(n_beh - n_ophys) <= 5, (n_beh, n_ophys)
    ...
    for k in list(beh.keys()):
        beh[k] = beh[k][:nframes]
    F = F[:, :nframes]
    Fneu = Fneu[:, :nframes]
...
assert np.all(np.isfinite(act)) and np.all(np.isfinite(inp))
```

iii. The behavior/imaging mismatch occurs in ~10 two-plane sessions and is the same "one frame correction" handled in the reference `TwoPUtils.preprocessing.vr_align_to_2P`.

## 13-a. What are the most time-consuming steps of the code?

i. According to the AI's timing information:
1. **OASIS deconvolution** (~3.5 s/session) - the most expensive per-session step
2. **dF/F computation** (0.2-4.6 s/session depending on neuron count)
3. **Loading NWB files** (0.2-0.7 s/session)
Total: ~3.4 min for all 152 sessions with 8 parallel workers.

ii. N/A

iii. The AI used multiprocessing to parallelize across sessions, reducing wall-clock time from ~15 min serial to ~3.4 min.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` (iterating over `zip(starts, stops)`) processes each trial sequentially. The lick-error computation uses a list comprehension. The dF/F baseline computation loops over trials. However, variable trial lengths make full vectorization difficult.

ii. N/A

iii. The AI vectorized the interneuron correlation computation (matrix product instead of per-cell loop).

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded only once per session (no separate survey step). However, the dF/F and deconvolution are both computed even when only dF/F is saved (deconvolution is always run for interneuron exclusion... actually, interneuron exclusion uses dF/F not events, so the deconvolution is not strictly needed when saving dF/F, but it is computed anyway).

ii.
```python
dff = compute_dff(F, Fneu, starts, stops)
...
dff_kept = dff[keep_cells]
events = deconvolve(dff_kept, starts, stops)
...
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. The deconvolution is always computed even when dF/F is saved, adding ~3.5 s/session of unnecessary work when `--neural-signal dff` (the default).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The OASIS deconvolution is always computed but discarded when `--neural-signal dff` is selected (the default). The `plot_processing` function is only run for up to 2 sessions. The per-session metadata (timings, scene, etc.) is stored in the pickle but not used by the decoder.

ii.
```python
events = deconvolve(dff_kept, starts, stops)
...
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. The deconvolution takes ~3.5 s/session and could be skipped when saving dF/F.
