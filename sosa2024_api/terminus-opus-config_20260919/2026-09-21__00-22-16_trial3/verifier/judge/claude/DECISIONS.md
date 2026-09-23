# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files using `sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))`. Each NWB file is loaded with `pynwb.NWBHDF5IO`. Processing is parallelized across sessions using `ProcessPoolExecutor` with a spawn context. Each session is fully self-contained in one NWB file.

ii.
```python
files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))
# ...
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The AI's CONVERSION_NOTES document that there are 152 NWB files across 11 subjects. The glob pattern finds all NWB files in all subject directories. pynwb is used as required by the instructions.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file metadata (`nwb.subject.subject_id`). The unique subject IDs are collected from all processed sessions and sorted.

ii.
```python
subject = nwb.subject.subject_id
# ...
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The subject ID is read directly from the NWB metadata, which is the canonical source.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from `nwb.session_id`.

ii.
```python
session_id = nwb.session_id
```

iii. The one-file-per-session structure is consistent with the DANDI dataset organization.

## 1-d. How are the data split into trials?

i. Trials are identified by the `trial_start` and `teleport` behavior time series. Trial starts are frames where `trial_start > 0`. Trial ends are frames where `teleport > 0`. Each trial spans `[start, stop)`.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
# ...
assert len(stops) == n_trials_raw, 'trial_start/teleport count mismatch'
assert np.all(stops > starts), 'teleport before trial_start'
```

iii. The CONVERSION_NOTES verify that `trial_start` and `teleport` counts match in all 152 sessions. Positions at trial-start frames are ~0-5 cm and at teleport-1 are ~445-450 cm, confirming the boundaries are correct.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies five filters: (1) drop the first trial of each session (previous trial outcome undefined), (2) drop lick-sensor-error trials (>30% of frames with cumulative lick count >2), (3) drop trials with fewer than 10 frames, (4) drop trials with non-finite neural/behavioral values, (5) drop trials where scanning != 1. This results in 11,983 kept trials from 12,216 raw trials.

ii.
```python
if i == 0:
    n_drop_first += 1
    continue
if lick_error[i]:
    n_drop_lick += 1
    continue
if (e - s) < MIN_TRIAL_FRAMES:
    n_drop_short += 1
    continue
if np.any(scanning[s:e] != 1):
    n_drop_scan += 1
    continue
# ...
if (np.any(~np.isfinite(ev)) or np.any(~np.isfinite(p)) or
        np.any(~np.isfinite(sp)) or np.any(~np.isfinite(lk))):
    n_drop_nan += 1
    continue
```

iii. The lick-error criterion exactly reproduces the paper's 81 dropped trials. Dropping the first trial is justified because `prev_trial_outcome` is undefined. The other filters are defensive checks (0 trials dropped by short/nan/scanning in practice).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB ophys processing module. NOT the stored `Deconvolved` field.

ii.
```python
oph = nwb.processing['ophys'].data_interfaces
# ...
Fp = np.asarray(oph['Fluorescence'].roi_response_series[name].data[:]).T
Np = np.asarray(oph['Neuropil'].roi_response_series[name].data[:]).T
```

iii. The CONVERSION_NOTES explain that the NWB's `Deconvolved` array is suite2p's own deconvolution of raw fluorescence, not the signal the paper analyzes. The paper computes its own dF/F and events from F and Fneu.

## 2-b. How is the `neural` data processed?

i. The AI re-implements the paper's dF/F pipeline: within-trial masking (NaN outside trials), neuropil subtraction (coefficient 0.7) with per-trial neuropil mean added back, per-trial maximin baseline (Gaussian smooth sigma=15, min filter 300 frames, max filter 300 frames), dF/F = (F - baseline)/|baseline|, Gaussian smoothing with sigma=2 frames. OASIS deconvolution is also computed but NOT used as the final neural signal. The final signal is **dF/F**, not the deconvolved events.

ii.
```python
def compute_dff_and_events(F, Fneu, starts, stops):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    f_ -= NEU_COEF * fneu_
    # ... per-trial baseline: gaussian -> min_filter -> max_filter
    dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
    # ... smooth and OASIS
    events[:, s:e] = dcnv.oasis(...)
    return dff, events
# ...
if signal == 'dff':
    events = dff  # uses dF/F as the neural signal
```

iii. The AI tested both dF/F and OASIS events and found dF/F gave better decoder accuracy for every output. The AI justified this by noting that dF/F preserves amplitude information that the decoder can exploit, while OASIS discards it. However, the paper uses events for its decoder. The baseline window always restricts to within-trial only (no `keep_teleports` logic is implemented).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) only `iscell==1` ROIs are kept (suite2p's manual curation), (2) putative interneurons are excluded based on Pearson correlation between dF/F and running speed exceeding 0.5. Additionally, ROIs with non-finite speed correlations (zero-variance, dead) are excluded.

ii.
```python
iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
# ...
F_list.append(Fp[keep])
# ...
r_speed = (dff_c @ sp_c) / denom
keep_cells = ~(r_speed > SPEED_CORR_THR)
keep_cells &= np.isfinite(r_speed)
```

iii. This matches the paper's methods: "putative interneurons" with Pearson correlation >0.5 with running speed are excluded. The AI reports 0.29% of cells excluded as interneurons, consistent with the paper's 0.42 +/- 0.85%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start. Neural and behavioral data share the same imaging frame clock in the NWB, so no additional alignment is needed. Per-trial data is simply sliced from `starts[i]` to `stops[i]`.

ii.
```python
ev = events[:, s:e]  # s = starts[i], e = stops[i]
```

iii. The NWB stores behavior interpolated onto the imaging frame clock, so all streams are inherently aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate: 64.48 ms (15.5078 Hz). No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 15.5078125  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # 64.48 ms
```

iii. The reference analyses operate at the imaging frame rate. The CONVERSION_NOTES confirm that all behavior timestamps have a consistent dt of 0.0644836 s.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior `position` timestamps.

ii.
```python
t = np.asarray(beh['position'].timestamps[:])
# ...
tt = t[s:e] - t[s]
inp[0] = tt
```

iii. All behavior time series share the same timestamps (verified in CONVERSION_NOTES).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first frame in the trial is subtracted from all timestamps in the trial.

ii.
```python
tt = t[s:e] - t[s]
inp[0] = tt
```

iii. Straightforward: time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (imaging frame clock), so no additional alignment is needed.

ii. Same frame indices `s:e` are used for both neural and behavioral data.

iii. Verified by the NWB structure where behavior is interpolated onto the imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = g('environment')
# ...
env_vals = np.unique(env[s:e])
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
```

iii. The environment variable is 0 for ENV1 and 1 for ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique environment value within the trial is taken. If there is more than one unique value (shouldn't happen within a trial), the median is used.

ii.
```python
env_vals = np.unique(env[s:e])
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
```

iii. Environment is constant within a trial, so this effectively just reads the value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index from the enumeration loop.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    # ...
    inp[2] = i
```

iii. The raw trial index `i` is used. Note that since trial 0 is dropped, the minimum trial number in the output is 1.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the loop index is used directly. It is constant across all timepoints within a trial.

ii.
```python
inp[2] = i
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps and the `reward_zone` series. A trial is considered rewarded if a reward was delivered AND the animal was in the reward zone during that trial.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
reward_idx = np.searchsorted(t, reward_times)
# ...
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
# ...
inp[3] = rewarded[i - 1]
```

iii. The reward outcome requires both conditions (reward delivery AND zone occupancy), matching the paper's `behavior.get_trial_types` definition of `isreward`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the previous trial (i-1) is used. The first trial of each session is dropped because there is no previous trial.

ii.
```python
if i == 0:
    n_drop_first += 1
    continue
# ...
inp[3] = rewarded[i - 1]
```

iii. The value is constant across all timepoints in the trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. The reward zone location is determined from the NWB `identifier` (which encodes the scene name, e.g. `Env1_LocationA_to_C`), parsed with `scene_reward_zones()` which re-implements `behavior.get_reward_zones`.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = scene_reward_zones(scene, n_trials_raw)
# ...
zlab = zone_labels[i]
z0, z1 = REWARD_ZONES[zlab]
d = signed_distance_to_zone(p, z0, z1)
```

iii. The scene-derived zone labels were validated against the `reward_zone` data stream on every rewarded trial across all 152 sessions with 0 mismatches (documented in CONVERSION_NOTES).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest edge of the reward zone: negative before the zone, 0 inside, positive after.

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

iii. Matches the instruction spec of "distance to any location in the reward zone."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic:
- 0: d < -50
- 1: -50 <= d < -10
- 2: -10 <= d < 0
- 3: d == 0 (inside zone)
- 4: 0 < d <= 10
- 5: 10 < d <= 50
- 6: d > 50

ii.
```python
def discretize_distance(d):
    b = np.full(d.shape, 3, dtype=np.int64)  # default: inside zone
    b[(d < 0) & (d >= -10)] = 2
    b[(d < -10) & (d >= -50)] = 1
    b[d < -50] = 0
    b[(d > 0) & (d <= 10)] = 4
    b[(d > 10) & (d <= 50)] = 5
    b[d > 50] = 6
    return b
```

iii. Matches the instruction spec bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices used for both neural and position data. No additional alignment.

ii.
```python
p = pos[s:e]
d = signed_distance_to_zone(p, z0, z1)
```

iii. All streams share the imaging frame clock.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = g('position')
# ...
p = pos[s:e]
out[1] = np.digitize(p, POS_EDGES)
```

iii. The position variable records the animal's location on the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
p = pos[s:e]
out[1] = np.digitize(p, POS_EDGES)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins using `np.digitize` with edges [90, 180, 270, 360].

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
out[1] = np.digitize(p, POS_EDGES)
```

Note: `np.digitize` with these edges produces bins 0-4 directly (no -1 offset needed since there's no -inf edge).

iii. Five 90 cm bins spanning the 450 cm track, matching the instruction spec.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices. No additional alignment needed.

ii. Same `s:e` indexing.

iii. Shared imaging frame clock.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = g('lick')
# ...
lk = lick[s:e]
```

iii. The lick variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0. Additionally, trials with lick-sensor errors (>30% of frames with cumulative count >2) are dropped entirely.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
# ...
lick_error[i] = (lick[s:e] > 2).mean() > LICK_ERROR_FRAC
```

iii. The paper NaN's lick data on error trials; since the target format forbids NaNs and lick is a decoder output, the AI drops these trials instead.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices. No additional alignment needed.

ii. Same `s:e` indexing.

iii. Shared imaging frame clock.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field which encodes the scene name (e.g. `Env1_LocationA_to_C`). The function `scene_reward_zones()` parses the scene name to determine the reward zone (A, B, or C) per trial, with a switch at trial index 30 for switch sessions.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = scene_reward_zones(scene, n_trials_raw)

def scene_reward_zones(scene, n_trials, change_trial=CHANGE_TRIAL):
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return np.array([m.group(1)] * n_trials, dtype='<U1')
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        labels = np.array([m.group(1)] * n_trials, dtype='<U1')
        labels[change_trial:] = m.group(2)
        return labels
```

iii. This re-implements the reference code's `behavior.get_reward_zones`. Validated with 0 mismatches against the recorded `reward_zone` stream across all 152 sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A/B/C) is mapped to an integer (0/1/2). The value is constant across all timepoints in the trial.

ii.
```python
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
out[4] = ZONE_TO_IDX[zlab]
```

iii. Straightforward mapping.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps and the `reward_zone` series. A trial is rewarded if a reward was delivered AND the animal was in the reward zone.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
reward_idx = np.searchsorted(t, reward_times)
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. The dual condition (reward AND in_zone) matches the paper's `get_trial_types` definition.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if the trial was rewarded (reward delivered while in the reward zone), 0 otherwise. Constant across all timepoints in the trial.

ii.
```python
out[5] = rewarded[i]
```

iii. The rewarded fraction is 0.847, consistent with the paper's ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior frame count mismatch**: 10 multi-plane sessions (m17/m18) have one extra imaging frame vs. behavior timestamps. All streams are truncated to the common length.
- **Lick-sensor errors**: Trials with >30% of frames having cumulative lick count >2 are dropped (81 trials, matching the paper exactly).
- **Short trials**: Trials with <10 frames dropped (0 in practice).
- **Non-finite values**: Trials with NaN/inf in neural or behavioral data dropped (0 in practice).
- **Non-scanning frames**: Trials where scanning != 1 are dropped (0 in practice).

ii.
```python
n_frames = min(F.shape[1], len(t))
if F.shape[1] != n_frames:
    F = F[:, :n_frames]
    Fneu = Fneu[:, :n_frames]
```

iii. The frame-count mismatch was discovered during the full conversion and documented in CONVERSION_NOTES.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading ophys data** from NWB files (I/O bound, ~0.4-1.0 s/session)
2. **dF/F + OASIS computation** (~1.5-2.5 s/session, up to ~10 s for large m18 sessions)
3. Total ~2-10 s per session. With 12 parallel workers, all 152 sessions complete in ~127 s.

ii. N/A (timing is printed in the output logs)

iii. The AI implemented parallel processing across sessions to keep total time well under the 15-minute budget.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for building input/output arrays iterates over each trial sequentially. The dF/F baseline computation loops over trials (required because baselines are per-trial). The putative interneuron speed correlation is vectorized (matrix multiply instead of per-cell loop).

ii. N/A

iii. Variable trial lengths make full vectorization awkward. The key optimization is parallelization across sessions.

## 13-c. What processing does the code repeat multiple times?

i. The code does not repeat processing — each session is processed once. There is no separate survey step that re-loads files (unlike the reference solution which has a survey pass and a conversion pass).

ii. N/A

iii. The AI's code is a single-pass design (one load per NWB file).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes OASIS-deconvolved events (`compute_dff_and_events` returns both `dff` and `events`) but uses dF/F as the final neural signal (when `--signal dff` is used, which is the default). The deconvolution step is wasted computation when using dF/F. Additionally, the `make_processing_plot` function plots the events even though they are not used.

ii.
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
# ...
if signal == 'dff':
    events = dff  # overwrite events with dF/F
```

iii. The events computation is retained for comparison purposes (`--signal events` flag) but is unnecessary for the default conversion.
