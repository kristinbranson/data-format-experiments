# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing `data/sub-*/*.nwb` using `pathlib.Path`. Each file is processed individually via `process_file()`. The files are sorted by path. In `--sample` mode, only the first 2 files are used.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files
```
```python
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The AI uses `pathlib.Path.glob` to find all NWB files. This should find all session files across all subjects. The CONVERSION_NOTES.md confirms 152 sessions across 11 subjects were found.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file's `subject.subject_id` attribute, falling back to the parent directory name. Unique subjects are collected from all sessions and sorted.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI infers subjects from the NWB metadata rather than parsing directory names. The result should be equivalent since the directory names follow the `sub-<id>` pattern matching the subject IDs.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session ID is taken from `nwb.session_id`.

ii.
```python
sess_id = nwb.session_id
```

iii. This matches the data organization where each NWB file is a separate session.

## 1-d. How are the data split into trials?

i. Trials are identified using only the `trial_start` behavior time series. Each trial starts where `trial_start > 0` and ends at the next `trial_start` (or end of recording). The `teleport` signal is NOT used to define trial ends.

ii.
```python
def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds
```

iii. The AI chose to define trial boundaries using only `trial_start`, with each trial extending to the next `trial_start`. The CONVERSION_NOTES Step 4 mentions using `trial_start` as onset and "the next `trial_start` (or `teleport`/end-of-recording for the final trial) as trial end," but the actual code only uses the next `trial_start`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if they have fewer than 2 timepoints, if the max `trial_num` within the trial is < 0 (baseline), or if all positions are <= -100 (sentinel values).

ii.
```python
if e - s < 2:
    continue
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue
```

iii. The AI applies three filters: minimum length (2 timepoints), baseline exclusion, and sentinel position exclusion. Additionally, sessions with fewer than 2 trials after filtering are excluded from the final dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field, specifically only `plane0`.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The AI explicitly accesses only `plane0` of the deconvolved data, rather than iterating over all planes. The CONVERSION_NOTES Step 5 maps neural data to `processing['ophys']['Deconvolved'].roi_response_series['plane0']`.

## 2-b. How is the `neural` data processed?

i. The neural data is loaded directly as float32, transposed from (time, neurons) to (neurons, time) per trial. No filtering by `iscell` is applied. No multi-plane concatenation is performed.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The AI loads deconvolved activity and transposes it. It does not filter by `iscell` and does not handle multi-plane recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural data filtering is applied. The `iscell` variable is not used.

ii.
```python
# No iscell filtering code exists in the AI's convert_data.py
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The CONVERSION_NOTES do not mention `iscell` filtering. The AI reports 260,091 total neurons, which is much higher than what would be obtained after `iscell` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data using the trial boundary indices. Since the trial boundaries are defined from `trial_start`, the neural data naturally starts at trial onset.

ii.
```python
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The alignment is implicit from the trial segmentation. Neural and behavioral data share the same time indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The time bin size is computed from the deconvolved data's stored rate as `1000.0 / rate`. The AI does not account for multi-plane recordings in computing the effective time bin size.

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
...
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. The time bin size is derived from the stored sampling rate. Since the AI only reads `plane0`, the rate may not need multi-plane correction — but this depends on whether the stored rate already accounts for interleaved plane scanning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `deconvolved` data's timestamps (or synthesized from the rate if timestamps are not available).

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
...
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The AI uses neural timestamps rather than behavioral timestamps. Since both should share the same time base, this should produce equivalent results.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the start of the trial is subtracted from all timestamps within the trial.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. Standard approach to compute relative time within trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamps used are from the neural data directly (deconvolved timestamps), so alignment is inherent.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
```

iii. Since timestamps come from the neural data source, neural-input alignment is guaranteed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = np.asarray(behavior['environment'][0]).ravel()
...
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
```

iii. The `environment` variable from the NWB behavior data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median of valid (>= 0) environment values within the trial and rounds to get a single integer label. Values of -1 (baseline) are excluded. If no valid values exist, defaults to 0.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
inp = np.vstack([
    rel_time,
    np.full(e - s, env_label, dtype=np.float32),
    ...
])
```

iii. The median approach adds robustness but adds unnecessary complexity since environment should be constant within a valid trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file, not the loop counter.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The AI uses the stored `trial number` variable from NWB, taking the median of valid values. Falls back to the enumeration index `ti` if no valid values exist.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Takes the median of valid (>= 0) trial number values within the trial. The value is broadcast as a constant across all timepoints.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
...
np.full(e - s, tr_label, dtype=np.float32),
```

iii. Using the stored trial number rather than the sequential index may produce different values if trial numbering in the NWB file differs from the sequential order.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
...
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
```

iii. The AI computes reward outcome per trial from reward event timestamps, then uses the previous trial's outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward is determined by whether any `Reward` timestamp falls within the trial's time window. The previous trial's reward outcome is used as the input. For the first trial, defaults to 0.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
...
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
```

iii. The approach is functionally similar to the reference, though it uses timestamp comparison rather than index-based lookup. The `1e-9` epsilon handles floating-point edge cases.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavior time series. The reward zone identity is determined per trial using `reward_zone_centers()` and `collapse_zone_position_to_abc()`.

ii.
```python
def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        pos = pos[np.isfinite(pos)]
        if len(pos):
            centers[int(code)] = float(np.median(pos))
    return centers

def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0
    if pos < 260:
        return 1
    return 2
```

iii. The AI computes per-session median positions for each reward zone code, then collapses to A/B/C using position thresholds.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The distance is computed as `position - zone_center`, where `zone_center` is a fixed center point for each zone label (A=90, B=205, C=325). This is distance to a single center point, NOT distance to the nearest edge of a reward zone range.

ii.
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. The AI uses distance to a single center point. The reference uses signed distance to the nearest edge of a reward zone range (e.g., [80, 130] for zone A), where distance is 0 when inside the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using a per-element function `distance_bin()` with thresholds matching the instructions: < -50, -50 to -10, -10 to 0, exactly 0, 0 to 10, 10 to 50, > 50.

ii.
```python
def distance_bin(d):
    if d < -50:
        return 0
    if d < -10:
        return 1
    if d < 0:
        return 2
    if d == 0:
        return 3
    if d <= 10:
        return 4
    if d <= 50:
        return 5
    return 6
```

iii. The bin boundaries match the instruction specification. However, since the distance is computed differently (to center vs. to zone edges), the bin assignments will differ from the reference.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The position data and neural data share the same time indices within each trial, so no additional alignment is needed.

ii.
```python
d = position[s:e] - zc
```

iii. Same indexing as neural data ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
...
out_pos = pos_bins(position[s:e])
```

iii. Directly from the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the range [0, 450) and then digitized into 5 equal bins using `np.linspace(0, 450, 6)` which produces edges at [0, 90, 180, 270, 360, 450].

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The AI creates 5 equal bins spanning [0, 450] cm. This differs from the reference which uses bins [-inf, 50, 150, 250, 350, inf] spanning approximately [-50, 450] cm with 100 cm wide bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins with edges at [0, 90, 180, 270, 360, 450] cm, each 90 cm wide. Position is clipped to [0, 450).

ii.
```python
edges = np.linspace(lo, hi, 6)  # [0, 90, 180, 270, 360, 450]
out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
```

iii. The bin edges differ from the reference's [-inf, 50, 150, 250, 350, inf]. The clipping to [0, 450) also differs from the reference which does not clip.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as the neural data within each trial.

ii.
```python
out_pos = pos_bins(position[s:e])
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
...
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Matches the instructions for binary lick output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as the neural data within each trial.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series. The reward zone code is mapped to a physical position using median position, then collapsed to A/B/C using position thresholds.

ii.
```python
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ridx = min(ridx, len(position) - 1)
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The AI uses per-session median positions of reward zone codes to determine physical location, then maps to A/B/C. For trials without reward zone data, it falls back to reward position or median trial position.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Three-step process: (1) find the most common reward zone code in the trial, (2) look up its median position from session-level statistics, (3) collapse position to A/B/C using thresholds (< 150 = A, < 260 = B, >= 260 = C). Fallbacks handle missing data.

ii.
```python
def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0
    if pos < 260:
        return 1
    return 2
```

iii. The approach is simpler than the reference's Viterbi algorithm but should produce similar results for most trials since reward zone positions cluster near the expected ranges.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. Reward events are identified by their timestamps within the trial's time window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked as rewarded (1) if any `Reward` timestamp falls within the trial's time window [t0, t1 + 1e-9], otherwise 0. The value is constant across the trial.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
...
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. The timestamp comparison approach is functionally equivalent to the reference's index-based approach. The epsilon prevents missing events at exact boundaries.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Short trials**: Trials with fewer than 2 timepoints are skipped.
- **Baseline trials**: Trials where `trial_num` is all < 0 are skipped.
- **Sentinel positions**: Trials where all positions are <= -100 are skipped.
- **Missing reward zone**: Falls back to reward position or median trial position.
- **Missing timestamps**: Synthesized from rate if deconvolved timestamps are None.
- **Missing environment**: Defaults to 0 if no valid environment values.
- **Sessions with < 2 trials**: Excluded from final dataset.

ii.
```python
if e - s < 2:
    continue
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue
...
if len(sess['neural']) >= 2:
    sessions.append(sess)
```

iii. The AI applies several defensive checks. However, it does NOT handle the neural/behavior length mismatch that the reference handles by cropping to the minimum.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and reading NWB files (`NWBHDF5IO` and reading large arrays). The code processes files sequentially, with timing printed for each session.

ii.
```python
for i, f in enumerate(files, 1):
    st = time.time()
    sess = process_file(f)
    ...
    print(f'processed {i}/{len(files)} {f.name}: ... time={time.time()-st:.2f}s', flush=True)
```

iii. The CONVERSION_NOTES estimate ~0.7 s/session, totaling ~2-3 minutes for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `distance_bin()` and `speed_bin()` functions are applied element-wise using list comprehensions, which could be vectorized using `np.digitize`.

ii.
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. These element-wise Python loops are slower than vectorized numpy operations like `np.digitize` used in the reference.

## 13-c. What processing does the code repeat multiple times?

i. The code does a single pass over all files, so there is no repeated processing of the same data. However, `reward_zone_centers()` is computed per session inside `process_file()`, which means it's computed once per session rather than once globally.

ii.
```python
zone_centers = reward_zone_centers(position, reward_zone)
```

iii. Each NWB file is only opened and read once, which is more efficient than the reference's two-pass approach (survey + conversion).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads all behavior time series including `autoreward` and `scanning` which are not used in the final output.

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
```

iii. Loading `autoreward` and `scanning` is wasteful since they are constant (all zeros and all ones respectively) and not used.
