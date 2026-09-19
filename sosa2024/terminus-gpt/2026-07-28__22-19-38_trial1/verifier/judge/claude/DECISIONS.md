# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files found by globbing `data/sub-*/*.nwb`, sorted alphabetically. Each file is opened with `pynwb.NWBHDF5IO`. From each file, it reads: the `Deconvolved` neural activity from `processing['ophys']` (specifically `plane0` only), and all behavior time series from `processing['behavior']['BehavioralTimeSeries']`. In `--sample` mode, only the first 2 files are processed.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

# In process_file:
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
    bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
    neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES document that NWB files are organized under per-subject directories and that each file corresponds to a session. The AI chose to read `Deconvolved` activity because it noted "methods use deconvolved activity for remapping and GLM analyses."

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id` or the parent directory name of each NWB file. Unique subjects are collected across all sessions and sorted.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
# ...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI relies on the NWB metadata or file path to determine the subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions with fewer than 2 trials after filtering are excluded.

ii.
```python
for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```

iii. The AI noted that the data organization has one file per session and that at least 2 trials are needed for decoder evaluation.

## 1-d. How are the data split into trials?

i. Trials are identified by finding indices where `trial_start > 0`. Each trial spans from one `trial_start` index to the next `trial_start` index (or end of recording for the last trial). This means the inter-trial teleport period is included in each trial's data.

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

iii. The AI's CONVERSION_NOTES say "Use `trial_start` as trial onset and `teleport`/next `trial_start` to delimit trial end", but the actual code only uses the next `trial_start` and does not use the `teleport` signal to end trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered out if: (1) they have fewer than 2 timepoints (`e - s < 2`), (2) all trial numbers are negative (`np.nanmax(trial_num[s:e]) < 0`), or (3) all positions are <= -100 cm. Sessions with < 2 remaining trials are also excluded.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    if e - s < 2:
        continue
    if np.nanmax(trial_num[s:e]) < 0:
        continue
    if np.all(position[s:e] <= -100):
        continue
```

iii. The AI's CONVERSION_NOTES mention "Baseline exclusion: Exclude pre-task baseline samples where trial number/environment are -1 or position is sentinel-valued."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the `Deconvolved` data interface from `processing['ophys']`, specifically the `plane0` roi_response_series. This is suite2p's own deconvolution stored in the NWB file, NOT the paper's custom dF/F + OASIS deconvolution pipeline.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES state "Neural signal used in reference text is deconvolved activity" and "Use deconvolved activity as primary neural signal." The AI identified that the NWB contains `Deconvolved`, `Fluorescence`, and `Neuropil` interfaces but chose to use `Deconvolved` directly, apparently not recognizing that the paper computes its own deconvolved signal from `Fluorescence` and `Neuropil`.

## 2-b. How is the `neural` data processed?

i. No processing is applied. The AI reads the `Deconvolved` data directly and casts it to float32. The paper's processing pipeline (neuropil subtraction, maximin baseline dF/F, Gaussian smoothing, OASIS deconvolution with tau=0.7) is not implemented.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
# ...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The AI did not implement any neural processing because it assumed the `Deconvolved` NWB variable was the correct signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The AI does not check `iscell` (suite2p's manual curation flag) and does not remove putative interneurons based on speed correlation.

ii.
```python
# No filtering code - all neurons from plane0 are included
neural = np.asarray(deconv.data[:], dtype=np.float32)
n_neurons = neural.shape[1]
```

iii. The AI's CONVERSION_NOTES mention neuron curation rules from the paper (SI significance, FDE thresholds) but these are not implemented. The notes do not mention `iscell` or interneuron filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is sliced by the trial start and end indices, so alignment to trial start is automatic.

ii.
```python
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Since the instruction says to align to start of trial, and the data is sliced starting from `trial_start`, this is correct by construction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native sampling rate from the deconvolved data without any rebinning. The time bin size is computed as `1000.0 / rate`. However, only `plane0` is read, so for multi-plane sessions (m17, m18 which have 2 planes), the reported rate may be the scanner rate rather than the per-plane rate, potentially yielding an incorrect time bin size.

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
# ...
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. The AI notes the sampling rate is ~15.5 Hz giving ~64.5 ms bins. No rebinning is applied.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `Deconvolved` neural data timestamps (or computed from `rate` if timestamps are unavailable).

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
# ...
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The AI uses neural timestamps since they share the same time base as behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the start of the trial is subtracted from all timestamps within the trial.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. Standard approach to compute relative time within a trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same timestamps array is used for both neural data and time computation, so they are inherently aligned.

ii.
```python
# Both use the same [s:e] indices
trial_neural = neural[s:e, :].T.astype(np.float32)
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. Neural and behavior data share the same sampling grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = np.asarray(behavior['environment'][0]).ravel()
# ...
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
```

iii. The AI identified `environment` as containing binary 0/1 task codes with -1 for baseline periods.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI filters out negative values (baseline), takes the median of valid values, rounds, and casts to int. This produces a single per-trial constant that is broadcast across all timepoints.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
inp = np.vstack([
    rel_time,
    np.full(e - s, env_label, dtype=np.float32),
    # ...
])
```

iii. The AI uses median to handle potential mixed values within a trial. Since environment is constant per trial in valid data, this should yield the correct value, but could be problematic if teleport-period samples (where environment = -1) are included in the trial boundaries.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series stored in the NWB file.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
# ...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The AI uses the NWB stored trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Filters out negative trial numbers, takes the median of valid values, and uses the result as a per-trial constant broadcast across timepoints. Falls back to the trial enumeration index if no valid values exist.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
np.full(e - s, tr_label, dtype=np.float32),
```

iii. The AI chose to use the stored trial number from the NWB file as it represents the experimental trial index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
```

iii. Reward events have their own timestamps distinct from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome is computed by checking if any reward timestamp falls within the trial's time range. The previous trial's outcome is then used as this trial's input. For the first trial, the value is 0.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
# ...
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
```

iii. The AI accumulates reward outcomes for all processed trials and indexes into this list to get the previous trial's outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and a reward zone center position. The zone center is determined from the `reward_zone` behavior time series (taking the median position of the animal during active reward zone periods), or from the reward event position, or from the median position of the trial. The zone is classified as A/B/C using simple position thresholds.

ii.
```python
def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        centers[int(code)] = float(np.median(pos))
    return centers

def collapse_zone_position_to_abc(pos):
    if pos < 150: return 0
    if pos < 260: return 1
    return 2
```

iii. The AI inferred reward zone positions from the data and used simple position thresholds to assign A/B/C labels.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes distance as `position - zone_center`, where `zone_center` is a fixed lookup value per zone (A=90, B=205, C=325). This is distance from a single center point, not distance to the nearest edge of the reward zone range. The value is 0 only when the animal is exactly at the center position, not when it is anywhere inside the zone.

ii.
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. The AI used center points rather than zone edge ranges.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using a per-element function with if/else thresholds matching the instruction's bin edges.

ii.
```python
def distance_bin(d):
    if d < -50: return 0
    if d < -10: return 1
    if d < 0: return 2
    if d == 0: return 3
    if d <= 10: return 4
    if d <= 50: return 5
    return 6
```

iii. The bin edges match the instruction specification, though applied to center-based distance rather than edge-based distance.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The same trial indices (`s:e`) are used for position and neural data, so alignment is automatic.

ii.
```python
d = position[s:e] - zc
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Both use the same time indices within each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
out_pos = pos_bins(position[s:e])
```

iii. Direct use of the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to [0, 450) and then discretized into 5 equal-sized bins using `np.digitize` with edges from `np.linspace(0, 450, 6)`.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The 450 cm track is divided into 5 bins of 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Bins are [0, 90), [90, 180), [180, 270), [270, 360), [360, 450), matching the instruction's 5 equal-sized bins.

ii.
```python
edges = np.linspace(lo, hi, 6)  # [0, 90, 180, 270, 360, 450]
out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
```

iii. Equal-sized bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
out_pos = pos_bins(position[s:e])
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Both use the same trial slice.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Direct use of the lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Matches the instruction's binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Both use the same trial slice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series, `position`, and `Reward` timestamps. The reward zone code is determined per trial, then its median position is used to classify into A (< 150 cm), B (150-260 cm), or C (> 260 cm).

ii.
```python
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[...][0])
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The AI uses position-based heuristics to determine which zone the animal is in, rather than using known zone boundaries.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial: (1) if the reward_zone variable has nonzero values, find the most common code and look up its median position across the session; (2) if no active zone but reward occurred, use the position at the reward timestamp; (3) otherwise use the median position of the trial. The position is then classified as A/B/C using fixed thresholds (< 150, < 260, else). The default for non-finite positions is 225 (zone B).

ii. See 10-a code snippet.

iii. The AI's approach is a heuristic that attempts to robustly determine the zone even when reward_zone data is missing.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. Reward events have separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time range [t0, t1 + 1e-9]. The result (0 or 1) is broadcast as a constant across all timepoints.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. Binary per-trial output matching the instruction.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled: (1) trials with < 2 timepoints are skipped; (2) trials where all trial numbers are negative (baseline periods) are skipped; (3) trials where all positions are <= -100 are skipped; (4) sessions with < 2 trials are excluded; (5) reward zone falls back to position-based heuristics when no active zone data exists; (6) environment defaults to 0 if no valid values; (7) trial number falls back to enumeration index.

ii.
```python
if e - s < 2: continue
if np.nanmax(trial_num[s:e]) < 0: continue
if np.all(position[s:e] <= -100): continue
# ...
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The AI implemented defensive checks for missing or invalid data. However, neural/behavior length mismatches are not handled (no cropping to minimum length).

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files is the most time-consuming step. The AI estimates ~0.7 seconds per session, for a total of ~2-3 minutes for all 152 sessions.

ii. N/A (profiling information from CONVERSION_NOTES)

iii. NWB files are large and I/O bound.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `distance_bin` and `speed_bin` functions use per-element Python loops with list comprehensions instead of vectorized `np.digitize`. This is the main vectorization opportunity.

ii.
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. These could be replaced with `np.digitize` for significant speedup on large arrays.

## 13-c. What processing does the code repeat multiple times?

i. The code does a single pass over all NWB files, so there is no repeated loading. This is more efficient than the reference solution which loads files twice (survey + conversion).

ii. N/A

iii. Single-pass design avoids redundant I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `zone_centers` for all reward zone codes across the entire session, even though only the per-trial zone assignment is needed. The `infer_region` function attempts to read imaging plane metadata that may not be needed if all sessions are CA1. The `speed_bin` function clamps negative speeds to 0, which is unnecessary since negative speeds would end up in bin 0 anyway.

ii.
```python
zone_centers = reward_zone_centers(position, reward_zone)
region = infer_region(nwb)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. These are minor inefficiencies that don't significantly impact runtime.
