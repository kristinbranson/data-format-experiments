# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `/app/data/sub-*/*.nwb` using `pynwb.NWBHDF5IO`. It finds all subject directories matching `sub-*`, then all `.nwb` files within each. Each NWB file is read to extract neural (Deconvolved), behavioral time series, and segmentation data. All data streams are extracted within a single `load_nwb_session()` function.

ii.
```python
def load_nwb_session(path):
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        ophys = nwb.processing['ophys']
        ...
        return out

files = sorted(Path('/app/data').glob('sub-*/*.nwb'))
```

iii. The AI uses glob pattern matching to find all NWB files. This covers all 11 subjects and 152 sessions, matching the NWB file inventory.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-m11`). A `subject_to_idx` dictionary maps subject names to indices.

ii.
```python
subj = sess['subject']  # path.parent.name
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_to_idx[subj])
```

iii. Subject names come from directory structure. The AI finds 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in sorted file order.

ii.
```python
files = sorted(Path('/app/data').glob('sub-*/*.nwb'))
for path in files:
    sess = load_nwb_session(path)
    ...
```

iii. The one-file-per-session structure is consistent with the NWB organization.

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial number` behavioral time series. Valid frames are those with `trial number >= 0`, `scanning == True`, and finite position/speed. Within each trial, the code aligns to the first `trial_start` pulse. Trials with fewer than 2 valid frames are dropped.

ii.
```python
trnum = np.asarray(sess['trial number']).astype(int)
tstart = np.asarray(sess['trial_start']) > 0
scanning = np.asarray(sess['scanning']) > 0
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
trial_ids = np.unique(trnum[valid])
...
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
    if len(idx) < 2:
        continue
    start_candidates = idx[tstart[idx]]
    start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
    idx = idx[idx >= start_idx]
```

iii. The AI uses `trial number` to group frames into trials rather than using `trial_start`/`teleport` boundaries as in the reference. It also refines start alignment using the `trial_start` pulse within each trial group.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 valid timepoints are dropped. Valid timepoints require `trial number >= 0`, `scanning > 0`, finite position, and finite speed. Sessions with fewer than 2 trials are also dropped.

ii.
```python
if len(idx) < 2:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. The minimum trial length threshold of 2 is much lower than the reference's 50 timepoints. The `scanning` and `isfinite` filters add additional quality control not in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB `Deconvolved` ROI response series in the `ophys` processing module. This is suite2p's own deconvolution of raw fluorescence.

ii.
```python
deconv = next(iter(ophys['Deconvolved'].roi_response_series.values()))
neural = np.asarray(deconv.data[:], dtype=np.float32)  # time x roi
```

iii. The AI chose to use the pre-computed `Deconvolved` data from NWB, noting that methods mention "deconvolved activity matrices." However, the paper actually computes its own deconvolution from raw Fluorescence and Neuropil using a custom `preprocessing.dff()` pipeline, which produces different results than suite2p's built-in deconvolution.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The AI reads the `Deconvolved` data directly and transposes it to (n_neurons, n_timepoints). It is cast to float32.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)  # time x roi
...
neu = neural[idx].T.astype(np.float32)
```

iii. The AI assumed the NWB `Deconvolved` data was the signal the paper analyzed. In reality, the paper computes dF/F from raw Fluorescence and Neuropil traces with neuropil subtraction (coefficient 0.7), maximin baseline correction (300-sample window), Gaussian smoothing (sigma=2), and OASIS deconvolution (tau=0.7) - a substantially different signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using `iscell > 0.5` from the ROI segmentation table. No interneuron filtering is applied.

ii.
```python
iscell = sess['iscell'] > 0.5
neural = neural[:, iscell]
```

iii. The `iscell` filter follows suite2p convention. However, the reference also filters putative interneurons (cells with dF/F-speed correlation > 0.5), which the AI does not do.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start by slicing out frames belonging to each trial (via `trial number` grouping) and starting from the first `trial_start` pulse within the trial.

ii.
```python
start_candidates = idx[tstart[idx]]
start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
idx = idx[idx >= start_idx]
trial_t = timestamps[idx] - timestamps[start_idx]
```

iii. Alignment to trial start is consistent with the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native frame rate is used with no rebinning. The time bin size is computed as the median of 1000/rate across sessions, approximately 64.5 ms (~15.5 Hz).

ii.
```python
rate = float(np.median(1.0 / np.diff(timestamps))) if len(timestamps) > 1 else float(getattr(deconv, 'rate', np.nan))
...
'time_bin_size': float(1000.0 / np.median([s['native_rate_hz'] for s in session_info]))
```

iii. No rebinning is applied, matching the reference approach.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `Deconvolved` timestamps (or generated from rate if no timestamps). These are the neural data timestamps.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:] if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate), dtype=np.float64)
...
trial_t = timestamps[idx] - timestamps[start_idx]
```

iii. The timestamps come from the neural data series rather than from the behavioral time series (which the reference uses).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The start timestamp of the trial is subtracted from each frame's timestamp.

ii.
```python
trial_t = timestamps[idx] - timestamps[start_idx]
inp = np.vstack([
    trial_t.astype(np.float32),
    ...
])
```

iii. Standard subtraction of start time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same timestamps and frame indices are used for neural and behavioral data, so alignment is inherent.

ii. Same `idx` array is used for both neural and input data extraction.

iii. Neural and behavioral streams are frame-aligned in the NWB file.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
env = np.asarray(sess['environment']).astype(int)
...
np.full(len(idx), int(np.round(np.median(trial_env[trial_env >= 0]))) if np.any(trial_env >= 0) else 0, dtype=np.float32),
```

iii. The `environment` variable stores 0 or 1 for ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of valid (>= 0) environment values within the trial is taken and rounded. This is used as a per-trial constant.

ii.
```python
np.full(len(idx), int(np.round(np.median(trial_env[trial_env >= 0]))) if np.any(trial_env >= 0) else 0, dtype=np.float32),
```

iii. The median-and-round approach handles potential noise or invalid values. The reference simply takes the raw value per timepoint. Both yield the same result since environment is constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral time series value (the trial's ID from the NWB data).

ii.
```python
np.full(len(idx), float(trial), dtype=np.float32),
```
where `trial` is from `trial_ids = np.unique(trnum[valid])`.

iii. The AI uses the NWB `trial number` value directly rather than a sequential loop index. The reference uses the loop counter (0, 1, 2...).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the raw trial number value from the NWB file is used as a constant across all timepoints in the trial.

ii.
```python
np.full(len(idx), float(trial), dtype=np.float32),
```

iii. The trial number is the NWB-stored value, which should be a sequential integer for each trial within a session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and behavioral timestamps. Reward times are matched to trial time windows.

ii.
```python
reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)
...
trial_reward = {}
for trial in trial_ids:
    ...
    t0 = timestamps[idx[0]]
    t1 = timestamps[idx[-1]]
    trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```

iii. Reward events have their own timestamps that are matched to trial boundaries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward event timestamp falls within the previous trial's time range. Uses `trial - 1` as the previous trial ID. Defaults to 0 if no previous trial exists.

ii.
```python
prev_outcome = trial_reward.get(trial - 1, 0)
...
np.full(len(idx), float(prev_outcome), dtype=np.float32),
```

iii. The code uses the trial number minus 1 to look up the previous trial's outcome. This relies on trial numbers being sequential integers, which is generally the case.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. The reward zone center is inferred per trial from the mean position where `reward_zone > 0`.

ii.
```python
def infer_zone_center(pos_trial, reward_zone_trial):
    m = reward_zone_trial > 0
    if np.any(m):
        center = float(np.nanmean(pos_trial[m]))
    else:
        center = float(CANONICAL_ZONE_CENTERS[np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center))])
    return center
...
center = infer_zone_center(trial_pos, trial_rz)
dist = trial_pos - center
```

iii. The AI infers zone center from mean position when reward_zone is active, then computes distance as `position - center`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is computed as `position - zone_center`, giving a signed distance from the zone center. This is a different formulation than the reference, which computes distance to the nearest edge of the reward zone (0 when inside).

ii.
```python
dist = trial_pos - center
```

iii. The AI computes distance to center rather than distance to zone edges. The reference computes signed distance to the nearest zone boundary (negative before, zero inside, positive after).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic: `< -50`, `-50 to -10`, `-10 to 0`, `== 0`, `0 to 10`, `10 to 50`, `> 50`.

ii.
```python
def discretize_distance(dist):
    out = np.full(dist.shape, -1, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist <= -10)] = 1
    out[(dist > -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. The bin boundaries match the instructions. However, since distance is computed to center rather than zone edges, the meaning of "0 cm" differs -- it represents being at the zone center rather than at the zone edge.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices are used for both position and neural data, so alignment is inherent.

ii. The `idx` array is shared between neural and behavioral data extraction.

iii. Frame-aligned in the NWB file.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = np.asarray(sess['position'], dtype=np.float32)
...
trial_pos = pos[idx]
```

iii. Direct use of the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before discretization.

ii.
```python
discretize_position(np.clip(trial_pos, 0, 450))
```

iii. Clipping ensures positions outside the track boundaries are assigned to the edge bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: `< 90`, `90-180`, `180-270`, `270-360`, `>= 360`.

ii.
```python
def discretize_position(pos):
    out = np.full(pos.shape, -1, dtype=np.int64)
    out[pos < 90] = 0
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
    return out
```

iii. The 5 equal-sized bins of 90 cm each span the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices used for both. No additional alignment needed.

ii. Shared `idx` array.

iii. Frame-aligned in NWB.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
...
trial_lick = lick[idx]
```

iii. Direct binarization of the lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
...
trial_lick.astype(np.int64),
```

iii. Matches the instructions for binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices. No additional alignment needed.

ii. Shared `idx` array.

iii. Frame-aligned in NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. The zone center is inferred per trial from mean position where `reward_zone > 0`, then mapped to A/B/C by nearest canonical center (85, 205, 325 cm).

ii.
```python
CANONICAL_ZONE_CENTERS = np.array([85.0, 205.0, 325.0], dtype=np.float32)

def infer_zone_center(pos_trial, reward_zone_trial):
    m = reward_zone_trial > 0
    if np.any(m):
        center = float(np.nanmean(pos_trial[m]))
    ...

def zone_label_from_center(center):
    return int(np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center)))
```

iii. The canonical centers (85, 205, 325) are chosen as the midpoints of the reference's reward zone ranges ([80,130], [200,250], [320,370]).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial: (1) find frames where `reward_zone > 0`, (2) compute mean position at those frames, (3) map to nearest canonical center -> label 0 (A), 1 (B), or 2 (C). For trials without reward zone activity, falls back to nearest canonical center to median valid position.

ii.
```python
center = infer_zone_center(trial_pos, trial_rz)
zlabel = zone_label_from_center(center)
...
np.full(len(idx), zlabel, dtype=np.int64),
```

iii. The per-trial inference approach is simpler than the reference's Viterbi algorithm but should produce similar results for most trials where reward zone is clearly active.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps matched to trial time windows.

ii.
```python
reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)
...
trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```

iii. The `Reward` time series has separate timestamps from behavioral sampling.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window [t0, t1]. Output is binary (0/1), constant across all timepoints in the trial.

ii.
```python
t0 = timestamps[idx[0]]
t1 = timestamps[idx[-1]]
trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
...
np.full(len(idx), this_outcome, dtype=np.int64),
```

iii. Similar approach to the reference, using timestamp matching rather than index-based lookup.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Invalid frames**: Frames with `trial number < 0`, `scanning == False`, or non-finite position/speed are excluded via the `valid` mask.
- **Missing trial_start**: If no `trial_start` pulse is found within a trial's frames, the first valid frame is used as the start.
- **Short trials**: Trials with < 2 valid frames are dropped.
- **Missing reward zone**: If `reward_zone` is never active in a trial, falls back to nearest canonical center based on median position.
- **Sessions with < 2 trials**: Entire session is dropped.

ii.
```python
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
...
if len(idx) < 2:
    continue
...
start_candidates = idx[tstart[idx]]
start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
...
if len(neural_trials) < 2:
    continue
```

iii. These are defensive checks for data quality. The `valid` mask is more comprehensive than the reference's approach.

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files is the dominant cost, as each file contains full neural recordings. The conversion itself is relatively fast with vectorized operations.

ii. N/A

iii. NWB files are large and I/O bound.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trials()` iterates over each trial sequentially. Operations like `discretize_distance`, `discretize_position`, and `discretize_speed` could be applied to full-session arrays before splitting. The `trial_reward` computation also loops over trials.

ii. N/A

iii. Variable trial lengths make full vectorization awkward but possible with masking.

## 13-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each NWB file is loaded once. The code is relatively efficient in this regard.

ii. N/A

iii. The single-pass design avoids the survey/convert double-loading pattern seen in some approaches.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `zone_centers` and `zone_labels` lists are computed and returned but only partially used (for session info metadata). The `plane_idx` is returned but not used beyond `brain_region_idx` (which is always 0 for CA1). Position clipping to [0, 450] may unnecessarily alter edge values.

ii. N/A

iii. Minor inefficiencies with no significant impact.
