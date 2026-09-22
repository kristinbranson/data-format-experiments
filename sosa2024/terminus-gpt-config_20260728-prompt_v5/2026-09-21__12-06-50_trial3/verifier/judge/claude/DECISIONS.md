# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered via `Path('/app/data').rglob('*.nwb')`, sorted alphabetically. Each NWB file is opened with `h5py.File` and data arrays are read directly from their HDF5 paths. Subject IDs are read from `general/subject/subject_id` within each file. All 152 NWB files are processed (or first 2 with `--sample`).

ii.
```python
def find_nwb_files():
    return sorted(Path('/app/data').rglob('*.nwb'))
...
with h5py.File(fpath, 'r') as f:
    subj = read_scalar(f, 'general/subject/subject_id')
    neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
    ...
```

iii. The AI found all NWB files using recursive glob and processed them sequentially. Subject IDs are extracted from metadata embedded in each file.

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `general/subject/subject_id` from each NWB file. Unique subject IDs are accumulated as files are processed, and a `subject_to_idx` mapping tracks which sessions belong to which subject.

ii.
```python
subj = read_scalar(f, 'general/subject/subject_id')
...
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. Subject IDs are read directly from NWB metadata rather than parsed from directory structure or filenames.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session ID is read from `general/session_id` in the NWB file. All data from a single NWB file becomes one session.

ii.
```python
sess_id = read_scalar(f, 'general/session_id', fpath.stem)
```

iii. The one-file-per-session structure was identified during data exploration.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using `trial_start` and `teleport` behavior time series. Starts are detected at rising edges of `trial_start` (via `np.diff > 0`). Ends are detected at rising edges of `teleport`. Each start is paired with the next end after it. Trials are further required to have at least some timepoints where `trial number >= 0`.

ii.
```python
def get_trial_bounds(trial_start, teleport, trial_number):
    starts = np.where(np.diff(trial_start.astype(int), prepend=0) > 0)[0]
    ends = np.where(np.diff(teleport.astype(int), prepend=0) > 0)[0]
    bounds = []
    for s in starts:
        e_candidates = ends[ends > s]
        if e_candidates.size == 0:
            continue
        e = int(e_candidates[0])
        if np.any(trial_number[s:e] >= 0):
            bounds.append((int(s), int(e)))
    return bounds
```

iii. Trial boundaries from `trial_start` and `teleport` are consistent with the reference code's approach of using these variables to define trial windows.

## 1-e. How are trials filtered based on quality controls?

i. Within each trial, only timepoints where `trial number >= 0` AND `scanning > 0` are included. Trials with fewer than 2 valid timepoints after this filtering are skipped. Sessions with fewer than 2 valid trials are also skipped.

ii.
```python
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
if valid.sum() < 2:
    continue
idx = np.where(valid)[0] + s
```

iii. The `trial number >= 0` check excludes off-trial timepoints. The `scanning > 0` check ensures only periods with active scanning are included. The minimum of 2 valid timepoints is very lenient.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB's `processing/ophys/Deconvolved/plane0/data` array, which is suite2p's built-in deconvolution of fluorescence. Only `plane0` is read; plane1 data in multi-plane sessions is ignored.

ii.
```python
neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
```

iii. The AI reasoned that since the paper uses "deconvolved activity," the NWB `Deconvolved` field should be used directly. However, the paper actually computes its own deconvolved signal through a custom pipeline (neuropil subtraction, dF/F with maximin baseline, OASIS deconvolution), which produces a different result than suite2p's built-in deconvolution stored in the NWB.

## 2-b. How is the `neural` data processed?

i. No processing is applied. The raw `Deconvolved/plane0/data` array is used directly, cast to float32. There is no neuropil subtraction, no baseline correction, no dF/F computation, and no custom deconvolution.

ii.
```python
neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The AI treated the NWB Deconvolved data as ready-to-use. The CONVERSION_NOTES Step 5 states: "Use deconvolved activity as neural input: Methods explicitly state deconvolved activity was used as the response matrix; NWB provides aligned Deconvolved data matching fluorescence shapes."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural quality filtering is applied. All neurons (ROIs) from `plane0` are included regardless of whether they are classified as cells (`iscell`) or whether they are putative interneurons. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
# No iscell filtering, no interneuron filtering
...
neu = neural[idx].T.astype(np.float32)
```

iii. The AI did not implement the paper's cell curation steps (iscell manual curation and putative interneuron removal based on speed correlation > 0.5), despite the reference code's `is_putative_interneuron` function being identified during code exploration (Step 1 notes mention it).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by extracting only the valid timepoints within each trial's boundaries. Since both neural and behavioral data share the same time base, alignment is implicit.

ii.
```python
idx = np.where(valid)[0] + s  # valid timepoints within trial
neu = neural[idx].T.astype(np.float32)
```

iii. The instructions specify alignment to trial start. Extracting data starting from the trial start index achieves this.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at its native sampling rate. However, the `time_bin_size` in metadata is set to `None` rather than being computed from the sampling rate.

ii.
```python
'time_bin_size': None,
```

iii. The AI did not compute the time bin size from the neural data sampling rate. The verification output shows the time bin size is not recorded in metadata.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array of the `position` behavior time series.

ii.
```python
ts = np.asarray(f['processing/behavior/BehavioralTimeSeries/position/timestamps'], dtype=np.float64)
...
t_rel = (ts[idx] - t0).astype(np.float32)
```

iii. Position timestamps are used as the time base. All behavior time series share the same timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first valid timepoint in the trial is subtracted from all timestamps in the trial.

ii.
```python
t0 = ts[idx[0]]
t_rel = (ts[idx] - t0).astype(np.float32)
```

iii. Standard relative time computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same `idx` array of valid timepoints within each trial. No separate alignment is needed because both share the same time base in the NWB file.

ii.
```python
idx = np.where(valid)[0] + s
t_rel = (ts[idx] - t0).astype(np.float32)
neu = neural[idx].T.astype(np.float32)
```

iii. Both use the same index array, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series. The first valid (finite, >= 0) value within each trial is taken.

ii.
```python
env_vals = beh['environment'][s:e][valid]
env_per_trial.append(int(first_valid(env_vals)) if env_vals.size else -1)
```

iii. The environment variable is constant within trials and represents ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. A session-wide `compute_env_mapping` creates a mapping from raw environment codes to 0/1. In practice, the raw values are already 0 and 1, so this is an identity mapping.

ii.
```python
def compute_env_mapping(env_trials):
    vals = sorted({int(v) for v in env_trials if v >= 0})
    return {v: i for i, v in enumerate(vals[:2])}
...
np.full_like(t_rel, float(env_per_trial[i] if env_per_trial[i] >= 0 else 0), dtype=np.float32)
```

Note: The `env_map` is computed but never actually applied to the input values. The raw `env_per_trial[i]` values are used directly.

iii. The raw environment values happen to be 0 and 1 already, so the mapping is not needed in practice.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` behavior time series. The first valid (finite) value within each trial is taken.

ii.
```python
tn_vals = beh['trial number'][s:e][valid]
trial_nums.append(float(first_valid(tn_vals)) if tn_vals.size else len(trial_nums))
...
np.full_like(t_rel, trial_nums[i], dtype=np.float32)
```

iii. The AI uses the stored trial number from the NWB file rather than a sequential counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The first valid trial number value within each trial is used as a constant across all timepoints. If no valid trial number is found, the count of trials so far is used as a fallback.

ii.
```python
trial_nums.append(float(first_valid(tn_vals)) if tn_vals.size else len(trial_nums))
```

iii. The NWB trial number is used directly with minimal processing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. When the Reward array has different length from the behavior timestamps (indicating it has its own timestamp base), reward timestamps are used to match rewards to trials by time.

ii.
```python
reward = np.asarray(f[f'{base}/Reward/data']) if f'{base}/Reward/data' in f else None
reward_ts = np.asarray(f[f'{base}/Reward/timestamps']) if f'{base}/Reward/timestamps' in f else None
...
if reward is not None and reward_ts is not None and reward.shape[0] != ts.shape[0]:
    t_start = ts[s]
    t_end = ts[e-1] if e-1 < len(ts) else ts[-1]
    m_rew = (reward_ts >= t_start) & (reward_ts <= t_end)
    reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
```

iii. The AI correctly identifies that the Reward time series may have a different timestamp base and handles both cases.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is used. For the first trial (i=0), 0 is used as default.

ii.
```python
np.full_like(t_rel, reward_outcomes[i-1] if i > 0 else 0, dtype=np.float32)
```

iii. Note: When `i=0`, `reward_outcomes[i-1]` would be `reward_outcomes[-1]` (Python negative indexing), accessing the last element. But the `if i > 0 else 0` guard prevents this. The logic is correct.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and the per-trial reward zone center. The reward zone center for each trial is computed as the median position of the mouse when `reward_zone > 0`. Centers are then clustered across the session to map to A/B/C categories.

ii.
```python
rz_vals = beh['reward_zone'][s:e][valid]
m = rz_vals > 0
rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
rz_map = compute_rz_mapping_from_centers(rz_centers)
```

iii. The AI uses the median position when reward_zone is active as a proxy for the reward zone center, then clusters these centers to assign A/B/C labels.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is computed as `position - reward_zone_center`, a simple subtraction from the center point. This means distance is 0 only at the exact center of the reward zone, not across the full zone extent. The distance is then discretized into 7 bins.

ii.
```python
dist = pos - float(rz_center)
...
out = np.vstack([
    discretize_distance(dist),
    ...
])
```

iii. This is conceptually different from computing distance to the nearest edge of the reward zone (which would be 0 anywhere inside the zone). The AI's approach computes distance from a single point (the median position).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The `discretize_distance` function uses conditional masks to assign 7 bins. It uses `np.isclose(d, 0)` for the "0 cm" category (bin 3).

ii.
```python
def discretize_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d <= -10)] = 1
    out[(d > -10) & (d < 0)] = 2
    out[np.isclose(d, 0)] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The bin boundaries match the instructions. Using `np.isclose(d, 0)` for the "0 cm" bin means only values very close to 0 are placed in bin 3. With the center-point distance computation, this bin captures very few timepoints.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `idx` array is used for both neural and behavioral data within each trial.

ii.
```python
idx = np.where(valid)[0] + s
pos = beh['position'][idx].astype(np.float32)
neu = neural[idx].T.astype(np.float32)
```

iii. Alignment is implicit through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['position'][idx].astype(np.float32)
```

iii. The position variable records the animal's position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, 450]` before discretization.

ii.
```python
discretize_position(np.clip(pos, 0, TRACK_LEN))
```

iii. Clipping constrains values to the track range. A few samples may fall slightly outside [0, 450] and get clipped to the boundary.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using conditional masks with edges at 90, 180, 270, 360 cm.

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

iii. Five equal 90 cm bins spanning the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `idx` array as neural data.

ii. `pos = beh['position'][idx]`

iii. Shared indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
```

iii. Matches the binary specification in the instructions.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `idx` array as neural data.

ii. `lick = (beh['lick'][idx] > 0)`

iii. Shared indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series. For each trial, the median position when `reward_zone > 0` gives a center estimate. These centers are clustered across the session using `compute_rz_mapping_from_centers` to assign A/B/C labels.

ii.
```python
rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
rz_map = compute_rz_mapping_from_centers(rz_centers)
...
nearest = min(rz_map.keys(), key=lambda x: abs(x - rz_center))
rz_cat = rz_map[nearest]
```

iii. The AI uses a data-driven approach to discover reward zone locations rather than using known zone boundaries from the paper.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. `compute_rz_mapping_from_centers` collects all unique center positions, merges centers within 30 cm of each other, and assigns A=0, B=1, C=2 by sorted position. Each trial's center is mapped to the nearest cluster.

ii.
```python
def compute_rz_mapping_from_centers(centers):
    vals = sorted({round(float(v), 1) for v in centers if v is not None and np.isfinite(v)})
    merged = []
    for v in vals:
        if not merged or abs(v - merged[-1]) > 30:
            merged.append(v)
    return {v: min(i, 2) for i, v in enumerate(merged[:3])}
```

iii. This data-driven clustering approach can fail if the reward zone positions are noisy or overlap.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series, handling both cases where Reward has its own timestamp base or shares the behavior timestamp base.

ii.
```python
if reward is not None and reward_ts is not None and reward.shape[0] != ts.shape[0]:
    t_start = ts[s]
    t_end = ts[e-1] if e-1 < len(ts) else ts[-1]
    m_rew = (reward_ts >= t_start) & (reward_ts <= t_end)
    reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
elif reward is not None:
    reward_outcomes.append(int(np.any(reward[s:e][valid] > 0)))
```

iii. The dual handling was needed because the Reward time series has its own timestamp base.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, binary: 1 if any reward > 0 occurred within the trial, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
np.full(t_rel.shape, reward_outcomes[i], dtype=np.int64)
```

iii. Per-trial binary output matching the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Length mismatches**: If neural and behavior arrays differ in length, all are trimmed to the common minimum.
- **Invalid timepoints**: Timepoints where `trial number < 0` or `scanning <= 0` are excluded within trials.
- **Missing reward zone**: If no valid reward zone center is found for a trial, the median position of all valid samples is used as fallback.
- **Missing reward data**: If `Reward/data` is not found in the NWB, reward outcome defaults to 0.
- **Short trials**: Trials with < 2 valid timepoints are skipped.
- **Sessions with few trials**: Sessions with < 2 valid trials are skipped.

ii.
```python
if len(set(lengths)) != 1:
    print(f'length mismatch...')
common_len = min(lengths)
...
if rz_center is None or not np.isfinite(rz_center):
    rz_center = float(np.nanmedian(pos))
```

iii. These defensive checks handle data inconsistencies found across sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files with h5py and reading the large neural arrays. Each session takes 0.14-0.85 seconds. Total conversion for 152 sessions runs in approximately 60 seconds.

ii. N/A

iii. The code is efficient since it reads pre-computed Deconvolved data directly without reprocessing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over trial bounds sequentially. Operations like discretization could potentially be applied to session-level arrays before splitting into trials, but variable trial lengths make this awkward. The `compute_rz_mapping_from_centers` has a simple loop over centers.

ii. N/A

iii. The per-trial loop is a natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code makes a single pass over all NWB files. Within each session, reward zone centers are computed in a first pass over trials, then used in a second pass for output construction. This is a minor duplication but necessary for the data-driven reward zone mapping.

ii. N/A

iii. The two-pass per-session design (first compute rz_centers, then convert trials) adds minimal overhead.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `fluor` (Fluorescence) array is read from NWB but only used for a shape assertion (`assert neural.shape == fluor.shape`), then discarded. This wastes I/O and memory reading a large array unnecessarily. The `rz_series` (reward zone values per timepoint) is also computed but never used. The `env_map` is computed but never applied.

ii.
```python
fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
assert neural.shape == fluor.shape  # only use of fluor
...
rz_series = beh['reward_zone'][idx].astype(np.float32)  # computed but unused
...
env_map = compute_env_mapping(env_per_trial)  # computed but unused
```

iii. The fluorescence data read is wasteful I/O. The unused variables are remnants of development.
