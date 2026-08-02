# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read NWB files (rather than `pynwb`). It finds all NWB files via `data_dir.glob("sub-*/sub-*_behavior+ophys.nwb")`, sorted by a `subject_session_key` that parses subject and session numbers from the file path. Each NWB file is opened with `h5py.File(path, "r")` and all relevant data streams (behavior, neural, fluorescence, neuropil) are read.

ii.
```python
def build_full_dataset(data_dir: Path):
    session_records = []
    for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
        record = build_session(path)
        session_records.append(record)

def build_session(path: Path):
    with h5py.File(path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"][()])
        ...
        beh = f["processing/behavior/BehavioralTimeSeries"]
        position = np.asarray(beh["position/data"], dtype=np.float32)
        ...
```

iii. The AI chose h5py for direct access to HDF5 data within NWB files. This avoids the pynwb dependency while still reading the same underlying data. All subjects and sessions are found via globbing the data directory.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject_id` field inside each NWB file. Unique subjects are collected in order of first appearance as files are processed.

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"][()])
...
subjects = []
subject_to_idx = {}
for record in session_records:
    subject = record["subject"]
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subjects)
        subjects.append(subject)
```

iii. The subject ID is read directly from the NWB metadata rather than parsing directory names. Both approaches yield the same subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session identifier (experiment day) is read from `general/session_id` in the NWB file.

ii.
```python
session_id = decode_scalar(f["general/session_id"][()])
...
"exp_day": int(session_id),
```

iii. The one-file-per-session structure is standard for this NWB dataset.

## 1-d. How are the data split into trials?

i. Trial starts are identified from the `trial_start` behavior variable (nonzero entries). Trial ends are defined as the last frame where position is within the corridor bounds [0, 450.5] cm, before the next trial start. This excludes teleport-zone frames.

ii.
```python
def find_trial_segments(position: np.ndarray, trial_start: np.ndarray):
    starts = np.flatnonzero(trial_start > 0)
    next_starts = np.concatenate([starts[1:], [len(position)]])
    ends = []
    for start, next_start in zip(starts, next_starts):
        trial_pos = position[start:next_start]
        valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
        if len(valid) == 0:
            continue
        end = start + valid[-1] + 1
        if end <= start:
            continue
        ends.append(end)
    starts = starts[:len(ends)]
    ends = np.array(ends, dtype=np.int64)
    return starts.astype(np.int64), ends
```

iii. The AI's approach clips trial ends to the last valid corridor frame, excluding the teleport period. The CONVERSION_NOTES state: "Corridor frames are defined by position in [0, 450] cm; teleport-zone frames are excluded."

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped only if no valid corridor frames exist (empty trial) or if `end <= start`. There is no explicit minimum timepoints filter. Sessions with fewer than 2 trials are rejected entirely.

ii.
```python
if len(valid) == 0:
    continue
end = start + valid[-1] + 1
if end <= start:
    continue
...
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. The AI relies on the position-based trial segmentation to naturally exclude degenerate trials rather than applying a minimum timepoints threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` traces in `processing/ophys/Deconvolved`.

ii.
```python
events = f["processing/ophys/Deconvolved"]
...
plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
```

iii. The paper describes using deconvolved calcium events for neural decoding. The CONVERSION_NOTES confirm: "Used the NWB processing/ophys/Deconvolved traces as the decoder neural input."

## 2-b. How is the `neural` data processed?

i. Neural data is processed in multiple steps: (1) combine cells across planes, (2) filter by `iscell` (manual curation), (3) exclude putative interneurons with `corr(dff, speed) > 0.5`. The interneuron identification involves computing dF/F from Fluorescence and Neuropil traces with neuropil coefficient 0.7, applying Gaussian smoothing and min/max baseline estimation, then computing correlation with speed.

ii.
```python
def compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length):
    ...
    for plane_name in get_plane_names(fluorescence):
        f_roi = np.asarray(fluorescence[plane_name]["data"][:common_length, curated_mask], dtype=np.float32).T
        f_neu = np.asarray(neuropil[plane_name]["data"][:common_length, curated_mask], dtype=np.float32).T
        ...
        for start, end in zip(starts, ends):
            trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
            trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
            baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
            baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
            baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
            trial_dff = (trial_roi - baseline) / np.abs(baseline)
            trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
            ...
        corr = ...
        all_is_int.append(corr > 0.5)
    ...
    keep_mask = ~is_int
    events = load_curated_events(f, plane_masks, keep_mask, common_length)
```

iii. The CONVERSION_NOTES state this matches the reference logic in `dayData.py` and the manuscript criterion for interneuron exclusion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) `iscell[:, 0] == 1` from `ImageSegmentation/PlaneSegmentation`, and (2) exclusion of putative interneurons with `corr(dff, speed) > 0.5`.

ii.
```python
def get_curated_plane_masks(f):
    seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
    ...
# Then interneuron exclusion:
is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
keep_mask = ~is_int
```

iii. The CONVERSION_NOTES report that interneuron removal fraction is very small (~0.3% of cells per session), consistent with the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Each trial's neural data is extracted by indexing the full session array with `events[start:end]`, where `start` corresponds to the `trial_start` frame.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    neural = events[start:end].T.astype(np.float32)
```

iii. The instructions specify "Temporally align based on start of the trial," which is directly satisfied by the indexing scheme.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at the stored sampling rate. The time bin size is computed from the median inter-sample interval of the behavior timestamps.

ii.
```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. The CONVERSION_NOTES report the median frame interval as 64.48 ms, matching the ~15.5 Hz per-plane sampling described in the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavior time series.

ii.
```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The behavior timestamps are consistent across all variables, so any could have been used.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from all timestamps within the trial.

ii.
```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. Standard approach for computing time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data share the same frame indices within each trial (same `start:end` range), so alignment is inherent.

ii.
```python
neural = events[start:end].T.astype(np.float32)
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. All data streams are clipped to a common length before trial segmentation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from parsing the session's `scene` string from the NWB file's `identifier` field, using `scene_schedule()`. The behavior `environment` data is used for validation but not as the primary source.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
inputs = np.vstack([
    ...
    np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
    ...
])
```

iii. The CONVERSION_NOTES say: "Parsed the session scene from the NWB identifier and reproduced the same schedule logic as `reward_relative.behavior.get_reward_zones`." The AI validated with 0 environment mismatches against the behavior data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string is parsed via regex to determine environment assignment per trial. Fixed sessions use one environment throughout. Switch sessions split at trial 30 (`CHANGE_TRIAL`).

ii.
```python
def scene_schedule(scene: str, ntrials: int, change_trial: int = CHANGE_TRIAL):
    fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    if fixed_match:
        env = ENV_TO_IDX[fixed_match.group(1)]
        env_by_trial[:] = env
    elif env_switch_match:
        ...
        split = min(change_trial, ntrials)
        env_by_trial[:split] = env0
        env_by_trial[split:] = env1
```

iii. This replicates the schedule logic from the reference code's `get_reward_zones`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-indexed loop counter (`trial_idx`) over trials within a session.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. The CONVERSION_NOTES state: "trial_number is 0-indexed within session, matching the aligned behavior stream."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop index. The value is constant across all timepoints within a trial.

ii.
```python
np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. Straightforward sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` behavior time series, compared against `position/timestamps` to determine whether a reward was delivered within each trial's time window.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. Reward events have their own timestamps separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward timestamp falls within that trial's time window. The previous trial's reward outcome is then used for the current trial's input. The first trial defaults to 0.

ii.
```python
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
...
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
```

iii. The instructions specify binary (omitted=0, rewarded=1). The first trial uses 0 since there is no previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior data and the reward zone boundaries determined by `scene_schedule()` (parsed from the NWB identifier's scene string).

ii.
```python
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
zone_start, zone_end = zone_bounds[trial_idx]
...
discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The reward zone boundaries are from ZONE_BOUNDS_CM: A=[80,130], B=[200,250], C=[320,370], matching the reference paper.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the reward zone is computed: negative when before the zone, 0 when inside, positive when past the zone.

ii.
```python
def discretize_distance_to_zone(position_cm, zone_start, zone_end):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    ...
```

iii. Matches the paper's concept of signed distance relative to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit comparisons matching the instruction bin edges.

ii.
```python
bins[dist < -50.0] = 0
bins[(dist >= -50.0) & (dist < -10.0)] = 1
bins[(dist >= -10.0) & (dist < 0.0)] = 2
bins[dist == 0.0] = 3
bins[(dist > 0.0) & (dist <= 10.0)] = 4
bins[(dist > 10.0) & (dist <= 50.0)] = 5
bins[dist > 50.0] = 6
```

iii. The bin edges match the instructions: <-50, -50 to -10, -10 to <0, 0, >0 to +10, +10 to +50, >+50.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (`start:end`) are used for both neural and position data, so alignment is inherent.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
neural = events[start:end].T.astype(np.float32)
```

iii. All data streams are aligned by sharing the same time indices within each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The position records the animal's location in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm before discretization.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
...
discretize_position(pos_trial)
```

iii. Clipping ensures all positions fall within the corridor range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each by `floor(clip(pos, 0, 450) / 90)`, capped at 4.

ii.
```python
def discretize_position(position_cm):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. The corridor is 450 cm, so 5 bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360-450).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
neural = events[start:end].T.astype(np.float32)
```

iii. Inherent alignment through shared time indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
neural = events[start:end].T.astype(np.float32)
```

iii. Inherent alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from parsing the session's `scene` string from the NWB `identifier` field via `scene_schedule()`. The `reward_zone` behavior data is used only for validation.

ii.
```python
scene = identifier.split("/")[-1]
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. The CONVERSION_NOTES report perfect reward-zone agreement between the scene-parsed schedule and the observed behavior data.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed via regex to determine the reward zone schedule (which zone is active on each trial). Fixed sessions use one zone; switch sessions change at trial 30.

ii.
```python
def scene_schedule(scene, ntrials, change_trial=CHANGE_TRIAL):
    ...
    zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
    ...
```

iii. Zone mapping: A=0, B=1, C=2, matching the instructions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` behavior time series.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. Reward events are timestamped separately from the behavior sampling rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window. Binary output: 0=no, 1=yes. The value is constant across all timepoints in the trial.

ii.
```python
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```

iii. The 1e-9 tolerance avoids floating-point edge cases.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Stream length mismatch**: All data arrays (behavior, neural, fluorescence, neuropil) are clipped to the minimum common length.
- **Empty trials**: Trials with no valid corridor frames are skipped.
- **Sessions with too few trials**: Sessions with <2 trials raise an error.
- **No neurons after filtering**: Sessions where all neurons are excluded raise an error.

ii.
```python
common_length = min(dense_lengths)
...
position = position[:common_length]
...
if len(valid) == 0:
    continue
...
if ntrials < 2:
    raise ValueError(...)
if n_neurons == 0:
    raise ValueError(...)
```

iii. The CONVERSION_NOTES report the number of clipped samples per session in the session summaries.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Interneuron mask computation**: computing dF/F from Fluorescence/Neuropil for all curated cells, applying Gaussian smoothing, min/max filtering, and correlation with speed.
2. **Loading NWB files**: reading large arrays from HDF5 files (I/O bound).
3. **Full dataset assembly**: iterating over all sessions and extracting trial data.

ii. N/A

iii. The interneuron computation is particularly expensive as it processes raw fluorescence for every curated cell across all trial frames.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_session` (lines 338-367) iterates over each trial sequentially. The interneuron dF/F computation has a per-trial loop (lines 154-162) that processes each trial segment. These could potentially be vectorized with padded arrays or concatenated operations.

ii. N/A

iii. Variable trial lengths make vectorization awkward but not impossible.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session in a single pass (no separate survey step), so there is minimal repeated processing. However, the environment validation loop (lines 300-319) and the main trial extraction loop (lines 338-367) both iterate over trials, and some data (like position) is accessed in both.

ii. N/A

iii. Unlike the reference solution which has a separate survey step that re-loads every NWB file, the AI's code avoids this by doing everything in one pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron detection computes full dF/F traces from Fluorescence and Neuropil data solely to identify putative interneurons. The dF/F itself is discarded -- only the boolean mask is kept. The environment validation (comparing scene-derived vs behavior-derived env values) produces diagnostic info but is not used for the final output. The zone validation (comparing parsed zone to observed reward_zone positions) is similarly diagnostic only.

ii.
```python
# dF/F computed only for interneuron detection, then discarded:
del f_roi, f_neu, dff, dff_valid, dff_centered
```

iii. The dF/F computation is the most expensive "unnecessary" processing, though it serves an important quality control purpose.
