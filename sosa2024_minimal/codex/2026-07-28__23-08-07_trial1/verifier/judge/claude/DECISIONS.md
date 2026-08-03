# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `sub-*/sub-*_behavior+ophys.nwb` under the data directory, sorted by subject and session number. Each NWB file is opened using `h5py` (not `pynwb`). Every NWB file found is processed; no sessions are excluded a priori.

ii.
```python
def build_full_dataset(data_dir: Path):
    session_records = []
    for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
        print(f"Converting {path.relative_to(data_dir.parent)}")
        record = build_session(path)
        ...
        session_records.append(record)
```
```python
def build_session(path: Path):
    with h5py.File(path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"][()])
        ...
```

iii. The agent confirmed through a full inventory scan that there are 11 mice and 152 sessions matching the manuscript. It chose `h5py` over `pynwb` for direct control of data loading. From trajectory step 59: "152 sessions across 11 switch-task mice, 12,216 trials total, mean 80.37 trials/session."

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading the `general/subject/subject_id` field from each NWB file. Unique subjects are accumulated in order of first appearance during the sorted glob traversal.

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

iii. The agent verified the total number of subjects (11) matches the paper. Subject IDs are read from the NWB metadata rather than parsed from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted by (subject, session_number) using `subject_session_key`. The session ID / experiment day is parsed from the filename.

ii.
```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session
```

iii. The agent noted that sorting ensures deterministic ordering. Session IDs are also extracted from the NWB `general/session_id` field for metadata.

## 1-d. How are the data split into trials?

i. Trial starts are identified by nonzero values of the `trial_start` behavior time series. Trial ends are determined by finding the last frame with a valid corridor position (0 to 450 cm) before the next trial start. This excludes teleport-zone frames from the trial window.

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

iii. The agent noted: "Corridor frames are defined by position in [0, 450] cm; teleport-zone frames are excluded. This matches the reference focus on the 450 cm track rather than the teleport period." (CONVERSION_NOTES.md)

## 1-e. How are trials filtered based on quality controls?

i. Trials where no valid corridor frames exist (position not in [0, 450.5]) are excluded during trial segmentation. There is no explicit minimum trial length filter. Sessions with fewer than 2 trials raise an error.

ii.
```python
valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
if len(valid) == 0:
    continue
end = start + valid[-1] + 1
if end <= start:
    continue
...
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. The agent relied on the corridor position bounds as the primary quality filter. From CONVERSION_NOTES.md: "frames are kept from that start until the last frame still on the corridor."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `processing/ophys/Deconvolved` traces in the NWB file, which contain OASIS deconvolved calcium events.

ii.
```python
events = f["processing/ophys/Deconvolved"]
plane_names = get_plane_names(events)
...
plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
```

iii. From CONVERSION_NOTES.md: "Used the NWB processing/ophys/Deconvolved traces as the decoder neural input." The agent confirmed this matches the paper's decoder which uses deconvolved activity.

## 2-b. How is the `neural` data processed?

i. The neural data from multiple imaging planes is combined by concatenating curated cells across planes. The data is clipped to a common length shared by all behavior and ophys streams.

ii.
```python
arrays = []
for plane_name, plane_keep in zip(plane_names, keep_splits):
    if not np.any(plane_keep):
        continue
    curated_mask = plane_masks[plane_name]
    plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
    arrays.append(plane_data[:, plane_keep])
if not arrays:
    return np.zeros((common_length, 0), dtype=np.float32)
return np.concatenate(arrays, axis=1)
```

iii. The agent combined planes in sorted order and applied both the iscell mask and interneuron exclusion mask before concatenation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Only ROIs with `iscell[:, 0] == 1` are kept. (2) Putative interneurons are excluded by computing the correlation between each cell's dF/F and running speed; cells with `corr(dff, speed) > 0.5` are removed.

ii.
```python
def get_curated_plane_masks(f: h5py.File):
    seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
    ...
```
```python
def compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length):
    ...
    corr = np.divide(numerator, denom, ...)
    all_is_int.append(corr > 0.5)
    ...
```
```python
keep_mask = ~is_int
events = load_curated_events(f, plane_masks, keep_mask, common_length)
```

iii. From CONVERSION_NOTES.md: "Kept only manually curated ROIs with iscell[:, 0] == 1. Excluded putative interneurons with corr(dff, speed) > 0.5, matching the manuscript criterion and dayData.py."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The neural data for each trial is simply sliced from the continuous recording using the trial start and end indices. Since neural and behavior streams share the same frame rate, no interpolation is needed.

ii.
```python
neural = events[start:end].T.astype(np.float32)
```

iii. The alignment event is trial start, which is the natural boundary. No additional processing is needed since the data streams are already temporally co-registered in the NWB file.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is kept at the native imaging frame rate. The time bin size is computed as the median inter-frame interval from behavior timestamps, yielding approximately 64.5 ms. No rebinning is applied.

ii.
```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. From CONVERSION_NOTES.md: "Median frame interval from the aligned behavior timestamps: 0.0644836 s / 64.4836 ms. This matches the approximately 15.5 Hz per-plane sampling described in the paper."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavior time series.

ii.
```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The agent used position timestamps as the canonical time reference since all behavior streams share the same timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start frame is subtracted from all timestamps within the trial, yielding time relative to trial start in seconds.

ii.
```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. Straightforward subtraction of the first timestamp in the trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The neural and behavior data share the same frame indices, so indexing by the same `start:end` range ensures alignment. No interpolation or resampling is needed.

ii.
```python
neural = events[start:end].T.astype(np.float32)
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. Both are indexed by the same frame range within each trial.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily derived from the `scene` string parsed from the NWB file identifier, using `scene_schedule()`. The behavior `environment` stream is used as a cross-validation check.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
```

iii. From CONVERSION_NOTES.md: "environment_type uses the manuscript/code mapping Env1 -> 0, Env2 -> 1." The agent cross-validated against the behavior stream and found 0 mismatches.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string is parsed using regex to identify the environment(s). For switch sessions (e.g., `Env1_C_to_Env2_B`), the first 30 trials use the first environment and the rest use the second. For fixed sessions, all trials use the same environment.

ii.
```python
def scene_schedule(scene: str, ntrials: int, change_trial: int = CHANGE_TRIAL):
    env_by_trial = np.zeros(ntrials, dtype=np.int64)
    ...
    fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    ...
```

iii. The switch trial (30) comes from the reference code constant `CHANGE_TRIAL = 30`. The agent verified 0 environment-mismatched trials across all sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-indexed loop counter over trials within a session, derived from the enumeration of trial segments.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. From CONVERSION_NOTES.md: "trial_number is 0-indexed within session, matching the aligned behavior stream."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. Sequential 0-indexed trial counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` in the behavior time series. For each trial, reward outcome is determined by checking whether any reward timestamp falls within the trial's time window.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The reward timestamps are sparse (not frame-aligned), so a time-window check is used rather than index matching.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is looked up. For the first trial, previous outcome is set to 0. The value is constant across all timepoints in the trial.

ii.
```python
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
...
np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
```

iii. From CONVERSION_NOTES.md: "previous_trial_outcome uses the previous imaged trial in the same session; the first trial is set to 0."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone bounds for the current trial. The reward zone assignment comes from parsing the `scene` string from the NWB identifier using `scene_schedule()`, not from the `reward_zone` behavior stream directly.

ii.
```python
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
zone_start, zone_end = zone_bounds[trial_idx]
```
The zone bounds are defined as:
```python
ZONE_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
```

iii. The agent parsed the scene name to determine which reward zone is active on each trial, cross-validated against observed reward-zone occupancy with mean agreement of 1.0.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The signed distance from the animal's position to the nearest edge of the reward zone is computed. Distance is negative before the zone, 0 inside the zone, and positive after the zone.

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

iii. This follows the same signed-distance logic as the reference code's `reward_relative` behavior module.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit conditional comparisons matching the instruction bin edges.

ii.
```python
bins = np.empty(position_cm.shape[0], dtype=np.int64)
bins[dist < -50.0] = 0
bins[(dist >= -50.0) & (dist < -10.0)] = 1
bins[(dist >= -10.0) & (dist < 0.0)] = 2
bins[dist == 0.0] = 3
bins[(dist > 0.0) & (dist <= 10.0)] = 4
bins[(dist > 10.0) & (dist <= 50.0)] = 5
bins[dist > 50.0] = 6
```

iii. The bin edges match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices are used for neural and position data within each trial, so no additional alignment is needed.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
neural = events[start:end].T.astype(np.float32)
```

iii. Both indexed by `start:end`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The position is clipped to [0, 450] cm to stay within the corridor bounds.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm, then discretized into 5 equal-sized 90 cm bins using `floor(position / 90)`.

ii.
```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. The 450 cm corridor is divided into 5 bins of 90 cm each: [0-90), [90-180), [180-270), [270-360), [360-450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `floor(clipped_position / 90)`, capped at bin 4.

ii.
```python
bins = np.floor(clipped / 90.0).astype(np.int64)
bins[bins > 4] = 4
```

iii. This produces 5 equal-sized 90 cm bins covering the 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial. No additional alignment needed.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
neural = events[start:end].T.astype(np.float32)
```

iii. Both indexed by `start:end`.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The lick variable records frame-level lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial. No additional alignment needed.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
neural = events[start:end].T.astype(np.float32)
```

iii. Both indexed by `start:end`.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `scene` string parsed from the NWB file identifier using `scene_schedule()`. The `reward_zone` behavior stream is used only for cross-validation, not as the primary source.

ii.
```python
scene = identifier.split("/")[-1]
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. From CONVERSION_NOTES.md: "Parsed the session scene from the NWB identifier and reproduced the same schedule logic as reward_relative.behavior.get_reward_zones." Cross-validated with observed reward-zone occupancy achieving 100% agreement.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regex to determine the reward zone schedule. For switch sessions, the first 30 trials use the first zone and the rest use the second. The zone label (A/B/C) is mapped to an index (0/1/2).

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
...
zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
```

iii. This mirrors the reference code's `get_reward_zones` function.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` in the behavior time series. A reward is detected if any reward timestamp falls within the trial's time window.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The sparse reward timestamps are compared against the trial's temporal bounds.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's start and end timestamps. The value is binary (0 = no, 1 = yes) and constant across all timepoints in the trial.

ii.
```python
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```

iii. The omission fraction (0.1534) closely matches the manuscript target of approximately 15%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Stream length mismatch**: All behavior and ophys streams are clipped to a common minimum length before processing.
- **Missing corridor frames**: Trials with no valid corridor position frames (outside [0, 450.5]) are skipped.
- **Minimum session size**: Sessions with fewer than 2 trials raise an error.
- **Zero-neuron sessions**: Sessions where all neurons are excluded after interneuron filtering raise an error.

ii.
```python
common_length = min(dense_lengths)
clipped_samples = max(dense_lengths) - common_length
position = position[:common_length]
...
valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
if len(valid) == 0:
    continue
...
if n_neurons == 0:
    raise ValueError(f"{path.name}: no neurons remain after interneuron exclusion")
```

iii. From trajectory step 93: "a few session arrays are off by one sample between behavior and ophys. I'm patching the loader to trim every stream in a session to a common length before trial extraction."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Interneuron screening**: Computing per-trial dF/F from fluorescence and neuropil for every cell, then correlating with speed. This involves Gaussian filtering, min/max filtering, and per-cell per-trial baseline estimation.
2. **Loading NWB files**: Reading large arrays from HDF5.
3. **Saving the pickle file**.

ii. The `compute_interneuron_mask` function (lines 126-180) performs extensive per-trial processing:
```python
trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
trial_dff = (trial_roi - baseline) / np.abs(baseline)
trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
```

iii. From trajectory step 85: "The converter is CPU-bound rather than stuck, which is what I expected from the per-session dF/F reconstruction."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop within `build_session` iterates over each trial to compute environment, reward outcome, and trial lengths. Some of these per-trial computations (e.g., environment median, reward timestamp checks) could be vectorized. The interneuron mask computation loops over trials for dF/F baseline estimation, which could potentially use batch operations.

ii. N/A

iii. Variable-length trials make full vectorization awkward. The per-trial dF/F loop is the main bottleneck.

## 13-c. What processing does the code repeat multiple times?

i. The code reads each NWB file only once per session. However, within `build_session`, the position array is used multiple times: once for trial segmentation, once for corridor clipping, once for reward-zone cross-validation, and once for output computation. The zone bounds are looked up repeatedly per trial.

ii. N/A

iii. The single-pass design is efficient. Unlike the reference solution which has separate survey and conversion passes, this code processes each file in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron screening is computationally expensive and reconstructs per-trial dF/F from fluorescence and neuropil traces. This dF/F is used only to compute speed correlations and is then discarded; the actual neural data used for the decoder is the pre-computed deconvolved events. Additionally, the environment cross-validation against the behavior stream and the reward-zone agreement computation produce diagnostic statistics but are not used in the final dataset.

ii.
```python
def compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length):
    ...
    # Computes full dF/F only to get speed correlation
    del f_roi, f_neu, dff, dff_valid, dff_centered  # explicitly freed
```

iii. The dF/F reconstruction is necessary to match the paper's interneuron exclusion criterion but is expensive relative to the small number of neurons it removes (mean 0.31% of curated cells).
