# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing `sub-*/sub-*_behavior+ophys.nwb` under the data directory, sorted by subject and session number. Files are opened with `h5py` (not `pynwb`). Each NWB file corresponds to one session. All behavior and neural data arrays are read from the HDF5 groups.

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
with h5py.File(path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"][()])
    ...
    beh = f["processing/behavior/BehavioralTimeSeries"]
    position = np.asarray(beh["position/data"], dtype=np.float32)
    ...
```

iii. The agent examined the directory structure and confirmed 11 subject directories matching the paper. It chose `h5py` over `pynwb` for direct HDF5 access. The glob pattern captures all NWB files across all subjects.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject_id` field stored inside each NWB file (`f["general/subject/subject_id"]`). A unique subject list is built as sessions are processed.

ii.
```python
def build_full_dataset(data_dir: Path):
    ...
    subjects = []
    subject_to_idx = {}
    for record in session_records:
        subject = record["subject"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)
```

iii. The agent confirmed 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID (experiment day) is parsed from the file's `session_id` field and from the filename pattern `ses-XX`.

ii.
```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session
```

```python
session_id = decode_scalar(f["general/session_id"][()])
```

iii. The agent verified that sessions correspond to experiment days as described in the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined using the `trial_start` behavior variable to find starts, then for each trial, the end is found by scanning the position array for the last valid corridor sample (position between 0 and 450.5 cm) before the next trial start.

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

iii. The agent investigated the trial structure and chose to use position validity (0-450.5 cm) to identify the end of each trial rather than using the `teleport` signal.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial length filtering is applied. Trials are only excluded if the position-based end detection yields zero valid frames or an empty segment (end <= start). The code requires at least 2 trials per session.

ii.
```python
if len(valid) == 0:
    continue
end = start + valid[-1] + 1
if end <= start:
    continue
ends.append(end)
...
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. The agent did not implement a minimum trial length filter. The position-based segmentation naturally excludes degenerate trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's pre-stored `Deconvolved` events (suite2p's OASIS deconvolution), NOT from recomputing dF/F from raw Fluorescence and Neuropil traces. However, the Fluorescence and Neuropil traces ARE loaded separately for the interneuron screen (computing dF/F correlation with speed).

ii.
```python
def load_curated_events(f, plane_masks, keep_mask, common_length):
    events = f["processing/ophys/Deconvolved"]
    plane_names = get_plane_names(events)
    ...
    for plane_name, plane_keep in zip(plane_names, keep_splits):
        ...
        plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
        arrays.append(plane_data[:, plane_keep])
    ...
    return np.concatenate(arrays, axis=1)
```

iii. The agent's metadata states: `"neural_signal": "suite2p OASIS deconvolved calcium events exported in the NWB files"`. The agent identified the Deconvolved field in the NWB and used it directly rather than recomputing deconvolution from raw traces.

## 2-b. How is the `neural` data processed?

i. No processing beyond loading and filtering. The pre-stored Deconvolved events are loaded directly and used as-is. The only processing is subsetting to curated cells (`iscell`) and removing putative interneurons.

ii.
```python
events = load_curated_events(f, plane_masks, keep_mask, common_length)
...
neural = events[start:end].T.astype(np.float32)
```

iii. The agent chose to use the stored deconvolved signal rather than recomputing the paper's processing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` curation mask from the `ImageSegmentation/PlaneSegmentation` table, and (2) putative interneuron exclusion based on dF/F-speed correlation > 0.5. For the interneuron screen, the agent recomputes a simplified dF/F from Fluorescence/Neuropil (neuropil subtraction, maximin baseline, Gaussian smoothing) and correlates it with speed.

ii.
```python
def get_curated_plane_masks(f: h5py.File):
    seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
    ...

def compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length):
    ...
    for start, end in zip(starts, ends):
        trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
        trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
        baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
        baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
        baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
        trial_dff = (trial_roi - baseline) / np.abs(baseline)
        trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
        dff[:, start:end] = trial_dff
    ...
    corr = ...
    all_is_int.append(corr > 0.5)
```

iii. The agent identified that the paper excludes putative interneurons with speed correlation > 0.5, matching the paper's Methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. No additional temporal alignment is needed since neural and behavioral data share the same frame indices.

ii.
```python
neural = events[start:end].T.astype(np.float32)
```

iii. The alignment is implicit: the trial start index is used to slice both neural and behavioral data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the original sampling rate with no rebinning. The time bin size is computed as the median of timestamp differences.

ii.
```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. The agent did not apply any temporal rebinning, using the native imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps (`beh["position/timestamps"]`).

ii.
```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The agent used the position timestamps as they are aligned with the neural data frames.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from all timestamps within the trial.

ii.
```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. Simple subtraction of the initial timestamp.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices, so no additional alignment is needed.

ii. Same `start:end` indices used for both neural and behavioral slicing.

iii. The agent verified that all dense streams share the same time axis.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The environment type is derived from the NWB `identifier` field's scene name, parsed via regex to extract `Env1`/`Env2` labels. The `scene_schedule()` function handles fixed, same-env-switch, and env-switch patterns.

ii.
```python
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)

def scene_schedule(scene, ntrials, change_trial=CHANGE_TRIAL):
    fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    ...
```

iii. The agent found that the scene name encoded in the NWB identifier reliably determines the environment schedule, including mid-session switches.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine the environment for each trial. For fixed sessions, all trials get the same environment. For switch sessions, trials before trial 30 get one environment and trials after get another. The agent also cross-checks against the behavior `environment` stream.

ii.
```python
if fixed_match:
    env = ENV_TO_IDX[fixed_match.group(1)]
    env_by_trial[:] = env
...
elif env_switch_match:
    split = min(change_trial, ntrials)
    env_by_trial[:split] = env0
    env_by_trial[split:] = env1
```

iii. The agent used scene name parsing rather than the raw `environment` behavior variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index (0-indexed), derived from the loop counter over trials.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    inputs = np.vstack([
        ...
        np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
        ...
    ])
```

iii. The trial number is simply the sequential index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop index. The value is constant across all timepoints within a trial.

ii.
```python
np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. Direct assignment of the enumeration index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `position` timestamps. Reward events are matched to trial boundaries by checking if any reward timestamp falls within the trial's time window.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The agent used reward timestamps to determine if a reward was delivered during each trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome is computed by checking if any reward timestamp falls within the trial's time window. The previous trial's outcome is then used as the input for the current trial. For the first trial, the value is 0.

ii.
```python
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
...
np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
```

iii. The agent first computes reward outcome for all trials, then shifts by one to get previous trial outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone bounds. The reward zone location for each trial is determined by parsing the scene name from the NWB `identifier` field via `scene_schedule()`.

ii.
```python
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
zone_start, zone_end = zone_bounds[trial_idx]
```

iii. The agent used the scene name to determine reward zone locations rather than the `reward_zone` behavior variable.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from the animal's position to the nearest edge of the reward zone. Distance is 0 when inside the zone, negative when before, positive when past. Position is clipped to [0, 450] cm before computation.

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

iii. Standard signed distance computation to the reward zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit conditional assignments matching the instruction bins.

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

iii. The bin edges match the instruction specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices used for both neural and behavioral slicing within each trial.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. Aligned by construction through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. Position directly records the animal's corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm, then discretized into 5 equal bins of 90 cm using `np.floor(clipped / 90.0)`, with values > 4 capped at 4.

ii.
```python
def discretize_position(position_cm):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. The 5 bins of 90 cm span the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Uses `np.floor(clipped / 90.0)` with capping at bin 4. This produces 5 bins: [0,90), [90,180), [180,270), [270,360), [360,450].

ii.
```python
bins = np.floor(clipped / 90.0).astype(np.int64)
bins[bins > 4] = 4
```

iii. Equivalent to equal-sized 90 cm bins across the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as the neural data within each trial.

ii. `pos_trial = np.clip(position[start:end], 0.0, 450.0)`

iii. Aligned by construction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. Matches the instruction specification of binary output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as the neural data within each trial.

ii. `lick[start:end]`

iii. Aligned by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field's scene name, parsed by `scene_schedule()` to determine the reward zone label (A, B, or C) for each trial. The function handles fixed, same-env-switch, and env-switch session types.

ii.
```python
scene = identifier.split("/")[-1]
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. The agent cross-validated the scene-derived labels against the `reward_zone` behavior variable's position data and reported high agreement.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed with regex to extract the zone label. For switch sessions, the first 30 trials get one zone and subsequent trials get another. Zone labels are mapped to indices (A=0, B=1, C=2).

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
...
zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
```

iii. The agent verified zone distributions matched expectations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and `position` timestamps.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The reward timestamp is compared against the trial time window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's position timestamp range (with a small epsilon). Output is binary: 1 if rewarded, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```

iii. The small epsilon (1e-9) handles floating-point boundary cases.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: All dense streams (behavior and neural) are trimmed to the minimum common length.
- **Empty trials**: Trials with no valid position frames (0-450.5 cm) are skipped.
- **Minimum session size**: Sessions with fewer than 2 trials raise an error.
- **No neurons after filtering**: Sessions with 0 neurons after interneuron exclusion raise an error.

ii.
```python
dense_lengths = [len(position), len(position_ts), len(speed), ...]
for grp_name in ("Deconvolved", "Fluorescence", "Neuropil"):
    ...
    dense_lengths.append(grp[plane_name]["data"].shape[0])
common_length = min(dense_lengths)
position = position[:common_length]
...
```

iii. The agent encountered an off-by-one mismatch between behavior and ophys frames in some sessions and implemented the common-length trimming as a fix.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Interneuron screening** (`compute_interneuron_mask`): loads raw fluorescence/neuropil, computes trial-by-trial dF/F, and correlates with speed for every curated cell.
2. **Loading NWB files**: I/O-bound reading of large HDF5 arrays.
3. **Loading deconvolved events** (`load_curated_events`): reading the full neural data arrays.

ii. N/A

iii. The agent noted the conversion was CPU-bound due to the dF/F reconstruction for the interneuron screen.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `compute_interneuron_mask` iterates over each trial for baseline computation. The per-trial loop in `build_session` for computing reward outcomes and environment assignments could be partially vectorized. The correlation computation in `compute_interneuron_mask` is already vectorized across cells.

ii. N/A

iii. Variable trial lengths make full vectorization difficult.

## 13-c. What processing does the code repeat multiple times?

i. For the interneuron screen, the code loads and processes raw Fluorescence/Neuropil data to compute dF/F, even though the final neural signal comes from the pre-stored Deconvolved events. This means fluorescence data is loaded once for interneuron screening and deconvolved data is loaded separately for the final output.

ii. N/A

iii. The two-pass approach (interneuron screen + event loading) is necessary because the interneuron mask must be computed before selecting which neurons to keep from the deconvolved events.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes environment type from scene names and cross-validates against the behavior `environment` stream, counting mismatches. This cross-validation is stored in metadata but not used for the actual output. The code also builds a `sample_data.pkl` subset that may not be needed for the full evaluation.

ii.
```python
env_mismatch = int(np.sum(env_by_trial != np.asarray(behavior_env_by_trial, dtype=np.int64)))
```

iii. The cross-validation is a sanity check rather than a processing step.
