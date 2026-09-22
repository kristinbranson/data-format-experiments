# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `pathlib.Path.glob` to find all NWB files matching `sub-*/sub-*_behavior+ophys.nwb` under the data root directory. Each NWB file is opened with `pynwb.NWBHDF5IO` and all behavioral and neural data arrays are read from it. All subjects and sessions found in the directory are processed.

ii.
```python
DATA_ROOT = Path("/app/data")
...
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
with NWBHDF5IO(str(session_path), "r", load_namespaces=True) as io:
    nwb = io.read()
    beh = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
```

iii. The agent explored the data directory structure in early steps (steps 10-19) and confirmed 11 switch-task mice are present. The glob pattern finds all NWB files across all subject directories.

## 1-b. How are the data split into subjects?

i. Subjects are identified by parsing the parent directory name of each NWB file, stripping the `sub-` prefix. A mapping from subject name to index is built incrementally.

ii.
```python
subject = session_path.parent.name.replace("sub-", "")
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

iii. The agent confirmed 11 mice are present in the data directory (step 19).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The glob returns sorted paths, so sessions are processed in filesystem order.

ii.
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for session_idx, session_path in enumerate(session_paths, start=1):
```

iii. The agent noted "trials are sliced from trial_start inclusive to teleport exclusive" per NWB file (step 150).

## 1-d. How are the data split into trials?

i. Trials are defined by pairing `trial_start` events (positive values) with `teleport` events (positive values). Each trial spans from the `trial_start` index (inclusive) to the `teleport` index (exclusive). Edge cases are handled: a leading teleport before the first start is dropped; a trailing start after the last teleport is dropped; pairs where stop <= start are skipped.

ii.
```python
def _prepare_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    if teleports.size == starts.size + 1 and teleports[0] < starts[0]:
        teleports = teleports[1:]
    if starts.size == teleports.size + 1 and starts[-1] > teleports[-1]:
        starts = starts[:-1]
    n_pairs = min(starts.size, teleports.size)
    starts = starts[:n_pairs]
    teleports = teleports[:n_pairs]
    bounds = []
    for start, stop in zip(starts, teleports):
        if stop > start:
            bounds.append((int(start), int(stop)))
    return bounds
```

iii. The agent investigated trial boundaries in steps 39-43, confirming `trial_start` and `teleport` are the correct signals, matching the repository's `keep_teleports=False` preprocessing path (step 39).

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter trials based on minimum trial length or any other quality control. All valid trial bounds (where stop > start) are kept.

ii.
```python
# No trial filtering code exists beyond the stop > start check in _prepare_trial_bounds
bounds = []
for start, stop in zip(starts, teleports):
    if stop > start:
        bounds.append((int(start), int(stop)))
```

iii. No explicit justification was given for omitting trial filtering. The agent's trajectory does not discuss minimum trial length thresholds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the NWB's `Deconvolved` data interface, which contains suite2p's own deconvolution. This is NOT the paper's custom dF/F + OASIS deconvolution pipeline that operates on raw Fluorescence and Neuropil traces.

ii.
```python
deconv_iface = nwb.processing["ophys"].data_interfaces["Deconvolved"]
...
for series_name, rr in sorted(deconv_iface.roi_response_series.items(), ...):
    deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. The agent benchmarked two possible approaches (step 32): "directly on the provided deconvolved traces versus recomputing dF/F with the repo's maximin baseline." The agent chose to use the pre-computed Deconvolved traces from the NWB file, noting it as "Suite2p deconvolved activity from NWB, filtered to iscell ROIs" in the metadata.

## 2-b. How is the `neural` data processed?

i. No additional processing is done. The AI reads the pre-existing `Deconvolved` data from the NWB file and uses it directly. The data is cast to float16 for storage. No custom dF/F computation, no maximin baseline, no neuropil correction, no OASIS deconvolution.

ii.
```python
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
...
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The agent considered recomputing dF/F (step 32) but decided the pre-computed deconvolved traces were sufficient, calling it "a practical approximation" in the metadata.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` filtering from the plane segmentation table, keeping only manually curated ROIs; (2) speed-correlation filtering to remove putative interneurons (correlation threshold > 0.5). The interneuron screen is done on the deconvolved traces rather than on dF/F.

ii.
```python
iscell_all = np.asarray(seg["iscell"].data[:])[:, 0].astype(bool)
...
keep_plane = iscell_all[plane_mask]
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
...
non_interneuron = _speed_corr_filter(deconv, speed, bounds)
deconv = deconv[:, non_interneuron]
```

iii. The agent investigated `iscell` encoding (step 24) and implemented the paper's interneuron exclusion criterion (step 32). The metadata notes it as "a practical approximation to the paper's speed-correlated interneuron screen."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by simply slicing data from the `trial_start` index to the `teleport` index. No additional alignment or shifting is needed since neural and behavioral data share the same timebase.

ii.
```python
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The agent confirmed neural and behavioral data are synchronized in the NWB file (steps 14, 39).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is set to `1000/15.5078125 = ~64.48 ms`, a hardcoded frame rate constant. This is the scanner rate, not the per-plane rate for multi-plane sessions.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
```

iii. No explicit discussion about multi-plane rate adjustments is present in the trajectory. The agent hardcoded the frame rate rather than reading it from the NWB file per session.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series timestamps.

ii.
```python
timestamps = np.asarray(beh["position"].timestamps[:], dtype=np.float64)
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent used position timestamps as the canonical timebase for the session.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the trial is subtracted from all timestamps in the trial.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Standard approach to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same frame indices, so the same `start:stop` slice ensures alignment. No interpolation or resampling is needed.

ii.
```python
# Same start:stop indices used for both
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The NWB file stores synchronized neural and behavioral data at the same sampling rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = np.asarray(beh["environment"].data[:], dtype=np.float32)
...
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]
env = float(_safe_mode(env_vals.astype(int), fallback=0))
```

iii. The agent investigated environment encoding (step 53) and confirmed it can change within a session on day 8.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The mode (most common value) of the `environment` values within the trial is used, after filtering out negative values. This makes the environment a per-trial constant.

ii.
```python
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]
env = float(_safe_mode(env_vals.astype(int), fallback=0))
...
np.full(T, env, dtype=np.float32),
```

iii. The agent checked whether environment changes within sessions (step 53) and used mode to handle the per-trial assignment.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series stored in the NWB file, with a fallback to the trial loop index if the value is negative.

ii.
```python
trial_number = np.asarray(beh["trial number"].data[:], dtype=np.float32)
...
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
```

iii. No explicit justification in trajectory for using the stored `trial number` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The `trial number` value at the first frame of the trial is used directly. If negative, falls back to the sequential trial index. The value is constant for all timepoints in the trial.

ii.
```python
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
np.full(T, tnum, dtype=np.float32),
```

iii. Simple extraction from the NWB stored trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps and the behavioral timestamps.

ii.
```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
...
reward_outcome_by_trial = np.zeros(len(bounds), dtype=np.uint8)
for trial_idx, (start, stop) in enumerate(bounds):
    trial_start_t = timestamps[start]
    trial_end_t = timestamps[stop]
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any(
        (reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
```

iii. The agent derived reward outcomes by checking whether any reward timestamp falls within each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, reward outcomes are computed for all trials by checking if any reward timestamp falls within [trial_start_time, trial_end_time). Then for each trial, the previous trial's outcome is used. For the first trial, 0 is used.

ii.
```python
prev_reward = float(reward_outcome_by_trial[trial_idx - 1]) if trial_idx > 0 else 0.0
...
np.full(T, prev_reward, dtype=np.float32),
```

iii. Standard lag-1 approach. Uses timestamp-based matching rather than index-based matching.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the inferred reward zone label for each trial. Reward zone inference uses `reward_zone` behavior time series and `position` to determine which zone (A, B, C) is active per trial.

ii.
```python
position = np.asarray(beh["position"].data[:], dtype=np.float32)
reward_zone = np.asarray(beh["reward_zone"].data[:], dtype=np.float32)
...
zone_by_trial = _infer_reward_zone_by_trial(position, reward_zone, bounds)
```

iii. The agent investigated how reward zones are encoded in the data and built an inference method based on median position during reward zone activity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to [0, 450] first. Signed distance is computed from position to the nearest edge of the assigned reward zone. Distance is 0 when inside the zone, negative when before, positive when past.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
...
def _bin_distance_to_reward_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    start = ZONE_STARTS_CM[zone_idx]
    end = ZONE_ENDS_CM[zone_idx]
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start
    after = position_cm > end
    dist[before] = position_cm[before] - start
    dist[after] = position_cm[after] - end
```

iii. Matches the concept from the paper of distance relative to the reward zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins with explicit conditionals: < -50 (0), -50 to -10 (1), -10 to 0 (2), exactly 0 (3), 0 to 10 (4), 10 to 50 (5), > 50 (6).

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

iii. Matches the 7-bin specification from the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (`start:stop`) used for both neural and position data, so alignment is automatic.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM)
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. Neural and behavioral data share the same timebase in the NWB file.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(beh["position"].data[:], dtype=np.float32)
...
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
```

iii. Direct use of the stored position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before binning.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
```

iii. Clipping ensures all positions fall within the track bounds.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: < 90 (0), 90-180 (1), 180-270 (2), 270-360 (3), >= 360 (4).

ii.
```python
def _bin_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    bins = np.empty(position_cm.shape[0], dtype=np.uint8)
    bins[position_cm < 90.0] = 0
    bins[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    bins[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    bins[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    bins[position_cm >= 360.0] = 4
    return bins
```

iii. Matches the 5 equal-sized bin specification (90 cm each over 450 cm track).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices used for both, ensuring automatic alignment.

ii. Same `start:stop` slicing as neural data.

iii. Synchronized via NWB frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(beh["lick"].data[:], dtype=np.float32)
...
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
```

iii. Direct use of stored lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive value becomes 1, otherwise 0.

ii.
```python
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
```

iii. Instructions specify binary lick output (0 = no, 1 = yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices, automatic alignment.

ii. Same `start:stop` slicing.

iii. Synchronized via NWB frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavior time series. The `_infer_reward_zone_by_trial` function determines the zone (A=0, B=1, C=2) for each trial based on the median position when the reward zone indicator is active.

ii.
```python
def _infer_reward_zone_by_trial(position, reward_zone, bounds):
    raw_zone = np.full(len(bounds), -1, dtype=np.int64)
    for trial_idx, (start, stop) in enumerate(bounds):
        rz_pos = position[start:stop][reward_zone[start:stop] > 0]
        if rz_pos.size:
            raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
```

iii. The agent investigated the reward zone variable encoding and built a zone inference algorithm.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the median position when `reward_zone > 0` is computed, and the closest zone center is assigned. For trials with no reward zone activity, a two-phase fill is used: mode of pre-switch (first 30 trials) and mode of post-switch trials. A nearest-neighbor fallback handles remaining gaps.

ii.
```python
pre_valid = raw_zone[:SWITCH_SPLIT_TRIAL][raw_zone[:SWITCH_SPLIT_TRIAL] >= 0]
post_valid = raw_zone[SWITCH_SPLIT_TRIAL:][raw_zone[SWITCH_SPLIT_TRIAL:] >= 0]
if pre_valid.size and post_valid.size:
    inferred = np.empty(len(bounds), dtype=np.int64)
    inferred[:SWITCH_SPLIT_TRIAL] = _safe_mode(pre_valid)
    inferred[SWITCH_SPLIT_TRIAL:] = _safe_mode(post_valid, fallback=int(inferred[SWITCH_SPLIT_TRIAL - 1]))
    return inferred
```

iii. This approach uses the paper's 30-trial switch structure to infer zones for trials where the reward zone event was missed.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
...
reward_outcome_by_trial[trial_idx] = np.uint8(np.any(
    (reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
```

iii. Reward event presence within trial time window determines outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within [trial_start_time, trial_end_time). Binary 0/1 per trial, constant across all timepoints.

ii.
```python
for trial_idx, (start, stop) in enumerate(bounds):
    trial_start_t = timestamps[start]
    trial_end_t = timestamps[stop]
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any(
        (reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
```

iii. Standard approach using timestamp comparison.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: All arrays are cropped to `n_frames = min(timestamps, position, speed, lick, environment, reward_zone, trial_number, trial_start, teleport, deconv)`.
- **Trial bound edge cases**: Leading teleports and trailing starts are trimmed; bounds where stop <= start are skipped.
- **Missing reward zone data**: Trials with no reward zone activity get -1 and are filled via the mode/nearest-neighbor algorithm.
- **No neurons after filtering**: A RuntimeError is raised.
- **Fewer than 2 trials**: A RuntimeError is raised.

ii.
```python
n_frames = min(timestamps.size, position.size, speed.size, lick.size,
               environment.size, reward_zone.size, trial_number.size,
               trial_start.size, teleport.size, deconv.shape[0])
...
if teleports.size == starts.size + 1 and teleports[0] < starts[0]:
    teleports = teleports[1:]
```

iii. The agent encountered and handled edge cases during development (steps 75, 86).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** - I/O bound, reading large neural data matrices
2. **Speed correlation filtering** - computing correlation for each neuron against speed
3. **Saving the pickle file** - writing the full dataset

ii. N/A

iii. The agent noted the conversion was "mostly I/O-bound" (step 73).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `_speed_corr_filter` function is already vectorized. The `_infer_reward_zone_by_trial` per-trial loop could potentially be vectorized. The `_prepare_trial_bounds` uses numpy operations efficiently.

ii. N/A

iii. The code is generally well-vectorized compared to the reference.

## 13-c. What processing does the code repeat multiple times?

i. The code does not repeat processing. Each NWB file is loaded only once. All data extraction happens in a single pass per session.

ii. N/A

iii. Unlike the reference which has a survey + conversion two-pass approach, this code processes everything in one pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code clips position to [0, 450] which alters the raw data. It also clips speed to >= 0 with `np.maximum(speed, 0.0)`. These modifications are not necessary as the raw values are already within expected ranges for on-track data. The `session_info` metadata dictionary is quite detailed but not used downstream.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
spd = np.maximum(speed[start:stop], 0.0).astype(np.float32, copy=False)
```

iii. No explicit justification given for clipping.
