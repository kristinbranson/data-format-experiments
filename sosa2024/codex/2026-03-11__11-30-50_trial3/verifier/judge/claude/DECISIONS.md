# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files directly using `h5py` (not `pynwb`). It finds all NWB files by globbing `data/sub-*/sub-*_behavior+ophys.nwb`. Each file corresponds to one session. Data is loaded session-by-session, reading behavior time series from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/Deconvolved`. Files are sorted by path for deterministic ordering.

ii.
```python
def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```
```python
with h5py.File(path, "r") as f:
    behavior_group = f["processing/behavior/BehavioralTimeSeries"]
    ophys_group = f["processing/ophys"]
    position = behavior_group["position/data"][()].astype(np.float32)
    ...
```

iii. The AI chose `h5py` over `pynwb` for speed. All NWB files in `data/` subdirectories are discovered via glob pattern. The 152 files across 11 subjects match the expected dataset.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the parent directory name of each NWB file (e.g., `sub-m11` -> `m11`). A sorted unique set of subjects is built from all processed sessions.

ii.
```python
subject = path.parent.name.replace("sub-", "")
...
subjects = sorted({sess["subject"] for sess in processed_sessions})
```

iii. The directory structure encodes subject identity. The AI extracts this from the path rather than parsing filenames.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed one file at a time.

ii.
```python
for idx, path in enumerate(nwb_files, start=1):
    session_start = time.time()
    session, examples = process_session(path)
```

iii. The one-file-per-session structure matches the data organization.

## 1-d. How are the data split into trials?

i. Trials are identified by finding frames where `trial_start > 0` (start markers) and `teleport > 0` (end markers). For each start, the next teleport after it defines the trial boundary. The trial runs from the start frame up to (but not including) the teleport frame.

ii.
```python
def find_complete_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    teleport_idx = 0
    for start in starts:
        while teleport_idx < len(teleports) and teleports[teleport_idx] <= start:
            teleport_idx += 1
        if teleport_idx >= len(teleports):
            break
        stop = teleports[teleport_idx]
        if stop > start:
            bounds.append((int(start), int(stop)))
        teleport_idx += 1
    return bounds
```

iii. The AI uses a forward-scanning approach pairing each start with the next subsequent teleport. The reference uses `nonzero(trial_start)` for starts and detects teleport onset transitions for ends.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered on three criteria: (1) trials with fewer than `MIN_TRIAL_FRAMES=5` frames are dropped, (2) trials with lick sensor errors (>35% of frames having cumulative lick count >2) are dropped, (3) trials with insufficient valid (finite, non-NaN) behavior/neural frames are dropped.

ii.
```python
MIN_TRIAL_FRAMES = 5
LICK_ERROR_FRACTION = 0.35
...
def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
...
if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
...
if meta["lick_error"]:
    dropped_lick += 1
    continue
```

iii. The lick error filtering follows the reference code's `glmUtils.get_timeseries_data` threshold of 35%. The reference human solution uses a minimum of 50 timepoints; the AI uses 5.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `Deconvolved` traces in `processing/ophys/Deconvolved/plane{N}/data`, filtered by the `iscell` mask.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
...
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. The paper's decoder uses deconvolved calcium events, matching this choice.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are filtered by `iscell[:,0] == 1`, concatenated across planes for multi-plane sessions, and cast to float32. No further processing (e.g., dF/F recomputation, normalization) is applied. Additionally, frames with non-finite neural values are masked out.

ii.
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
...
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    plane_data = plane_data.astype(np.float32, copy=False)
    ...
    deconv[:, dest_cols] = plane_data
...
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```

iii. The NWB deconvolved data is already processed. Multi-plane sessions are pooled across planes consistent with the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered using the `iscell[:,0] == 1` mask from suite2p curation. No additional filtering (e.g., speed-correlated interneuron exclusion) is applied.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
```

iii. The AI documented that `iscell` filtering is the primary shared curation available in NWB. Additional interneuron exclusion (speed correlation > 0.5) was not implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The trial boundaries define the slice of neural data for each trial, starting from the trial_start frame. No temporal offset is applied.

ii.
```python
trial_slice = slice(start, stop)
...
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. Since the data is already frame-aligned and trials start at the trial_start marker, alignment to trial start is inherent in the slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native sampling rate (~15.5 Hz, ~64.5 ms per frame). No temporal rebinning is applied. The time bin size is computed as the median of timestamp differences across sessions.

ii.
```python
"time_bin_size": median_bin_ms,
...
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
```

iii. The reference paper and code use the same frame-aligned ~15.5 Hz sampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavior time series.

ii.
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. The AI uses position timestamps as the canonical time base. The reference uses `trial number` timestamps, but they are equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first frame in the trial is subtracted from all timestamps within the trial.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. Standard approach to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame-aligned time base. The same frame mask is applied to both, ensuring alignment.

ii.
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The common frame mask ensures all data streams are temporally aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment/data` behavior time series.

ii.
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
```

iii. The environment variable encodes ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial environment is computed as the modal (most frequent) valid value within the trial, rounded and thresholded at 0.5, then broadcast as a constant across all timepoints.

ii.
```python
def env_to_binary(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid environment values in trial.")
    return int(np.round(np.median(values)) > 0.5)
...
env_bin = env_to_binary(environment[start:stop])
...
np.full(time_trial.shape, meta["environment"], dtype=np.float32),
```

iii. The reference code uses the raw environment value directly (cast to int). The AI takes a median-based approach with filtering of invalid values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number/data` behavior time series via the `modal_trial_number` function, which finds the most frequent valid trial number value within each trial's time window.

ii.
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
...
def modal_trial_number(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid trial numbers in trial.")
    counts = np.bincount(values)
    return int(np.argmax(counts))
...
trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The reference human solution uses the loop counter (0, 1, 2, ...) rather than the stored trial number variable. The AI uses the stored `trial number` from the NWB file.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The modal (most frequent) valid, non-negative trial number from the raw data is computed using `np.bincount` and `np.argmax`. This value is then broadcast as a constant across all timepoints in the trial.

ii.
```python
trial_num = modal_trial_number(trial_number[start:stop])
...
np.full(time_trial.shape, trial_num, dtype=np.float32),
```

iii. The AI chose to use the stored trial number for consistency with the NWB data rather than a sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` (reward delivery timestamps), `position/timestamps` (behavior timestamps), and `reward_zone/data` (reward zone entry events).

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
...
def reward_outcome_for_trial(
    reward_timestamps: np.ndarray,
    t_start: float,
    t_stop: float,
    reward_zone_segment: np.ndarray,
) -> int:
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)
```

iii. The AI determines reward outcome by checking both reward delivery events AND reward zone entry within the trial. The previous trial's outcome is then used for the current trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the previous trial (by trial number) is looked up from a dictionary. If the previous trial number doesn't exist, defaults to 0. The value is broadcast as a constant across all timepoints.

ii.
```python
reward_by_trial_number: dict[int, int] = {}
...
reward_outcome = reward_outcome_for_trial(...)
reward_by_trial_number[trial_num] = reward_outcome
...
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
...
np.full(time_trial.shape, prev_outcome, dtype=np.float32),
```

iii. The AI uses stored trial numbers to look up previous outcomes, rather than sequential trial indices. This means it looks up `trial_num - 1` rather than the previous trial in the loop.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position/data` (animal position) and the reward zone coordinates determined from the session's scene metadata parsed from the NWB `identifier` string.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. The AI parses the scene name from the NWB identifier to determine reward zone (A/B/C), using regex patterns for single, switch, and cross-environment scenes. This is similar to the reference code's `behavior.get_reward_zones`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from animal position to the nearest edge of the reward zone. Distance is 0 when inside the zone, negative when before, positive when after.

ii.
```python
def signed_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. This matches the reference's `compute_distance_to_reward_zone` function logic.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 categories using explicit conditional logic rather than `np.digitize`.

ii.
```python
def discretize_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, -1, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    out[distance_cm > 50] = 6
    return out
```

iii. The bin boundaries match the instructions: `< -50`, `-50 to -10`, `-10 to < 0`, `0`, `>0 to +10`, `+10 to +50`, `> +50`. The reference uses `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame-aligned indices and frame mask as the neural data.

ii.
```python
trial_slice = slice(start, stop)
position_trial = position[trial_slice][frame_mask]
...
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. Common indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position/data` behavior time series.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
position_trial = position[trial_slice][frame_mask]
```

iii. The position variable directly records the animal's VR corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, 450)` and divided into 5 bins of 90 cm each using `np.floor(clipped / 90.0)`.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. Clips to [0, 450) then divides by 90 cm for 5 equal bins: [0-90), [90-180), [180-270), [270-360), [360-450).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each over the range [0, 450). Clipped to the range before binning.

ii.
```python
clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. The reference uses bin edges `[-inf, 50, 150, 250, 350, inf]` giving 100 cm bins centered differently. The AI uses 90 cm bins starting from 0.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-aligned indices and frame mask as neural data.

ii. Same indexing: `position_trial = position[trial_slice][frame_mask]`

iii. Common frame mask ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick/data` behavior time series.

ii.
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
...
lick_trial = lick[trial_slice][frame_mask]
```

iii. The lick variable records lick events per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any value > 0 is mapped to 1, otherwise 0.

ii.
```python
(lick_trial > 0).astype(np.int16),
```

iii. Matches the binary lick output specification.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-aligned indices and frame mask as neural data.

ii. Same indexing: `lick_trial = lick[trial_slice][frame_mask]`

iii. Common frame mask ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` string, which encodes the session's scene name (e.g., `Env1_LocationA`, `Env1_LocationA_to_C`). The scene name is parsed using regex to determine the reward zone (A, B, or C) and whether a switch occurs.

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")
```

iii. The AI matches the reference code's `behavior.get_reward_zones` approach of parsing scene metadata.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For switch sessions, trials before trial 30 use the "before" zone and trials >= 30 use the "after" zone. The zone label (A/B/C) is mapped to integer (0/1/2) and broadcast as a constant across all timepoints.

ii.
```python
SWITCH_TRIAL = 30
...
def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        assert scene_info.after_zone is not None
        return scene_info.after_zone
    return scene_info.before_zone
...
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
```

iii. The switch trial threshold of 30 matches the paper ("Each switch occurred after 30 trials").

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` and `reward_zone/data`.

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```

iii. Uses both reward delivery timestamps and reward zone entry events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded (1) if both: (a) at least one reward delivery timestamp falls within the trial's time window, AND (b) the reward_zone signal is positive at some point during the trial. The value is broadcast as a constant across all timepoints.

ii.
```python
def reward_outcome_for_trial(
    reward_timestamps: np.ndarray,
    t_start: float,
    t_stop: float,
    reward_zone_segment: np.ndarray,
) -> int:
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)
```

iii. The reference simply checks for any reward event within the trial. The AI adds an additional check that the reward zone was entered. This is more conservative but could differ in edge cases.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Non-finite values**: A per-frame validity mask checks that position, speed, lick, environment, and timestamps are all finite, position > -100, and all neural values are finite. Frames failing this mask are excluded.
- **Short trials**: Trials with fewer than 5 valid frames are dropped.
- **Lick sensor errors**: Trials where >35% of frames have cumulative lick count >2 are dropped.
- **Sessions with <2 valid trials**: Skipped entirely.

ii.
```python
valid_behavior_frames = (
    np.isfinite(position)
    & np.isfinite(speed)
    & np.isfinite(lick)
    & np.isfinite(environment)
    & np.isfinite(timestamps)
    & (position > -100.0)
)
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
...
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. The AI applies more aggressive frame-level filtering than the reference, which only crops to the minimum length if neural and behavior lengths differ.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files and reading the deconvolved neural data arrays. The full conversion takes about 3.36 minutes for 152 sessions. Using `h5py` directly instead of `pynwb` was chosen for speed.

ii. N/A

iii. The AI documented conversion took ~3.36 min for all sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that constructs input/output arrays processes each trial sequentially. The discretization functions (distance, position, speed) are already vectorized within each trial. The trial-level metadata computation (modal trial number, environment) uses vectorized numpy operations. There is no Viterbi loop as the AI uses scene parsing instead.

ii. N/A

iii. The main loop structure is natural given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session only once (no separate survey step). All data loading and processing happens in a single pass through `process_session()`.

ii. N/A

iii. This is more efficient than the reference which has a separate survey step that loads all NWB files before conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The frame-level validity mask (`valid_behavior_frames`, `valid_neural_frames`) checks for non-finite values and position > -100 at every frame, even though these conditions are likely rare within valid trial boundaries. The `modal_trial_number` computation uses `np.bincount` which is more complex than simply using a loop counter.

ii. N/A

iii. The overhead is minimal.
