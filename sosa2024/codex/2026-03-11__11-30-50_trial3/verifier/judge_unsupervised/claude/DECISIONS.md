# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored in the `data/` directory. It uses `h5py` to directly read HDF5/NWB files, iterating over all files matching the glob pattern `data/sub-*/sub-*_behavior+ophys.nwb`. Each NWB file contains one session's worth of behavioral and optical physiology data. The files are sorted by path, giving a deterministic subject-then-session ordering. In `--sample` mode, only the first 2 files are processed.

ii.
```python
def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```

```python
def process_session(path: Path) -> tuple[dict, list[dict]]:
    session_id = path.stem.replace("_behavior+ophys", "")
    with h5py.File(path, "r") as f:
        # ... reads behavior and ophys groups
        behavior_group = f["processing/behavior/BehavioralTimeSeries"]
        ophys_group = f["processing/ophys"]
        position = behavior_group["position/data"][()].astype(np.float32)
        speed = behavior_group["speed/data"][()].astype(np.float32)
        lick = behavior_group["lick/data"][()].astype(np.float32)
        environment = behavior_group["environment/data"][()].astype(np.float32)
        reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
        trial_number = behavior_group["trial number/data"][()].astype(np.float32)
        trial_start = behavior_group["trial_start/data"][()].astype(np.float32)
        teleport = behavior_group["teleport/data"][()].astype(np.float32)
        timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
        reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
```

iii. The AI chose `h5py` over higher-level NWB libraries for speed, noting in CONVERSION_NOTES.md Step 6 that this avoids overhead. The NWB files are the released data format equivalent to the reference code's `sess` pickles.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the parent directory name of each NWB file, stripping the `sub-` prefix. A sorted set of unique subjects becomes the `subjects` list, and each session is mapped to its subject index.

ii.
```python
subject = path.parent.name.replace("sub-", "")
```
```python
def build_dataset(processed_sessions: list[dict]) -> dict:
    subjects = sorted({sess["subject"] for sess in processed_sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    # ...
    "subject_idx": np.asarray(
        [subject_to_idx[sess["subject"]] for sess in processed_sessions],
        dtype=np.int64,
    ),
```

iii. CONVERSION_NOTES.md Step 2 documents 11 subjects (m3, m4, m7, m11-m15, m17-m19) consistent with the paper's 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed sequentially in sorted file-path order. Sessions with fewer than 2 valid trials after filtering are skipped.

ii.
```python
for idx, path in enumerate(nwb_files, start=1):
    session_start = time.time()
    session, examples = process_session(path)
    if len(session["neural"]) < 2:
        print(f"[{idx}/{len(nwb_files)}] Skipping {path.name}: ...")
        continue
    processed_sessions.append(session)
```

iii. CONVERSION_NOTES.md Step 2 confirms 152 NWB files across 11 subjects. The conversion log shows all 152 sessions were processed with none skipped.

## 1-d. How are the data split into trials?

i. Trials are defined by boundaries between `trial_start > 0` events and the next `teleport > 0` event. Only "complete" trials with both a start marker and a subsequent teleport marker are included.

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

iii. CONVERSION_NOTES.md Step 4 notes this matches the reference code's use of `trial_start_inds` and `teleport_inds` to delimit trials. The trial range is [start, stop) -- inclusive of the start frame, exclusive of the teleport frame.

## 1-e. How are trials filtered based on quality controls?

i. Two quality filters are applied: (1) Lick artifact detection: trials where >35% of frames have cumulative lick count >2 are dropped. (2) Minimum valid frames: trials with fewer than 5 valid frames (after masking NaN/invalid behavior and neural data) are dropped.

ii.
```python
LICK_ERROR_FRACTION = 0.35
MIN_TRIAL_FRAMES = 5

def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
```
```python
for meta in trial_meta:
    if meta["lick_error"]:
        dropped_lick += 1
        continue
    # ...
    frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
    if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
        dropped_missing += 1
        continue
```

iii. CONVERSION_NOTES.md Step 4 documents that the reference code uses 35% threshold (in `glmUtils.get_timeseries_data`) while the methods text says 30%. The AI chose to follow the code (35%) over the text.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the pre-computed deconvolved calcium traces stored at `processing/ophys/Deconvolved/plane{N}/data` in each NWB file, filtered by the suite2p cell curation mask `iscell[:,0] == 1`.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
# ...
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. CONVERSION_NOTES.md Step 5 states: "Use NWB deconvolved traces directly" as equivalent to the reference `sess.timeseries['events']`. The paper's decoder uses deconvolved calcium events.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are loaded per-plane for multi-plane sessions, filtered to accepted cells only, concatenated into a single (T, n_neurons) matrix, then sliced by trial boundaries and transposed to (n_neurons, T) per trial. Within each trial, invalid frames (NaN behavior or neural data, invalid positions) are masked out.

ii.
```python
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 5 decision 3: "Do not recompute dF/F from fluorescence: The release already provides frame-aligned deconvolved traces." The AI uses the pre-computed deconvolved data directly without further processing (no dF/F, no smoothing, no additional deconvolution).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the suite2p `iscell[:,0] == 1` mask only. No additional filtering for putative interneurons (speed correlation > 0.5) is applied.

ii.
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
```

iii. CONVERSION_NOTES.md Step 5 decision 2: "Filter neurons with `iscell[:,0] == 1` only: This is the shared, explicit curated-cell mask available in NWB and matches the baseline neuron curation." Step 4 acknowledges that the paper also excludes putative interneurons by speed correlation > 0.5, but notes the NWB doesn't contain a ready-made interneuron mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data begins at the `trial_start` marker frame and ends at the `teleport` marker frame (exclusive). The time input variable is computed relative to the trial start timestamp, providing explicit temporal alignment.

ii.
```python
trial_slice = slice(start, stop)
# ...
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. The instructions specify "Temporally align based on start of the trial." The AI aligns to the first frame of each trial (the `trial_start` marker), consistent with both the instructions and the reference code's trial segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5 Hz (~64.5 ms per frame). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval across all sessions.

ii.
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
```
```python
median_bin_ms = float(
    np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
)
```

iii. CONVERSION_NOTES.md Step 3 documents the frame rate as ~15.5 Hz and notes that behavioral and neural time series are on the same frame-aligned grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavioral time series group.

ii.
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
```

iii. CONVERSION_NOTES.md Step 5: "Behavior timestamps are the reliable aligned time base, especially in multi-plane sessions where the ophys `rate` attribute can be misleading."

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp of each valid frame is subtracted from the timestamp of the trial's first frame (the `trial_start` marker frame). This gives time in seconds from trial start.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```
```python
input_trial = np.vstack([
    time_trial.astype(np.float32),
    # ...
])
```

iii. This is a straightforward computation matching the instruction specification of "Time from start of trial in seconds."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same frame mask is applied to both neural data and timestamps, ensuring perfect frame-by-frame alignment. Both use the same `trial_slice` and `frame_mask`.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. Alignment is inherent because all variables share the same frame-aligned time base from the NWB file.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment/data` array in the behavioral time series group.

ii.
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 maps this to `input[1]` and references the `behavior.get_trial_types` function which derives `morph` (0=Env1, 1=Env2).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the modal value of the environment variable is computed after filtering out NaN and negative values. The median is rounded to produce a binary 0/1 value. This per-trial value is then broadcast across all timepoints.

ii.
```python
def env_to_binary(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid environment values in trial.")
    return int(np.round(np.median(values)) > 0.5)
```
```python
env_bin = env_to_binary(environment[start:stop])
# ...
np.full(time_trial.shape, meta["environment"], dtype=np.float32),
```

iii. CONVERSION_NOTES.md describes environment as binary ENV1/ENV2 per trial. Using median and rounding is a robust way to determine the per-trial environment from frame-level values that may contain sentinel values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number/data` array in the behavioral time series group.

ii.
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 maps this to `input[2]`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, the modal (most frequent) value of the raw `trial number` variable within the trial's frame range is computed. NaN and negative values are filtered out first, then `np.bincount` + `np.argmax` finds the mode. The result is broadcast across all timepoints.

ii.
```python
def modal_trial_number(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid trial numbers in trial.")
    counts = np.bincount(values)
    return int(np.argmax(counts))
```
```python
trial_num = modal_trial_number(trial_number[start:stop])
np.full(time_trial.shape, trial_num, dtype=np.float32),
```

iii. The AI uses the raw NWB trial numbers (0-indexed) rather than reindexing. This preserves the original data's trial numbering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward outcome computed for each trial (which itself comes from `Reward/timestamps` and `reward_zone/data`). The previous trial's outcome is looked up by trial number.

ii.
```python
reward_by_trial_number[trial_num] = reward_outcome
# ...
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. CONVERSION_NOTES.md Step 5 decision 11: "Set first-trial previous outcome to 0: There is no previous within-session trial, so 0 is the least assumption-laden sentinel."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A dictionary maps trial numbers to their reward outcomes. For each trial, the previous trial number's outcome is looked up. If no previous trial exists (first trial or missing), defaults to 0. The value is broadcast across all timepoints.

ii.
```python
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
np.full(time_trial.shape, prev_outcome, dtype=np.float32),
```

iii. This correctly chains the reward outcomes across trials using the raw trial numbering. The dictionary is populated for ALL complete trials (before lick filtering), so even if a trial is later dropped due to lick artifacts, its reward outcome is available as the "previous" for the next trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from two sources: (1) the `position/data` behavioral variable giving absolute track position in cm, and (2) the reward zone coordinates (A: 80-130, B: 200-250, C: 320-370 cm) determined from the session's scene name parsed from the NWB identifier.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
# ...
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. CONVERSION_NOTES.md Step 5 maps position + trial-specific reward zone to output[0].

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the reward zone is computed: negative if before zone start, zero if inside zone, positive if past zone end.

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

iii. The instructions say "Distance to any location in the reward zone." The AI interprets this as minimum signed distance to the zone boundaries, which is 0 when inside the zone. This is consistent with the discretization bins where bin 3 = "0 cm" (in the zone).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 bins matching the instruction specification exactly.

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
    if np.any(out < 0):
        raise ValueError("Failed to discretize reward-zone distance.")
    return out
```

iii. The bins match the instruction: 0: < -50 cm, 1: -50 to -10 cm, 2: -10 cm to < 0 cm, 3: 0 cm, 4: >0 cm to +10 cm, 5: +10 to +50 cm, 6: > +50 cm.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The same frame mask and trial slice are used for position (which feeds distance computation) and neural data, ensuring frame-by-frame alignment.

ii.
```python
position_trial = position[trial_slice][frame_mask]
# ...
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. Alignment is inherent from the shared frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position/data` behavioral variable giving absolute track position in cm.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
position_trial = position[trial_slice][frame_mask]
```

iii. CONVERSION_NOTES.md Step 5 maps position to output[1].

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The absolute position is clipped to [0, 450) cm and divided into 5 equal bins of 90 cm each.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. The instructions specify "Absolute position in corridor, discretized into 5 equal-sized bins." The track is 450 cm, so 5 equal bins = 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90 and floored to get bin indices 0-4: bin 0 = [0, 90), bin 1 = [90, 180), bin 2 = [180, 270), bin 3 = [270, 360), bin 4 = [360, 450].

ii.
```python
clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. The `np.nextafter(450.0, 0.0)` prevents positions at exactly 450 from being binned to index 5.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame mask and trial slice as neural data.

ii.
```python
position_trial = position[trial_slice][frame_mask]
discretize_absolute_position(position_trial)
```

iii. All outputs share the same per-trial frame grid as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick/data` behavioral variable, which stores cumulative lick counts per frame.

ii.
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
lick_trial = lick[trial_slice][frame_mask]
```

iii. CONVERSION_NOTES.md Step 1 notes the reference code uses `licks[licks > 1] = 1` for binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The cumulative lick count is binarized: any value > 0 becomes 1, otherwise 0.

ii.
```python
(lick_trial > 0).astype(np.int16),
```

iii. The instructions specify "Lick, time-varying. 0 = no, 1 = yes." The reference code clips lick counts with `licks[licks > 1] = 1`, which is functionally equivalent to `> 0` binarization for non-negative integer counts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame mask and trial slice as neural data.

ii.
```python
lick_trial = lick[trial_slice][frame_mask]
(lick_trial > 0).astype(np.int16),
```

iii. Aligned via the shared frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `identifier` field in the NWB file, which encodes the session's scene name (e.g., `Env1_LocationA`, `Env1_LocationA_to_C`). The scene name is parsed to determine reward zone labels (A, B, or C).

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
```
```python
ZONE_TO_CODE = {"A": 0, "B": 1, "C": 2}
```

iii. CONVERSION_NOTES.md Step 5 maps scene-derived zone labels to output[4], consistent with `behavior.get_reward_zones` in the reference code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed using regex patterns to extract the environment and zone labels. For switch sessions (containing "_to_"), the zone switches at trial 30 (matching the reference code's `SWITCH_TRIAL = 30`). The zone label (A=0, B=1, C=2) is broadcast across all timepoints.

ii.
```python
SWITCH_TRIAL = 30

def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        assert scene_info.after_zone is not None
        return scene_info.after_zone
    return scene_info.before_zone
```
```python
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
```

iii. CONVERSION_NOTES.md Step 3 documents "Each switch occurred after 30 trials." The AI handles three scene name formats: single zone, same-environment switch, and cross-environment switch.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from two sources: (1) `Reward/timestamps` providing times of reward delivery events, and (2) `reward_zone/data` indicating reward zone entry events.

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 maps reward delivery within trial to output[5].

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked as rewarded (1) if BOTH: (a) at least one reward timestamp falls within the trial's time window, AND (b) at least one reward_zone entry event (>0) occurs within the trial. Otherwise 0. The value is broadcast across all timepoints.

ii.
```python
def reward_outcome_for_trial(
    reward_timestamps: np.ndarray, t_start: float, t_stop: float,
    reward_zone_segment: np.ndarray,
) -> int:
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)
```

iii. The dual condition (reward timestamp AND reward zone entry) provides a more robust reward determination than using either alone. The reference code's `get_trial_types` function also determines reward from events within trial boundaries.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms handle data issues:
- **NaN/infinite values**: A frame-level validity mask excludes frames where position, speed, lick, environment, or timestamps are NaN/infinite, or where any neuron's deconvolved value is NaN.
- **Sentinel positions**: Frames with position <= -100 are excluded (sentinel value -500 documented in data).
- **Short trials**: Trials with fewer than 5 valid frames are dropped entirely.
- **Lick sensor errors**: Trials with >35% of frames having lick count >2 are dropped.
- **Missing previous trial**: Previous trial outcome defaults to 0 for the first trial.
- **Incomplete trials**: Only trials with both start and teleport markers are included.

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
```
```python
if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. CONVERSION_NOTES.md Step 2 documents sentinel values (-1 for environment/trial number, -500 for position) and Step 10 discusses edge cases including the clipped first trial in m11 day 3.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files with h5py, particularly reading the deconvolved neural traces (large matrices of shape (T, n_neurons)). Multi-plane sessions (m17, m18) take longer due to loading and concatenating data from multiple planes. The full conversion took 3.36 minutes for 152 sessions.

ii.
```python
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. CONVERSION_NOTES.md Step 7 shows ~0.42s per session for the first 2 (small) sessions, with multi-plane sessions taking 1.5-3.5s each.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial processing loop iterates over trial boundaries twice: once to compute trial metadata (reward, zone, lick error) and once to build the actual arrays. Within each iteration, the individual frame-level operations (position clipping, distance computation, discretization) are already vectorized with numpy.

ii.
```python
for start, stop in trial_bounds:
    # first pass: compute metadata
    ...
for meta in trial_meta:
    # second pass: build arrays
    ...
```

iii. The per-trial loop is largely unavoidable since trials have variable lengths and cannot be stacked into a single array. However, the two-pass approach (metadata first, then array construction) could potentially be merged into a single pass.

## 13-c. What processing does the code repeat multiple times?

i. The code processes trial bounds in two passes: (1) first to compute per-trial metadata (trial number, environment, zone, reward, lick error), and (2) second to build the actual neural/input/output arrays for non-lick-error trials. Some computations like accessing the same array slices happen in both passes. The `reward_by_trial_number` dictionary is populated in pass 1 and consumed in pass 2.

ii.
```python
# Pass 1
for start, stop in trial_bounds:
    trial_num = modal_trial_number(trial_number[start:stop])
    env_bin = env_to_binary(environment[start:stop])
    # ...
# Pass 2
for meta in trial_meta:
    start = meta["start"]
    stop = meta["stop"]
    # re-slices the same arrays
```

iii. The two-pass design is intentional: pass 1 must complete before pass 2 so that `reward_by_trial_number` is fully populated for the "previous trial outcome" lookup.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores diagnostic information that is not used in the final pickle:
- Session-level summary dictionaries with detailed counts (stored in metadata but not used by the decoder).
- Example trial data for plotting (up to 3 per session), only used when `--show-processing` is enabled.
- The `speed` output (speed_bin) is computed and included as a decoder output; whether this is "unnecessary" depends on the downstream analysis, but the instructions explicitly request it.

ii.
```python
session_summary = {
    "subject": subject,
    "session_name": session_name,
    # ... extensive metadata
}
if len(kept_examples) < 3:
    kept_examples.append({...})
```

iii. The summary metadata is useful for debugging and validation but adds minimal overhead. The instructions explicitly list speed as a decoder output, so its inclusion is necessary.
