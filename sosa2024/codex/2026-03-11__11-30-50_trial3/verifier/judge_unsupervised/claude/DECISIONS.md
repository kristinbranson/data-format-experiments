# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored in `data/sub-*/sub-*_behavior+ophys.nwb`. Each NWB file is one session. It uses `h5py` to read behavioral time series from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys`. All 152 files are discovered via glob and processed sequentially.

ii.
```python
def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files

# In process_session:
with h5py.File(path, "r") as f:
    behavior_group = f["processing/behavior/BehavioralTimeSeries"]
    ophys_group = f["processing/ophys"]
    position = behavior_group["position/data"][()].astype(np.float32)
    speed = behavior_group["speed/data"][()].astype(np.float32)
    lick = behavior_group["lick/data"][()].astype(np.float32)
    environment = behavior_group["environment/data"][()].astype(np.float32)
    # ... etc
```

iii. The AI noted that the released data are NWB files (not the `sess` pickle objects used by the reference code), but contain equivalent content. The AI chose h5py for speed over higher-level NWB libraries. This is documented in CONVERSION_NOTES Steps 2 and 4.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the parent directory name of each NWB file (e.g., `sub-m11` -> `m11`). A sorted unique list of subjects is built, and each session is assigned a `subject_idx` mapping into this list.

ii.
```python
subject = path.parent.name.replace("sub-", "")
# In build_dataset:
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. CONVERSION_NOTES Step 2 documents 11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed one at a time. All 152 sessions are kept (none are dropped at the session level as long as they have >= 2 valid trials).

ii.
```python
for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
    if len(session["neural"]) < 2:
        print(f"Skipping {path.name}: only {len(session['neural'])} valid trials")
        continue
    processed_sessions.append(session)
```

iii. CONVERSION_NOTES Step 4 confirms 152 sessions = 10 mice x 14 days + m11 x 12 days.

## 1-d. How are the data split into trials?

i. Trials are defined as intervals from a `trial_start > 0` frame to the next `teleport > 0` frame. The function `find_complete_trial_bounds` pairs each start with its next teleport to define complete trial boundaries.

ii.
```python
def find_complete_trial_bounds(trial_start, teleport):
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    bounds = []
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

iii. CONVERSION_NOTES Step 1 notes that the reference code uses `trial_start_inds` and `teleport_inds` to delimit trials, and Step 4 confirms this approach is consistent.

## 1-e. How are trials filtered based on quality controls?

i. Two types of trial filtering: (1) trials with fewer than `MIN_TRIAL_FRAMES=5` frames are dropped, (2) trials with lick sensor errors (>35% of frames having cumulative lick count >2) are dropped. Additionally, within kept trials, individual frames with invalid behavior or neural data are masked out.

ii.
```python
LICK_ERROR_FRACTION = 0.35
MIN_TRIAL_FRAMES = 5

def has_lick_sensor_error(lick_segment):
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)

# In process_session:
if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1; continue
# ...
if meta["lick_error"]:
    dropped_lick += 1; continue
# Frame-level masking:
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1; continue
```

iii. CONVERSION_NOTES Step 4 documents the code vs. paper discrepancy (code uses 35%, paper says 30%) and chose to follow the code threshold. The agent explicitly noted "Prefer the released code threshold (35%) when reproducing continuous-sample processing."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the deconvolved calcium traces stored in `processing/ophys/Deconvolved/plane{N}/data`, filtered by `iscell[:,0] == 1`.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
# ...
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. CONVERSION_NOTES Step 1 documents that the reference code uses `sess.timeseries['events']` (deconvolved activity), and Step 5 maps NWB `Deconvolved` to this.

## 2-b. How is the `neural` data processed?

i. Deconvolved traces are loaded directly from NWB without further processing (no dF/F recomputation, no smoothing, no additional deconvolution). Only `iscell` filtering and trial segmentation are applied. For multi-plane sessions, per-plane data are concatenated in pooled ROI order.

ii.
```python
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    plane_data = plane_data.astype(np.float32, copy=False)
    dest_cols = np.searchsorted(accepted_idx, accepted_total_idx)
    deconv[:, dest_cols] = plane_data

# Per trial:
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 key decision: "Use NWB deconvolved traces directly" and "Do not recompute dF/F from fluorescence."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using `iscell[:,0] == 1` (suite2p manual curation). No additional filtering for putative interneurons (speed correlation > 0.5) is applied.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
```

iii. CONVERSION_NOTES Step 5 key decision 2: "Filter neurons with iscell[:,0] == 1 only." Step 4 acknowledges the paper excludes putative interneurons by speed correlation > 0.5 but notes "NWB does not contain a ready-made interneuron mask."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data starts at the `trial_start` frame index and ends at the `teleport` frame index. Frames with invalid data are masked out. Time is measured from the first frame's timestamp.

ii.
```python
trial_slice = slice(start, stop)
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
# Time alignment:
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. CONVERSION_NOTES Step 5 documents alignment to trial start, consistent with the instruction "Temporally align based on start of the trial."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native frame rate of ~15.5 Hz (~64.5 ms per frame) is preserved. No temporal rebinning is applied. The metadata `time_bin_size` is set to the median inter-frame interval across sessions.

ii.
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
# In build_dataset:
median_bin_ms = float(
    np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
)
```

iii. CONVERSION_NOTES Step 3 documents "~15.5 Hz (~0.0645 s per sample)" from the paper. The agent preserves this native resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
# Per trial:
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. CONVERSION_NOTES Step 5 maps this to `input[0]` as "timestamps[start:stop] - timestamps[start]".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from each frame's timestamp within the trial. Invalid frames (NaN behavior/neural values, position < -100) are removed by frame masking.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. Documented in CONVERSION_NOTES Step 5 variable mapping table.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `frame_mask` is applied to both the timestamps and the neural data within each trial, so they are inherently aligned sample-by-sample.

ii.
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The NWB data already shares the same frame-aligned sampling grid for behavior and neural data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 maps this to `input[1]`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial environment is computed as the rounded median of valid (finite, >= 0) environment values within the trial, thresholded at 0.5 to produce a binary 0/1 value. This is then broadcast across all timepoints in the trial.

ii.
```python
def env_to_binary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    return int(np.round(np.median(values)) > 0.5)

# Per trial input:
np.full(time_trial.shape, meta["environment"], dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 documents this as matching `behavior.get_trial_types` (`morph`).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/trial number/data`.

ii.
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 maps this to `input[2]`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The modal (most frequent) value of valid trial number samples within each trial is computed and used as the per-trial trial number. This is broadcast across all timepoints.

ii.
```python
def modal_trial_number(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    counts = np.bincount(values)
    return int(np.argmax(counts))

# Per trial input:
np.full(time_trial.shape, trial_num, dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 notes "Use per-trial integer index (0-based within session), repeated across timepoints."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward outcome computed for the previous trial (by trial number) within the same session.

ii.
```python
reward_by_trial_number[trial_num] = reward_outcome
# Later:
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. CONVERSION_NOTES Step 5 key decision 11: "Set first-trial previous outcome to 0."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A dictionary `reward_by_trial_number` stores the reward outcome for each trial number. For each trial, the previous trial number's outcome is looked up. If not found (first trial), defaults to 0. The value is broadcast across all timepoints.

ii.
```python
reward_by_trial_number: dict[int, int] = {}
# In first pass over trial_bounds:
reward_by_trial_number[trial_num] = reward_outcome
# In second pass:
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
np.full(time_trial.shape, prev_outcome, dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 key decision 11 documents this approach.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` (absolute position on track) and the reward zone coordinates (A: 80-130, B: 200-250, C: 320-370 cm), which are determined from the session scene metadata.

ii.
```python
ZONE_TO_COORDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. CONVERSION_NOTES Step 3 documents zone coordinates from the paper.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: if position < zone_start, distance = position - zone_start (negative); if position > zone_end, distance = position - zone_end (positive); if inside zone, distance = 0.

ii.
```python
def signed_distance_to_zone(position_cm, zone_start, zone_end):
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. CONVERSION_NOTES Step 5 describes the signed distance computation from reward-zone start.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins: 0 (<-50), 1 (-50 to -10), 2 (-10 to 0), 3 (0), 4 (0 to 10), 5 (10 to 50), 6 (>50).

ii.
```python
def discretize_distance(distance_cm):
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

iii. Matches the instruction specification for distance discretization bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `frame_mask` is applied to position data and neural data, ensuring sample-by-sample alignment.

ii.
```python
position_trial = position[trial_slice][frame_mask]
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. All time-varying signals share the same frame-aligned sampling grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
position_trial = position[trial_slice][frame_mask]
```

iii. CONVERSION_NOTES Step 5 maps position to `output[1]`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450) and divided into 5 equal 90 cm bins via floor division.

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. Matches instruction: "Absolute position in corridor, discretized into 5 equal-sized bins."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal bins of 90 cm each: bin0 (0-90), bin1 (90-180), bin2 (180-270), bin3 (270-360), bin4 (360-450).

ii. Same as 8-b above.

iii. Track length is 450 cm per the paper, divided into 5 equal bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `frame_mask` applied to both position and neural data within each trial.

ii.
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. Frame-aligned sampling grid shared across all signals.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 maps lick to `output[3]`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are binarized: any value > 0 becomes 1, otherwise 0. Trials with lick sensor errors (>35% of frames with lick > 2) are dropped entirely.

ii.
```python
(lick_trial > 0).astype(np.int16),
```

iii. CONVERSION_NOTES Step 1 notes that the reference code clips licks to binary with `licks[licks > 1] = 1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `frame_mask` applied to lick and neural data within each trial.

ii.
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. Frame-aligned sampling grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` string which encodes the session scene name (e.g., `Env1_LocationA`, `Env1_LocationA_to_C`), parsed to determine which reward zone (A, B, or C) applies for each trial.

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
# ...
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. CONVERSION_NOTES Step 1 documents `behavior.get_reward_zones` as the reference function. Step 5 key decision 8: "Use paper/code reward-zone semantics rather than inferring from sparse reward_zone events alone."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regex to determine pre-switch and post-switch zone labels. For switch sessions, trials with trial number >= 30 use the post-switch zone. Zone labels are encoded as A=0, B=1, C=2.

ii.
```python
SWITCH_TRIAL = 30
ZONE_TO_CODE = {"A": 0, "B": 1, "C": 2}

def zone_for_trial(scene_info, trial_number):
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        return scene_info.after_zone
    return scene_info.before_zone

# Output:
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
```

iii. CONVERSION_NOTES Step 3 documents "Each switch occurred after 30 trials" from the paper.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` and `reward_zone/data`.

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 maps reward outcome to `output[5]`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded (1) if any reward delivery timestamp falls within the trial's time window AND the trial has a reward zone entry event (reward_zone > 0). Otherwise 0.

ii.
```python
def reward_outcome_for_trial(reward_timestamps, t_start, t_stop, reward_zone_segment):
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)
```

iii. CONVERSION_NOTES Step 5: "Trial is rewarded if any reward delivery occurs within the trial and the trial has an rzone entry event."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Sentinel values (position < -100, environment = -1, NaN) are detected and masked at the frame level. (2) Trials with too few valid frames (< 5) are dropped. (3) Lick sensor error trials are dropped entirely. (4) The first trial's previous outcome defaults to 0. (5) Incomplete trials without both start and teleport markers are excluded.

ii.
```python
valid_behavior_frames = (
    np.isfinite(position) & np.isfinite(speed) & np.isfinite(lick)
    & np.isfinite(environment) & np.isfinite(timestamps) & (position > -100.0)
)
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1; continue
```

iii. CONVERSION_NOTES Step 4 documents sentinel values (-500 for position, -1 for environment/trial number). Step 5 key decision 10: "Drop lick-artifact trials entirely rather than leaving NaNs."

## 13-a. What are the most time-consuming steps of the code?

i. Loading the deconvolved neural data from NWB files is the main bottleneck. The full conversion of 152 sessions took 3.36 minutes, with larger sessions (m18 with 2000+ neurons) taking ~2-3 seconds each.

ii.
```python
# Per-plane data loading:
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. CONVERSION_NOTES Step 7 documents ~0.42 s/session for sample sessions and estimates ~64s total for all 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial processing loop iterates over each trial bound sequentially, building per-trial arrays one at a time. The `find_complete_trial_bounds` function also uses a sequential loop to pair starts with teleports. These could potentially be vectorized with numpy operations.

ii.
```python
for start, stop in trial_bounds:
    # sequential processing of each trial
    ...
for meta in trial_meta:
    # sequential building of per-trial arrays
    ...
```

iii. CONVERSION_NOTES Step 6 notes the code processes sessions sequentially. The actual runtime was only 3.36 minutes, suggesting vectorization was not critical.

## 13-c. What processing does the code repeat multiple times?

i. The code iterates over `trial_bounds` twice: once to build `trial_meta` (computing trial metadata and reward outcomes), and again to build the actual neural/input/output arrays. The reward outcome and zone labels are computed in the first pass and stored for lookup in the second pass.

ii.
```python
# First pass:
for start, stop in trial_bounds:
    trial_num = modal_trial_number(...)
    reward_outcome = reward_outcome_for_trial(...)
    trial_meta.append({...})

# Second pass:
for meta in trial_meta:
    prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
    # build arrays...
```

iii. This two-pass design is deliberate: the first pass builds the reward history needed to compute `previous_trial_rewarded` in the second pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores diagnostic `kept_examples` (up to 3 per session) for plotting purposes even when `--show-processing` is not used. The `session_summary` metadata includes detailed per-session statistics that are stored in the pickle but not used by the decoder. The code also loads the `reward_zone` behavioral signal for reward validation even though it is only used for a secondary check in `reward_outcome_for_trial`.

ii.
```python
if len(kept_examples) < 3:
    kept_examples.append({...})  # Always computed even without --show-processing
```

iii. No explicit justification in CONVERSION_NOTES for this extra processing.
