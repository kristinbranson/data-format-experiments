# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing every NWB file under `data/sub-*`, then processing each NWB file as one session with `h5py`. Within each file it reads continuous behavior arrays and deconvolved ophys arrays, then reconstructs trials from `trial_start`/`teleport`.

ii.
```python
def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files

with h5py.File(path, "r") as f:
    behavior_group = f["processing/behavior/BehavioralTimeSeries"]
    ophys_group = f["processing/ophys"]
```

iii. `CONVERSION_NOTES.md` says the release contains one NWB per subject-session and that reading NWB directly with `h5py` is faster. The agent also documented that the released NWB files are the processed, frame-aligned representation it wanted to preserve.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from directory names like `sub-m11`; the subject id is `path.parent.name.replace("sub-", "")`. The final dataset stores sorted unique subject ids and one `subject_idx` per kept session.

ii.
```python
subject = path.parent.name.replace("sub-", "")

subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes explicitly say the file layout is `data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, so subject identity is taken from the directory structure.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a session. Session ordering is the sorted file-path order, and each processed session contributes one entry to `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx`.

ii.
```python
session_id = path.stem.replace("_behavior+ophys", "")
session_name = path.stem

for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
    processed_sessions.append(session)
```

iii. In the notes, the AI states that there is one NWB file per subject-session and that sorting by file path gives stable session ordering.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from continuous behavior streams. Every `trial_start > 0` marks a candidate start, and the first later `teleport > 0` marks the stop; the stop index is exclusive. Only complete start-stop pairs are kept.

ii.
```python
def find_complete_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    ...
    for start in starts:
        while teleport_idx < len(teleports) and teleports[teleport_idx] <= start:
            teleport_idx += 1
        ...
        stop = teleports[teleport_idx]
        if stop > start:
            bounds.append((int(start), int(stop)))
```

iii. `CONVERSION_NOTES.md` says trial boundaries in NWB are represented by continuous `trial_start` and `teleport` vectors and that the converter should match the reference code’s `trial_start` to `teleport` trial-epoch logic while excluding incomplete boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials in three ways: incomplete trials are ignored when building bounds, trials shorter than `MIN_TRIAL_FRAMES = 5` are dropped, and trials with lick-sensor artifacts are dropped if more than 35% of in-trial samples have cumulative lick count above 2. Trials are also dropped if too few valid neural/behavior frames remain after masking invalid samples.

ii.
```python
MIN_TRIAL_FRAMES = 5
LICK_ERROR_FRACTION = 0.35

if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue

def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)

if meta["lick_error"]:
    dropped_lick += 1
    continue

frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. The notes say the AI chose the code-path lick-artifact rule (`>35%`) over the paper text’s `>30%`, and chose to drop incomplete or invalid trials rather than fabricate boundaries or keep NaNs because the validator expects clean arrays.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/Deconvolved/plane*/data`, filtered by the Suite2p `iscell[:, 0] == 1` ROI mask. Plane membership comes from `planeIdx`.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
...
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. The notes say the paper’s decoder uses deconvolved calcium events and that the NWB `Deconvolved` data are the released equivalent of reference `sess.timeseries['events']`.

## 2-b. How is the `neural` data processed?

i. The AI does not recompute dF/F or deconvolution. It reconstructs the accepted-cell deconvolved matrix across planes, filters accepted ROIs, slices each trial, masks invalid frames, and transposes to neuron-by-time.

ii.
```python
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    ...
    deconv[:, dest_cols] = plane_data

neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes justify this by saying the release already provides frame-aligned deconvolved traces, so recomputing dF/F or events would add unnecessary divergence. Multi-plane reconstruction was added after the agent discovered that pooled ROI metadata did not match single-plane assumptions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `iscell[:,0] == 1`, and trial frames are further filtered by requiring all neural values in that frame to be finite. Trials with fewer than 5 valid frames after masking are dropped.

ii.
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
...
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. The notes explicitly justify `iscell` as the shared curated-cell mask available in NWB. The extra finite-frame mask is justified as a way to avoid NaNs in the final dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by trial segmentation itself: each kept trial begins at the `trial_start` frame, ends just before teleport, and the trial-relative time vector is `timestamps[frame] - timestamps[start]`.

ii.
```python
trial_slice = slice(start, stop)
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say the target alignment event is trial start and that the reference trial-definition already excludes teleport periods, so no separate realignment step is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The temporal resolution is the existing frame-aligned sample grid, estimated from the median spacing of behavior timestamps, about 64.5 ms per sample.

ii.
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
...
median_bin_ms = float(
    np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
)
```

iii. The notes say behavior timestamps are the reliable aligned time base, especially because some multi-plane sessions report an ophys `rate` inconsistent with the shared frame grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps, specifically `position/timestamps`, not from the trial-number timestamps.

ii.
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. The notes justify using the common behavior timestamp grid as the reliable aligned time base for both neural and behavior streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI slices the timestamp array to the kept frames and subtracts the start timestamp of that trial.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
...
input_trial = np.vstack([time_trial.astype(np.float32), ...])
```

iii. The justification is straightforward trial-start alignment: the notes say the dataset should preserve the existing frame-aligned time base and express it relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input uses exactly the same `trial_slice` and `frame_mask` as the neural trial, so it is aligned frame-for-frame with `neural`.

ii.
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say the release is already frame-aligned and that the converter should preserve this shared sampling grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw `environment/data` behavior series.

ii.
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
...
env_bin = env_to_binary(environment[start:stop])
```

iii. The notes map NWB `environment` to the reference code’s `morph` variable and document that it is binary `0/1`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI computes a per-trial binary environment by taking the finite, nonnegative in-trial values, then thresholding the median at `0.5`. That scalar is repeated across the trial.

ii.
```python
def env_to_binary(values: np.ndarray) -> int:
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    return int(np.round(np.median(values)) > 0.5)

np.full(time_trial.shape, meta["environment"], dtype=np.float32)
```

iii. The notes say `environment` is a per-trial contextual variable and that repeating per-trial labels across time makes input arrays uniform for the validator.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the raw `trial number/data` behavior series by taking the modal in-trial value, rather than using the loop index over reconstructed trials.

ii.
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)

def modal_trial_number(values: np.ndarray) -> int:
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    counts = np.bincount(values)
    return int(np.argmax(counts))

trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The notes say the variable mapping could use `trial number/data or trial index`, but the implemented code chose the stored trial-number values while still relying on reconstructed trial epochs.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI rounds to integers, discards negative sentinel values, finds the modal value within the trial, and repeats that single integer across the trial.

ii.
```python
values = np.rint(values).astype(np.int64)
values = values[values >= 0]
counts = np.bincount(values)
return int(np.argmax(counts))
...
np.full(time_trial.shape, trial_num, dtype=np.float32)
```

iii. The main justification in the notes is to keep a stable per-trial identifier while preserving the raw frame-aligned metadata when possible.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery timestamps (`Reward/timestamps`) plus the in-trial `reward_zone/data` signal. The AI first computes one reward outcome per reconstructed trial, then looks up the previous trial number’s outcome.

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
...
reward_outcome = reward_outcome_for_trial(
    reward_timestamps=reward_timestamps,
    t_start=float(timestamps[start]),
    t_stop=float(timestamps[stop]),
    reward_zone_segment=reward_zone[start:stop],
)
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. The notes map this variable to prior trial reward outcome and say the first trial should default to `0`. The code also requires that the trial have a reward-zone entry event when assigning a rewarded outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the AI decides whether the previous trial was rewarded by checking whether any reward timestamp fell within that previous trial’s start-stop window and whether the previous trial had `reward_zone > 0` somewhere. The resulting binary value is repeated across the trial; if there is no prior trial number, it uses `0`.

ii.
```python
def reward_outcome_for_trial(...):
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)

prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
np.full(time_trial.shape, prev_outcome, dtype=np.float32)
```

iii. The notes justify the first-trial default as the least-assumptive binary sentinel and tie the reward logic to the reference code’s per-trial reward semantics.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` plus a trial-specific reward-zone identity inferred from the NWB `identifier` scene string, not from the raw `reward_zone` time series. The scene is parsed into pre-switch and post-switch reward zones, and `SWITCH_TRIAL = 30` determines when the zone changes.

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
position_trial = position[trial_slice][frame_mask]
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes explicitly say the `reward_zone` series marks sparse zone-entry events rather than full zone extent, so reward-zone identity should come from session condition metadata consistent with the paper/code reward-zone semantics.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the current trial’s reward-zone edges: negative before the zone, zero inside, positive after.

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

iii. The notes map this directly to reward-relative position logic from the reference code and the paper’s zone coordinates.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is manually thresholded into the 7 requested categories using comparisons at `-50`, `-10`, `0`, `10`, and `50` cm.

ii.
```python
out[distance_cm < -50] = 0
out[(distance_cm >= -50) & (distance_cm < -10)] = 1
out[(distance_cm >= -10) & (distance_cm < 0)] = 2
out[distance_cm == 0] = 3
out[(distance_cm > 0) & (distance_cm <= 10)] = 4
out[(distance_cm > 10) & (distance_cm <= 50)] = 5
out[distance_cm > 50] = 6
```

iii. The notes say these bins were chosen to match the task instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same `position_trial` slice and the same `frame_mask` used for `neural_trial`, so it is aligned frame-for-frame with the neural data.

ii.
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes say the NWB release is already frame-aligned and that trial slicing should preserve that alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `position/data`.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
position_trial = position[trial_slice][frame_mask]
```

iii. The notes map raw position to the target absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position to `[0, 450)` cm, divides by `90` cm, floors to an integer bin index, and caps values above the top bin at `4`.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. The notes justify this as using five equal bins over the 450 cm track, directly following the task instruction rather than the shifted `-50..450` range used elsewhere in the release.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `0-89.999`, `90-179.999`, `180-269.999`, `270-359.999`, and `360-449.999` cm after clipping to `[0, 450)`.

ii.
```python
clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. The notes say this was an intentional interpretation of “5 equal-sized bins.”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial frame slice and `frame_mask` as neural data.

ii.
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The justification is the shared frame-aligned NWB sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick/data` behavior series.

ii.
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
lick_trial = lick[trial_slice][frame_mask]
```

iii. The notes map `lick` directly to the target lick output after trial-level artifact handling.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the per-frame lick values with `lick > 0`, but only after dropping whole trials that fail the lick-sensor quality rule.

ii.
```python
if meta["lick_error"]:
    dropped_lick += 1
    continue

(lick_trial > 0).astype(np.int16)
```

iii. The notes say the reference code treats bad lick trials as invalid for licking analyses and that the validator should not receive NaNs, so it drops those trials entirely.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by using the same `trial_slice` and `frame_mask` as neural data.

ii.
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes justify this with the same frame-aligned NWB time base used throughout the conversion.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string parsed into environment/reward-zone condition metadata, not from the `reward_zone` timeseries itself.

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. The notes explicitly say reward-zone identity should come from scene/reward-zone metadata consistent with `behavior.get_reward_zones`, because the raw `reward_zone` series only marks sparse entry events.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses scene names for single-condition, same-environment switch, and cross-environment switch sessions; applies a fixed `SWITCH_TRIAL = 30` rule for switch days; maps zone labels `A/B/C` to integer codes `0/1/2`; and repeats the result across the trial.

ii.
```python
SWITCH_TRIAL = 30
ZONE_TO_CODE = {"A": 0, "B": 1, "C": 2}

def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        return scene_info.after_zone
    return scene_info.before_zone

np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16)
```

iii. The notes say this follows the paper/code reward-zone semantics, including the switch occurring after 30 trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` plus the in-trial `reward_zone/data` signal. A trial is labeled rewarded only if a reward timestamp falls inside the trial window and the trial contains a reward-zone entry sample.

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
...
reward_outcome = reward_outcome_for_trial(..., reward_zone_segment=reward_zone[start:stop])
```

iii. The notes say reward outcome should match per-trial reward logic from the reference behavior code; the implementation adds the requirement that a valid reward-zone entry occurred in that trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI uses `searchsorted` to count whether any reward timestamps fall in `[t_start, t_stop)`, checks that `reward_zone > 0` appears somewhere in the trial, converts the result to `0/1`, and repeats it across the trial.

ii.
```python
left = np.searchsorted(reward_timestamps, t_start, side="left")
right = np.searchsorted(reward_timestamps, t_stop, side="left")
has_reward = right > left
has_rzone_entry = np.any(reward_zone_segment > 0)
return int(has_reward and has_rzone_entry)
...
np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16)
```

iii. In Step 12 of the notes, the agent says it spot-checked raw NWB trials after training and kept this label definition unchanged because it believed the labels were correct even though decode performance was weak.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles bad or missing data by filtering rather than interpolating. It drops incomplete trials, drops lick-artifact trials, masks invalid neural/behavior frames inside each trial, drops trials with too few remaining valid frames, ignores negative/invalid environment and trial-number sentinels when computing per-trial labels, and explicitly fixed the multi-plane loading bug by reconstructing pooled ROI order.

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
values = values[np.isfinite(values)]
values = values[values >= 0]
...
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. The notes repeatedly justify this as avoiding NaNs in the validator and preserving only complete, frame-aligned data rather than fabricating corrections.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the full deconvolved matrices from each NWB session, reconstructing pooled accepted-cell matrices across planes, and then looping through all trials to build per-trial arrays. Optional diagnostic plotting is additional overhead.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    for plane in planes:
        ...
        plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
        ...

for meta in trial_meta:
    ...
    neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The notes explicitly list full-session deconvolved loading as a remaining inefficiency and report that the full conversion runtime is dominated by per-session processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `process_session` could be vectorized or partially precomputed, especially the first pass that builds `trial_meta` and the second pass that slices trials and constructs repeated label arrays. The loop over planes is structurally necessary, but destination-column lookup could also be precomputed once.

ii.
```python
for start, stop in trial_bounds:
    ...
    trial_meta.append({...})

for meta in trial_meta:
    ...
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The notes do not give a detailed vectorization plan, but they do say the code still loads and processes whole sessions sequentially and that some trial-level work could be sped up further.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial slicing and masking logic across multiple outputs, repeats per-trial scalar label expansion with `np.full`, and repeats trial metadata in the `kept_examples` block for plotting. It also computes session summaries that duplicate information already derivable from the final arrays.

ii.
```python
trial_slice = slice(start, stop)
position_trial = position[trial_slice][frame_mask]
speed_trial = speed[trial_slice][frame_mask]
lick_trial = lick[trial_slice][frame_mask]
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)

np.full(time_trial.shape, meta["environment"], dtype=np.float32)
np.full(time_trial.shape, trial_num, dtype=np.float32)
np.full(time_trial.shape, prev_outcome, dtype=np.float32)
```

iii. The notes mention session-level summary metadata and optional example plots as added convenience features, not core conversion requirements.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter builds `kept_examples` for plotting, records detailed `session_summary` metadata inside `metadata["session_info"]`, and contains plotting helpers plus `output_values` plumbing in `build_trial_plot` that are not needed for downstream decoder training. These artifacts are useful for validation but not consumed by the converted dataset itself.

ii.
```python
kept_examples: list[dict] = []
...
if len(kept_examples) < 3:
    kept_examples.append({...})

session_summary = {
    "subject": subject,
    "session_name": session_name,
    ...
}
...
"session_info": [sess["summary"] for sess in processed_sessions],
```

iii. The notes describe these as sanity-check and documentation aids added during validation rather than part of the essential reference processing.
