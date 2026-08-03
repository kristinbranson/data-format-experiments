# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file matching `data/sub-*/sub-*_behavior+ophys.nwb`, processes one file per session, and reads the needed behavioral and ophys arrays directly with `h5py` instead of `pynwb`.

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
```

iii. In `CONVERSION_NOTES.md`, the AI says the release contains one NWB per subject-session and that direct NWB access with `h5py` was chosen for speed while preserving the same released content.

## 1-b. How are the data split into subjects?

i. Subjects are the `sub-<mouse>` directory names. Each session stores a single `subject`, and the final dataset deduplicates and sorts them into `subjects` plus `subject_idx`.

ii. 
```python
subject = path.parent.name.replace("sub-", "")
```

```python
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes explicitly state that file paths are sorted by subject/session for stable subject ordering and reproducible `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The main loop calls `process_session(path)` once per file and appends one session entry to the final dataset.

ii. 
```python
for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
    ...
    processed_sessions.append(session)
```

iii. `CONVERSION_NOTES.md` says the released data layout is one NWB file per subject-session, so one file naturally maps to one session.

## 1-d. How are the data split into trials?

i. Trials are segmented from `trial_start > 0` to the next later `teleport > 0`. Only complete start/stop pairs are kept, and the stop index is exclusive.

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
```

iii. The notes say this matches the reference trial logic conceptually and that complete start/end markers are preferred so incomplete boundaries are not fabricated.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials shorter than 5 frames, drops trials flagged as lick-sensor artifacts, and also drops trials whose valid-frame mask leaves fewer than 5 retained frames.

ii. 
```python
MIN_TRIAL_FRAMES = 5
LICK_ERROR_FRACTION = 0.35
```

```python
if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
...
if meta["lick_error"]:
    dropped_lick += 1
    continue
...
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. In the notes, the AI justifies the lick-artifact rule from the paper/code discrepancy discussion and says it dropped bad-lick trials entirely because the validator does not permit NaNs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB deconvolved calcium traces in `processing/ophys/Deconvolved/plane*/data`, filtered by the ROI curation flag `iscell[:, 0]` and ordered using `planeIdx`.

ii. 
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
```

```python
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. The notes say the released NWB `Deconvolved` traces are the direct equivalent of the reference `sess.timeseries['events']`, so the AI chose not to recompute dF/F or deconvolution.

## 2-b. How is the `neural` data processed?

i. The AI reconstructs one session-level accepted-cell matrix by concatenating accepted ROIs from each plane into pooled ROI order, then slices it by trial and transposes each trial to `(neurons, time)`.

ii. 
```python
deconv_shape_t = None
deconv = None
planes = sorted(int(x) for x in np.unique(plane_idx_all))
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    ...
    dest_cols = np.searchsorted(accepted_idx, accepted_total_idx)
    deconv[:, dest_cols] = plane_data
```

```python
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say this multi-plane pooling was required to match the released pooled ROI order, especially for `m17`/`m18`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `iscell[:, 0] == 1`. In addition, timepoints are filtered with `valid_neural_frames = np.all(np.isfinite(deconv), axis=1)`, and trials can be dropped if too few valid frames remain.

ii. 
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
```

```python
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
```

iii. The notes explicitly justify `iscell` as the shared curated-cell mask and say invalid samples were masked jointly with behavior because the output format should not contain NaNs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by trial slicing. For each `(start, stop)` trial, the same frame slice is used for neural and behavioral variables, and time is expressed relative to the trial start.

ii. 
```python
trial_slice = slice(start, stop)
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say alignment should be to `trial_start`, and because all streams are already frame-aligned in the NWB release, trial slicing is the main alignment step.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the original frame-aligned sampling grid, with bin size taken from the median spacing of behavioral timestamps. No temporal rebinning or resampling is applied.

ii. 
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
```

```python
median_bin_ms = float(
    np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
)
...
"time_bin_size": median_bin_ms,
```

iii. The notes say behavior timestamps are the reliable aligned time base, especially because multi-plane sessions can have misleading ophys `rate` metadata.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps, specifically `position/timestamps`.

ii. 
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
```

iii. The notes say all behavioral series share the same frame-aligned time base, so one behavior timestamp vector is sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each kept trial, the AI slices the timestamps, applies the same valid-frame mask used for neural data, and subtracts the trial start timestamp.

ii. 
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

```python
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        ...
    ]
)
```

iii. The notes justify using actual timestamps rather than inferred rates and describe this variable as the frame-aligned time from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same `trial_slice` and the same `frame_mask` as the neural data for each trial.

ii. 
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes repeatedly state that the NWB release already contains behavior and neural data on one frame-aligned grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. 
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
```

iii. The notes map this variable to the paper/code `morph`-style binary environment identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI removes invalid values, takes a median-based binary environment label, and repeats that value across all kept timepoints in the trial.

ii. 
```python
def env_to_binary(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    ...
    return int(np.round(np.median(values)) > 0.5)
```

```python
np.full(time_trial.shape, meta["environment"], dtype=np.float32),
```

iii. The notes say environment is treated as a per-trial binary contextual variable and repeated across timepoints for uniform `(d, T)` input arrays.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw `trial number` behavior series by taking the modal nonnegative value within each segmented trial.

ii. 
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
```

```python
def modal_trial_number(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    ...
    return int(np.argmax(counts))
```

```python
trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The notes describe this as using the per-trial integer id on the complete-trial segmentation and mention a clipped first marker in `m11` day 3 as an edge case they checked.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI converts the within-trial `trial number` values to integers, takes the mode, then repeats that trial id across all timepoints in the kept trial.

ii. 
```python
trial_num = modal_trial_number(trial_number[start:stop])
...
np.full(time_trial.shape, trial_num, dtype=np.float32),
```

iii. The notes justify repeating per-trial labels across time to keep all input arrays 2D and uniform for the validator.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery timestamps in `Reward/timestamps`, combined with the current trial segmentation and a check that the trial contains a `reward_zone` entry event.

ii. 
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```

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

iii. The notes say this follows the reference reward-outcome semantics more closely than using timestamps alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary `reward_outcome` for each complete trial, stores it by trial number, then for each kept trial looks up `trial_num - 1`; if there is no previous trial, it uses `0`. The value is repeated across time.

ii. 
```python
reward_by_trial_number[trial_num] = reward_outcome
```

```python
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
...
np.full(time_trial.shape, prev_outcome, dtype=np.float32),
```

iii. The notes explicitly justify the first-trial default as `0` because there is no previous within-session trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-frame `position` and a trial-specific reward-zone label inferred from the session `identifier` scene string plus a hard-coded switch-after-trial-30 rule.

ii. 
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
```

```python
def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        ...
        return scene_info.after_zone
    return scene_info.before_zone
```

```python
position = behavior_group["position/data"][()].astype(np.float32)
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. The notes explicitly say the AI chose paper/code reward-zone semantics from scene metadata instead of inferring zones from sparse `reward_zone` events alone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest reward-zone edge: negative before the zone, `0` inside, positive after the zone, then discretizes the result.

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

```python
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes say this follows the reward-relative position semantics from the paper/code while adapting the output to the requested decoder bins.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded with explicit inequalities into the 7 requested categories.

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
```

iii. The notes list the same seven bins and say they were chosen to match the task instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by applying the same `trial_slice` and `frame_mask` to the position series that are applied to neural data before distance computation.

ii. 
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes say the NWB data are already frame-aligned, so shared indexing is the alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. 
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
position_trial = position[trial_slice][frame_mask]
```

iii. The notes treat raw VR position as the source signal for this output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position to the interval `[0, 450)` and maps it to 5 equal 90 cm bins using `floor(position / 90)`.

ii. 
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. In the notes, the AI explicitly says it chose 5 equal bins over the 450 cm corridor because that matches the task statement most literally.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are the 5 integer bins returned by the `[0, 450)` to `0..4` mapping above.

ii. 
```python
output_trial = np.vstack(
    [
        ...,
        discretize_absolute_position(position_trial),
        ...
    ]
)
```

iii. The notes describe these as equal-sized corridor bins and record them in `output_values` as `bin0` through `bin4`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing the same trial window and applying the same valid-frame mask before binning.

ii. 
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes again rely on the shared frame-aligned NWB sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. 
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
```

iii. The notes interpret this NWB field as cumulative lick-count samples that must be binarized for the decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. First, trials are screened for lick-sensor artifacts using the `>35% of samples > 2` rule. Kept trials are then binarized with `(lick_trial > 0)`.

ii. 
```python
def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
```

```python
if meta["lick_error"]:
    dropped_lick += 1
    continue
...
(lick_trial > 0).astype(np.int16),
```

iii. The notes say this was chosen to mirror the released code’s lick correction and to avoid NaNs in the final dataset.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the same trial interval and applying the same valid-frame mask as the neural data.

ii. 
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say all behavioral variables were kept on the same frame-aligned grid as neural activity.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string, parsed into pre-switch and post-switch reward-zone labels, then combined with trial number to choose the trial’s zone.

ii. 
```python
SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")
```

```python
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. The notes say this follows `behavior.get_reward_zones` semantics and avoids trying to recover full zone identity from sparse reward-zone entry pulses.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string, applies the switch-after-trial-30 rule when needed, maps `A/B/C` to `0/1/2`, and repeats the categorical label across timepoints.

ii. 
```python
SWITCH_TRIAL = 30
ZONE_TO_CODE = {"A": 0, "B": 1, "C": 2}
```

```python
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
```

iii. The notes explicitly state that switch sessions use the pre-switch zone for trials `<30` and the post-switch zone for trials `>=30`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, with trial windows from `trial_start`/`teleport`, plus a requirement that the trial also contains a `reward_zone` entry event.

ii. 
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```

```python
reward_outcome = reward_outcome_for_trial(
    reward_timestamps=reward_timestamps,
    t_start=float(timestamps[start]),
    t_stop=float(timestamps[stop]),
    reward_zone_segment=reward_zone[start:stop],
)
```

iii. The notes say this was intended to match the reference reward semantics more closely than using reward timestamps without context.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward outcome is computed per complete trial by testing whether any reward timestamp falls within `[t_start, t_stop)` and whether the trial entered the reward zone. The binary result is repeated across timepoints.

ii. 
```python
left = np.searchsorted(reward_timestamps, t_start, side="left")
right = np.searchsorted(reward_timestamps, t_stop, side="left")
has_reward = right > left
has_rzone_entry = np.any(reward_zone_segment > 0)
return int(has_reward and has_rzone_entry)
```

```python
np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16),
```

iii. The notes say reward outcome is a per-trial categorical variable, so repeating it across time was a deliberate format choice.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues defensively: it ignores invalid behavior samples (`NaN`, negative sentinels, implausible position `< -100`), ignores neural frames with nonfinite values, skips incomplete trials with no matching teleport, drops empty lick segments, and drops trials with too few remaining valid frames.

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
if teleport_idx >= len(teleports):
    break
...
if lick_segment.size == 0:
    return True
```

iii. The notes say the converter should preserve the released frame alignment but avoid NaNs and fabricated boundaries, so it prefers masking and dropping invalid data over imputation.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session-level NWB I/O, reconstructing the full accepted-cell deconvolved matrix across planes, and then iterating over all trials to build masked per-trial arrays. Optional plotting also adds cost when enabled.

ii. 
```python
with h5py.File(path, "r") as f:
    ...
    for plane in planes:
        ...
        plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

```python
for meta in trial_meta:
    ...
    neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes explicitly mention that full-session deconvolved matrices are still loaded into memory one session at a time and give runtime estimates dominated by per-session processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The two biggest candidates are the per-plane reconstruction loop and the second per-trial loop that repeatedly slices arrays, computes masks, and stacks outputs. Some of the per-trial discretization could be done session-wide before splitting.

ii. 
```python
for plane in planes:
    ...
    deconv[:, dest_cols] = plane_data
```

```python
for meta in trial_meta:
    ...
    distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The notes do not spell this out in detail, but the code structure makes the trial-building pass the main vectorization target.

## 13-c. What processing does the code repeat multiple times?

i. The code makes two passes over trial metadata: one pass computes per-trial metadata and reward outcomes, and a second pass re-slices the same trials to build final arrays. It also re-slices reward timestamps again for plotting examples.

ii. 
```python
for start, stop in trial_bounds:
    ...
    reward_outcome = reward_outcome_for_trial(...)
    trial_meta.append({...})
```

```python
for meta in trial_meta:
    ...
    position_trial = position[trial_slice][frame_mask]
    speed_trial = speed[trial_slice][frame_mask]
    lick_trial = lick[trial_slice][frame_mask]
```

iii. The notes emphasize speed and simplicity over maximal optimization, and they explicitly say diagnostic examples are collected during the same pass for sanity checking.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds `kept_examples` and optional diagnostic plots for human inspection, stores detailed `session_summary` metadata such as `trial_numbers_kept`, and computes plot-only reward-time offsets. None of those objects are used by the downstream decoder itself.

ii. 
```python
kept_examples: list[dict] = []
...
if len(kept_examples) < 3:
    trial_reward_times = reward_timestamps[
        (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
    ] - timestamps[start]
    kept_examples.append({...})
```

```python
session_summary = {
    ...
    "trial_numbers_kept": [int(np.rint(x[0, 0])) for x in input_trials] if input_trials else [],
}
```

```python
if args.show_processing and plots_made < 2:
    build_trial_plot(...)
```

iii. The notes justify these as validation/sanity-check artifacts rather than part of the final learning representation.
