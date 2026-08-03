# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `data/sub-*/sub-*_behavior+ophys.nwb`, sorts the file list, and processes one NWB file per session with `h5py`. Within each file it reads the behavior time series, reward timestamps, ROI curation arrays, plane assignments, and deconvolved neural activity. Trials are not loaded from an NWB `trials` table; they are reconstructed later from continuous `trial_start` and `teleport` streams.

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

```python
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

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent justified this as using the released NWB files as the available representation of the already aligned session data, rather than trying to recreate the original `sess` pickle pipeline. The notes say the NWB `Deconvolved` arrays are treated as the released equivalent of `sess.timeseries['events']`.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file, with `sub-` stripped off. The final dataset stores a sorted unique `subjects` list and a `subject_idx` entry for each session.

ii. 
```python
subject = path.parent.name.replace("sub-", "")
```

```python
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_to_idx[sess["subject"]] for sess in processed_sessions],
    dtype=np.int64,
),
```

iii. The notes explicitly say the agent chose stable session ordering by sorted file path so `subjects` and `subject_idx` would be reproducible.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order is the sorted file order from `list_nwb_files`, and each call to `process_session(path)` produces one session entry in `neural`, `input`, and `output`.

ii. 
```python
for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
    ...
    processed_sessions.append(session)
```

```python
"neural": [sess["neural"] for sess in processed_sessions],
"input": [sess["input"] for sess in processed_sessions],
"output": [sess["output"] for sess in processed_sessions],
```

iii. The notes describe the released dataset as one NWB file per subject-session and state that stable file sorting was a deliberate choice.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the continuous `trial_start` and `teleport` series. The helper pairs each positive `trial_start` sample with the next positive `teleport` sample after it, and the kept trial interval is `[start, stop)`, meaning trial start inclusive and teleport exclusive.

ii. 
```python
def find_complete_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    ...
    for start in starts:
        while teleport_idx < len(teleports) and teleports[teleport_idx] <= start:
            teleport_idx += 1
        if teleport_idx >= len(teleports):
            break
        stop = teleports[teleport_idx]
        if stop > start:
            bounds.append((int(start), int(stop)))
```

```python
trial_bounds = find_complete_trial_bounds(trial_start, teleport)
...
trial_slice = slice(start, stop)
```

iii. In the notes, the agent says this was chosen to match the reference `trial_start_inds` and `teleport_inds` semantics and to exclude teleport periods.

## 1-e. How are trials filtered based on quality controls?

i. The script drops three classes of trials: trials shorter than `MIN_TRIAL_FRAMES = 5`, trials flagged as lick-sensor failures when more than 35% of lick samples exceed 2, and trials with fewer than 5 valid samples remaining after finite-value and sentinel filtering. The filtering is global: once a trial is dropped, it is removed from all neural/input/output data, not just lick analyses.

ii. 
```python
LICK_ERROR_FRACTION = 0.35
MIN_TRIAL_FRAMES = 5
```

```python
if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
...
"lick_error": has_lick_sensor_error(lick[start:stop]),
```

```python
for meta in trial_meta:
    if meta["lick_error"]:
        dropped_lick += 1
        continue
    ...
    frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
    if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
        dropped_missing += 1
        continue
```

iii. The notes justify the 35% lick rule as following the released code rather than the paper text's 30% threshold. They also say the agent chose to drop lick-artifact trials entirely because the validator disallows NaNs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken from the NWB deconvolved calcium traces in `processing/ophys/Deconvolved/plane*/data`, plus the ROI curation arrays `iscell` and `planeIdx` used to select and reorder accepted cells.

ii. 
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
```

```python
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
...
deconv[:, dest_cols] = plane_data
```

iii. The notes say the agent intentionally used the released deconvolved traces as the NWB equivalent of `sess.timeseries['events']`, rather than recomputing them from fluorescence.

## 2-b. How is the `neural` data processed?

i. Neural data are processed by selecting accepted ROIs, reconstructing a pooled accepted-cell matrix across one or more imaging planes, then slicing it by trial and transposing from time-by-neuron to neuron-by-time. No new dF/F or deconvolution is computed.

ii. 
```python
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    ...
    deconv[:, dest_cols] = plane_data
```

```python
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The trajectory shows this changed after a multi-plane bug: the agent first assumed `plane0` alone, then patched the loader after discovering that `iscell` indexes pooled ROIs while multi-plane sessions store separate `plane0` and `plane1` arrays. The notes frame this as preserving the released `events` representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The neural signal is filtered only by keeping ROIs with `iscell[:,0] == 1` and by excluding non-finite neural frames from each trial through `valid_neural_frames`. There is no additional putative-interneuron exclusion or speed-correlation exclusion.

ii. 
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
...
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```

iii. In the notes, the agent explicitly states a key decision to use `iscell[:,0] == 1` only because that was the explicit mask available in NWB. The same notes also acknowledge that the paper mentions an additional speed-correlation interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start. Each trial uses the `trial_start` sample as time zero, and neural frames are taken from that start up to but excluding the paired teleport sample.

ii. 
```python
trial_bounds = find_complete_trial_bounds(trial_start, teleport)
...
trial_slice = slice(start, stop)
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes repeatedly describe trial-start alignment as the core alignment choice because the task instructions explicitly required temporal alignment to start of trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the original frame-aligned sampling grid, about 64.5 ms per sample. The script records the median timestamp spacing in milliseconds and does not rebin or resample the traces.

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

iii. The notes say behavior timestamps were treated as the reliable shared time base, especially for multi-plane sessions where the NWB `rate` attribute could be misleading.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior `position/timestamps` array together with trial boundaries from `trial_start` and `teleport`.

ii. 
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
trial_start = behavior_group["trial_start/data"][()].astype(np.float32)
teleport = behavior_group["teleport/data"][()].astype(np.float32)
```

iii. The notes say the agent chose actual timestamps rather than an assumed fixed frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Within each kept trial, the script subtracts the timestamp at the trial start frame from every kept frame timestamp.

ii. 
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
...
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        ...
    ]
)
```

iii. The notes describe this as a direct frame-aligned continuous input matching the reference time base.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is built from the same `trial_slice` and the same `frame_mask` used for `neural_trial`, so both have identical timepoints after filtering.

ii. 
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes list a specific sanity check for `time_from_trial_start_sec` versus raw NWB after the exact same slicing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. 
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
...
env_bin = env_to_binary(environment[start:stop])
```

iii. The notes map this directly to the reference per-trial `morph` variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script removes non-finite and negative values, takes the median environment value over the trial, thresholds it at 0.5 to get a binary value, and repeats that value across all timepoints in the trial.

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

iii. The notes justify this as reproducing the reference per-trial environment identity while keeping a uniform `(d, T)` input representation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavior series `processing/behavior/BehavioralTimeSeries/trial number/data`.

ii. 
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
...
trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The notes discuss either raw `trial number` or segmentation order, but the implemented code uses the raw per-frame trial-number stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, the code rounds the trial-number samples to integers, drops invalid and negative values, takes the modal value, and repeats that scalar across all timepoints in the trial.

ii. 
```python
def modal_trial_number(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    ...
    counts = np.bincount(values)
    return int(np.argmax(counts))
```

```python
np.full(time_trial.shape, trial_num, dtype=np.float32),
```

iii. The notes say the intent was to match the reference notion of per-trial trial IDs, but the actual code relies on the raw `trial number` stream rather than the loop index used in `get_timeseries_data`.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from the previous trial's reward outcome, which is computed from `Reward/timestamps`, the current trial's `reward_zone` segment, the trial bounds, and the trial-number mapping.

ii. 
```python
reward_outcome = reward_outcome_for_trial(
    reward_timestamps=reward_timestamps,
    t_start=float(timestamps[start]),
    t_stop=float(timestamps[stop]),
    reward_zone_segment=reward_zone[start:stop],
)
reward_by_trial_number[trial_num] = reward_outcome
```

```python
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. The notes justify this as deriving the binary previous-trial outcome from the same logic used for the current trial's reward outcome, with a sentinel `0` for the first within-session trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first computes and stores a reward-outcome label for every complete trial in `reward_by_trial_number`, then for each kept trial looks up the prior numeric trial's value, defaulting to `0` if none exists. That scalar is repeated across the trial's timepoints.

ii. 
```python
reward_by_trial_number: dict[int, int] = {}
...
reward_by_trial_number[trial_num] = reward_outcome
```

```python
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
...
np.full(time_trial.shape, prev_outcome, dtype=np.float32),
```

iii. In the notes, the agent explicitly chose `0` for the first trial as the least-assumptive binary placeholder.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the framewise `position` signal and from a per-trial reward-zone start/end coordinate chosen from the session scene metadata.

ii. 
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. The notes say the agent deliberately avoided inferring full zone extents from the sparse `reward_zone` time series and instead used paper/code reward-zone semantics from the session condition.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each kept frame, the script computes signed nearest distance to the active reward zone: values before the zone are `position - zone_start`, values inside the zone are `0`, and values after the zone are `position - zone_end`.

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

iii. The notes explicitly describe this as "signed nearest distance to zone" and tie it to the paper's reward-zone coordinate system.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into 7 categories using the task-specified bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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

iii. The notes reproduce the same binning and present it as an instruction-driven adaptation of the reward-relative position signal.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position_trial`, which is extracted with the same `trial_slice` and `frame_mask` as the neural data, so each distance label corresponds one-to-one with a neural frame.

ii. 
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
...
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes include an explicit raw-to-converted sanity check for this output after exact same slicing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. 
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
discretize_absolute_position(position_trial)
```

iii. The notes map this directly from the raw position stream to a task-required discretized output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the track interval `[0, 450)` and then divided into equal-width bins by `floor(position / 90)`.

ii. 
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. The notes describe this as using the 450 cm corridor length from the paper and converting it to 5 equal bins because the task required categorical outputs.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 equal 90 cm bins labeled `0` through `4`.

ii. 
```python
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. The notes say this binning came from the task specification rather than directly from the paper's original decoder.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses `position_trial` from the same kept frame indices as the neural data, so it stays framewise aligned with `neural_trial`.

ii. 
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
...
discretize_absolute_position(position_trial)
```

iii. The notes list this among the raw-to-converted output spot checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. 
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
...
lick_trial = lick[trial_slice][frame_mask]
```

iii. The notes say the output came from the same lick signal used in the reference `get_timeseries_data` path.

## 9-b. What processing is involved in computing `output` *Lick*?

i. First, whole trials can be removed by the lick-artifact rule. For surviving trials, lick samples are binarized as `(lick_trial > 0)`, producing a time-varying 0/1 output with no smoothing.

ii. 
```python
def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
```

```python
(lick_trial > 0).astype(np.int16),
```

iii. The notes justify the trial-level 35% artifact rule from released code and the binarization because the requested decoder output was categorical lick/no-lick.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is pulled with the same `trial_slice` and `frame_mask` as neural data, so each retained lick sample matches one neural timepoint.

ii. 
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say the output sanity checks compared these frame-aligned labels directly back to raw NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session `identifier` string, which encodes the scene and reward-zone condition, plus the trial number used to decide pre- versus post-switch state.

ii. 
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
```

```python
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. The notes explicitly say the agent used scene metadata because the `reward_zone` time series marks entries/events, not the zone identity A/B/C across the whole trial.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses scene names with regular expressions, extracts the pre-switch and optional post-switch zone letters, applies a hard-coded switch at trial 30, maps zone labels `A/B/C` to `0/1/2`, and repeats that code across the trial.

ii. 
```python
SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")
...
def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        ...
        return scene_info.after_zone
    return scene_info.before_zone
```

```python
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
```

iii. The notes justify this as matching `behavior.get_reward_zones`, while also documenting the assumption that switch trials default to 30 in the absence of an NWB `change_reward_trial` field.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, trial boundary timestamps, and the in-trial `reward_zone` segment used as an additional gate.

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
```

iii. The notes say this was meant to reproduce the reference per-trial reward logic from behavior within each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The helper finds whether any reward timestamp falls inside the trial window and whether the trial contains any positive `reward_zone` sample; if both are true, the trial is labeled rewarded. That binary value is then repeated across the trial's timepoints.

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

```python
np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16),
```

iii. The notes justify this as a per-trial categorical output adapted from the reference reward variables, and later note that the paper's GLM used a different time-varying rewarded regressor.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles imperfect data by ignoring negative/sentinel values in `environment` and `trial number`, treating `position <= -100` as invalid, masking out non-finite behavior or neural frames, dropping too-short trials, defaulting missing previous-trial outcome to `0`, and implicitly avoiding one-frame trailing mismatches because all trial slicing is driven by behavior bounds. It does not impute missing samples; it removes frames or entire trials.

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
values = values[np.isfinite(values)]
values = values[values >= 0]
```

```python
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
...
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. The notes and trajectory explicitly mention clipped first-trial markers, invalid sentinel values before synchronization, and a benign one-frame mismatch in one plane as edge cases the agent decided to handle conservatively by slicing on behavior bounds and dropping insufficient data.

## 13-a. What are the most time-consuming steps of the code?

i. The main expensive steps are session-by-session HDF5 reads, reconstructing the full accepted-cell deconvolved matrix across planes, iterating through all trials twice per session, and materializing trial-level `neural`, `input`, and `output` arrays. Optional plotting is also nontrivial but only for up to two sessions.

ii. 
```python
for plane in planes:
    ...
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    ...
    deconv[:, dest_cols] = plane_data
```

```python
for start, stop in trial_bounds:
    ...

for meta in trial_meta:
    ...
    neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
    ...
    output_trial = np.vstack([...])
```

iii. In Step 6 of the notes, the agent explicitly called out full-session deconvolved loading as a remaining inefficiency and estimated full conversion time from these operations.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops are the clearest vectorization candidates: the first pass over `trial_bounds` that computes metadata, the second pass over `trial_meta` that slices and stacks arrays, and the plane loop that reconstructs the pooled deconvolved matrix. All three are pure Python loops over large collections.

ii. 
```python
for start, stop in trial_bounds:
    ...
    trial_meta.append(...)
```

```python
for meta in trial_meta:
    ...
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

```python
for plane in planes:
    ...
    deconv[:, dest_cols] = plane_data
```

iii. The notes mention sequential session processing and list deconvolved reconstruction plus metadata parsing as places where the code stayed simple rather than aggressively optimized.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial-level indexing work in two separate passes: one to compute metadata and reward maps, and another to build the actual data arrays. It also repeatedly constructs full-length constant arrays with `np.full(...)` for per-trial labels and repeatedly recomputes masks and reward-time subsets for example trials.

ii. 
```python
for start, stop in trial_bounds:
    ...
    reward_by_trial_number[trial_num] = reward_outcome
    trial_meta.append(...)
```

```python
for meta in trial_meta:
    ...
    np.full(time_trial.shape, meta["environment"], dtype=np.float32)
    np.full(time_trial.shape, trial_num, dtype=np.float32)
    np.full(time_trial.shape, prev_outcome, dtype=np.float32)
    np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16)
    np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16)
```

iii. The notes explain that repeating per-trial labels across timepoints was an intentional representation choice for decoder compatibility, even though it duplicates information.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script always prepares `kept_examples` for up to three trials per session, including reward-time subsets and copies of position/speed/lick/output arrays, even when `--show-processing` is not used. It also carries session `summary` metadata and example payloads through processing even though only the final dataset fields are written to the pickle. Those extras are useful for diagnostics but not for downstream decoder training.

ii. 
```python
kept_examples: list[dict] = []
...
if len(kept_examples) < 3:
    trial_reward_times = reward_timestamps[
        (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
    ] - timestamps[start]
    kept_examples.append(
        {
            "trial_number": trial_num,
            "environment": meta["environment"],
            ...
            "output": output_trial,
            "reward_times": trial_reward_times,
        }
    )
```

```python
return (
    {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": brain_region_idx,
        "subject": subject,
        "summary": session_summary,
        "examples": kept_examples,
    },
    kept_examples,
)
```

iii. The notes say diagnostic plots and session summaries were kept for sanity checks and documentation, not because the converted dataset needed them for training.
