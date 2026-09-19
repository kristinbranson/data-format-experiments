# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file under `data/sub-*/sub-*_behavior+ophys.nwb`, sorts the paths, and processes them one session at a time. Each session is opened directly with `h5py`, not `pynwb`.

ii.
```python
def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
...
for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
```

```python
def process_session(path: Path) -> tuple[dict, list[dict]]:
    session_id = path.stem.replace("_behavior+ophys", "")
    with h5py.File(path, "r") as f:
        ...
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as using all released NWB files in the cohort and reading them directly with `h5py` “for speed.”

## 1-b. How are the data split into subjects?

i. Subjects are split by the `sub-<mouse>` parent directory of each NWB file. The final subject list is the sorted set of subjects observed in processed sessions.

ii.
```python
subject = path.parent.name.replace("sub-", "")
...
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes state that the release is organized as `data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, so the folder name is treated as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is taken from the file stem.

ii.
```python
def process_session(path: Path) -> tuple[dict, list[dict]]:
    session_id = path.stem.replace("_behavior+ophys", "")
    ...
    session_name = path.stem
```

iii. The notes repeatedly describe the release as “one NWB file per subject-session,” so the agent treated file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start > 0` to the next later sample with `teleport > 0`. Only complete start-stop pairs are kept.

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

iii. The notes justify this as matching the reference code’s `trial_start_inds` / `teleport_inds` logic and as avoiding fabricated partial trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials if they are shorter than 5 frames, if the lick signal looks corrupted, or if fewer than 5 frames remain after masking invalid behavior/neural samples.

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
for start, stop in trial_bounds:
    if stop - start < MIN_TRIAL_FRAMES:
        dropped_missing += 1
        continue
...
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

iii. The notes justify dropping lick-artifact trials with the paper-code-style `>35%` rule and dropping badly missing trials because the validator should not see NaNs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from the NWB `Deconvolved` traces, plus `iscell` and `planeIdx` metadata used to filter and reorder ROIs. It does not derive `neural` from `Fluorescence` and `Neuropil`.

ii.
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
...
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. The notes explicitly say: “Use NWB deconvolved traces directly” because the agent considered them the released equivalent of the paper’s `sess.timeseries['events']`.

## 2-b. How is the `neural` data processed?

i. The agent filters ROIs with `iscell[:,0] == 1`, reconstructs a pooled deconvolved matrix across planes, keeps the original framewise values, and transposes each trial to neuron-by-time. It does not recompute dF/F or deconvolution.

ii.
```python
for plane in planes:
    plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
    accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
    accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    ...
    dest_cols = np.searchsorted(accepted_idx, accepted_total_idx)
    deconv[:, dest_cols] = plane_data
...
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The main justification in the notes is that recomputing dF/F would create unnecessary divergence from the released data product, while the task only needs the deconvolved event-like signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural filtering consists of keeping only ROIs marked as cells by `iscell[:,0] == 1`, then excluding any frames where any accepted neural trace is non-finite. There is no explicit putative-interneuron exclusion.

ii.
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
...
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
...
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
```

iii. The notes justify using only the shared curated-cell mask available in NWB and explicitly say to “filter neurons with `iscell[:,0] == 1` only.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial from the `trial_start` sample to the corresponding `teleport` sample and treating the first sample of that slice as time zero for the trial.

ii.
```python
trial_bounds = find_complete_trial_bounds(trial_start, teleport)
...
trial_slice = slice(start, stop)
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say this matches the target requirement to align to `trial_start`; no extra event-specific shift is applied.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native frame-aligned sampling grid and does not rebin. It estimates the bin size from the median difference of behavior timestamps and stores that value in milliseconds in metadata.

ii.
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
...
median_bin_ms = float(
    np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
)
...
"metadata": {
    ...
    "time_bin_size": median_bin_ms,
```

iii. The notes justify using timestamps rather than the ophys `rate` attribute because multi-plane sessions can report `31.015625` even though the aligned sample grid is still about `15.5 Hz`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps associated with `position`.

ii.
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. The notes justify using behavior timestamps as the reliable aligned time base for the converted trials.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The agent subtracts the timestamp at the first frame of the trial from the timestamps of all kept frames in that trial.

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

iii. No separate justification was written beyond the mapping table; the implementation itself shows simple subtraction to express time from trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same trial slice and the same per-frame validity mask are applied to timestamps and neural data, so both streams keep the same frames.

ii.
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes describe this as preserving the shared frame-aligned time base already present in the NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
...
env_bin = env_to_binary(environment[start:stop])
```

iii. The mapping table in the notes names the `environment` series as the source for this input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the agent removes invalid or negative values, takes the median remaining value, thresholds it at `0.5` to get a binary trial label, and repeats that label across all timepoints in the trial.

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
np.full(time_trial.shape, meta["environment"], dtype=np.float32)
```

iii. The notes justify this as making environment a per-trial binary input consistent with the requested decoder format.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the stored `trial number` behavior series by taking the modal nonnegative value within each segmented trial.

ii.
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
...
trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The notes say the converter uses the “actual released-data” trial numbering and mention that one session has a clipped first start marker, so it keeps complete trial epochs and uses the stored trial numbers inside them.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent drops invalid values, rounds to integers, restricts to nonnegative values, takes the mode with `np.bincount`, and repeats the resulting trial number across all timepoints in the trial.

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
...
np.full(time_trial.shape, trial_num, dtype=np.float32)
```

iii. No detailed separate defense appears in the notes beyond using the stored released-data trial labels inside complete trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The previous-trial-outcome input is derived indirectly from the current session’s per-trial reward outcomes, which themselves are computed from `Reward/timestamps`, behavior `timestamps`, and the `reward_zone` series.

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
reward_by_trial_number[trial_num] = reward_outcome
```

iii. The notes say this follows `behavior.get_trial_types` semantics: a trial counts as rewarded when reward delivery occurs within the trial and the reward-zone signal indicates zone entry.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the agent looks up the previous trial number in a dictionary of trial outcomes; if there is no previous entry it uses `0`. The resulting scalar is repeated across all timepoints in the current trial.

ii.
```python
reward_by_trial_number: dict[int, int] = {}
...
reward_by_trial_number[trial_num] = reward_outcome
...
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
...
np.full(time_trial.shape, prev_outcome, dtype=np.float32)
```

iii. The notes explicitly justify setting the first trial’s previous outcome to `0` as the least assumption-laden binary sentinel.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from position samples and from reward-zone coordinates inferred from the session scene string in the NWB `identifier`. The agent does not infer the zone identity from the `reward_zone` behavior series.

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
position = behavior_group["position/data"][()].astype(np.float32)
...
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
...
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes explicitly justify using scene metadata because the `reward_zone` series marks sparse zone-entry events rather than the full zone extent.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each kept position sample, the agent computes signed distance to the nearest edge of the trial’s reward zone: negative before the zone, zero inside, positive after the zone.

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

iii. The notes tie this to the paper’s reward-relative-position semantics and to `behavior.get_reward_zones`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distances are discretized into 7 categories with explicit threshold comparisons at `-50`, `-10`, `0`, `10`, and `50` cm.

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

iii. The notes say these bins were chosen to match the task instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from `position_trial`, where `position_trial` uses the same trial slice and frame mask as `neural_trial`.

ii.
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
...
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes justify this by treating NWB behavior and neural arrays as already frame-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` series.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
position_trial = position[trial_slice][frame_mask]
```

iii. The mapping table in the notes directly maps `position` to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent masks out invalid frames with the shared frame mask, clips remaining positions into `[0, 450)`, divides by `90`, floors to integers, and caps the final bin index at `4`.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. The notes justify 5 equal bins across a 450 cm track and describe the representation as a time-varying categorical output.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholding is implicit in `floor(position / 90)`, producing 5 bins corresponding to `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm.

ii.
```python
clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. The notes justify this as the requested 5 equal-sized bins over the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position uses the same trial slice and validity mask as the neural data.

ii.
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes say the NWB release already preserves a shared frame-aligned time base between behavior and neural activity.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` series.

ii.
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
...
lick_trial = lick[trial_slice][frame_mask]
```

iii. The mapping table in the notes identifies `lick` as the source variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent binarizes the kept lick samples with `lick > 0`. Separately, entire trials may already have been discarded by the lick-sensor error rule.

ii.
```python
def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
...
(lick_trial > 0).astype(np.int16)
```

iii. The notes justify binary lick output and justify dropping lick-artifact trials because the validator should not receive NaNs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick output is computed from the same `trial_slice` and `frame_mask` used for neural data.

ii.
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes describe all behavioral outputs as frame-aligned to the neural sampling grid already stored in NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string, parsed into environment/zone transition metadata, plus the trial number used to decide pre-switch versus post-switch trials.

ii.
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
trial_num = modal_trial_number(trial_number[start:stop])
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. The notes explicitly justify using “paper/code reward-zone semantics rather than inferring from sparse `reward_zone` events alone.”

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regexes, switch sessions are split at trial `30`, the resulting zone label is mapped with `ZONE_TO_CODE`, and that code is repeated across timepoints in the trial.

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
...
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16)
```

iii. The notes justify the trial-30 switch rule from the paper/code and the use of scene parsing as the direct way to recover A/B/C labels.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, trial start/stop timestamps, and the trial’s `reward_zone` samples.

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

iii. The notes justify this as following the paper code’s `get_trial_types` semantics for rewarded versus omission trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if at least one reward timestamp falls between the trial’s start and stop times and the `reward_zone` series is positive somewhere in that trial. The scalar label is repeated across all timepoints in the trial.

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
...
np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16)
```

iii. The notes explicitly tie this rule to `behavior.get_trial_types` rather than to a simpler “any reward event in trial” criterion.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles minor data problems by keeping only complete trials, masking out invalid behavior and neural frames, dropping trials with too few valid frames, and dropping trials with apparent lick-sensor failures. It does not interpolate or repair missing values.

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
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue
```

iii. The notes justify this mainly in terms of the decoder validator: the converted dataset should not contain NaNs, so invalid frames and corrupted lick trials are removed rather than filled in.

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are session-level HDF5 reads of large neural matrices, multi-plane deconvolved-trace reconstruction, the per-trial slicing/building loop, and final pickling of the full dataset.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
...
for meta in trial_meta:
    ...
    neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
    ...
with args.outpicklefile.open("wb") as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly mention HDF5 speed, loading full-session deconvolved matrices one session at a time, and runtime estimates dominated by per-session processing plus pickle writing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate loops are the loop over planes when reconstructing pooled `deconv`, the first loop over trials to build `trial_meta` and reward dictionaries, and the second loop over trials that repeatedly slices arrays and stacks outputs.

ii.
```python
for plane in planes:
    ...
for start, stop in trial_bounds:
    ...
for meta in trial_meta:
    ...
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The notes already flag some remaining inefficiency, saying that full-session matrices are still loaded and processed session by session.

## 13-c. What processing does the code repeat multiple times?

i. The code makes two passes over trials in each session: one to compute per-trial metadata and outcomes, and another to actually slice neural/behavior data and build arrays. It also computes plot/example payloads for a few kept trials inside the same second pass.

ii.
```python
trial_meta: list[dict] = []
reward_by_trial_number: dict[int, int] = {}
...
for start, stop in trial_bounds:
    ...
    reward_by_trial_number[trial_num] = reward_outcome
    trial_meta.append({...})
...
for meta in trial_meta:
    ...
    neural_trials.append(neural_trial)
    ...
    if len(kept_examples) < 3:
        ...
        kept_examples.append({...})
```

iii. The notes do not call this out directly, but they do emphasize that the converter avoided a separate full-dataset survey pass and instead works session by session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent always builds `kept_examples` dictionaries for up to three trials per session, including reward-time lists and raw traces, even though these are only used for optional plotting and are not part of the saved dataset. It also stores verbose per-session summary metadata that the downstream decoder does not need.

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
            "zone_label": meta["zone_label"],
            ...
            "reward_times": trial_reward_times,
        }
    )
...
"session_info": [sess["summary"] for sess in processed_sessions],
```

iii. The notes justify these as diagnostics and validation support, not as part of the decoder-facing representation.
