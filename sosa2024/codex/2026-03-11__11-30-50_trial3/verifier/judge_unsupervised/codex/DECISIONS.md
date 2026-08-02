# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all session files by globbing `data/sub-*/sub-*_behavior+ophys.nwb`, sorting the paths, then opening each NWB file with `h5py`. Within each file it reads behavioral arrays from `processing/behavior/BehavioralTimeSeries` and neural arrays from `processing/ophys`.

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

iii. In `CONVERSION_NOTES.md`, the agent justified direct NWB loading as using the released representation of the processed dataset, saying the NWB files should be treated as equivalent to the reference `sess` content and that `h5py` was chosen for speed.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file, with the `sub-` prefix removed. The dataset-level `subjects` list is the sorted unique set of these IDs, and `subject_idx` maps each processed session to its subject.

ii. 
```python
subject = path.parent.name.replace("sub-", "")

subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes say sessions are sorted by file path for stable ordering, and subject IDs are intended to match mouse IDs from the released file layout.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The top-level dataset stores one entry per processed NWB file in `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx`.

ii. 
```python
for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
    ...
    processed_sessions.append(session)

"neural": [sess["neural"] for sess in processed_sessions],
"input": [sess["input"] for sess in processed_sessions],
"output": [sess["output"] for sess in processed_sessions],
```

iii. The agent documented that the released cohort has one file per subject-session and chose file-path order for reproducibility.

## 1-d. How are the data split into trials?

i. Trials are defined from the first `trial_start > 0` sample after the previous teleport to the next `teleport > 0` sample, using complete start/stop pairs only. Within each pair, the trial is later sliced as `start:stop`, so trial start is included and teleport is excluded.

ii. 
```python
def find_complete_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    ...
    if stop > start:
        bounds.append((int(start), int(stop)))

trial_bounds = find_complete_trial_bounds(trial_start, teleport)
trial_slice = slice(start, stop)
```

iii. The notes say this was chosen to match the reference code’s `trial_start_inds`/`teleport_inds` semantics and to avoid fabricating incomplete first or last trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they are shorter than 5 frames, if the lick trace shows a sensor-error pattern (`>35%` of trial frames have lick values `>2`), or if after frame-level masking there are fewer than 5 valid frames. Sessions with fewer than 2 valid trials are skipped entirely.

ii. 
```python
LICK_ERROR_FRACTION = 0.35
MIN_TRIAL_FRAMES = 5

if stop - start < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue

"lick_error": has_lick_sensor_error(lick[start:stop]),

frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
    dropped_missing += 1
    continue

if len(session["neural"]) < 2:
    ... continue
```

iii. The agent explicitly chose the code-style `35%` lick threshold over the paper text’s `30%`, citing the released code path. It also justified complete-trial filtering and minimum-length checks as avoiding malformed examples that the validator would reject.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from `processing/ophys/Deconvolved/plane*/data`, restricted to ROIs whose `ImageSegmentation/PlaneSegmentation/iscell[:,0] == 1`. Plane assignments come from `planeIdx`.

ii. 
```python
iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
accepted_mask = np.asarray(iscell[:, 0]) == 1
plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
...
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. The notes say the agent treated NWB `Deconvolved` as the released equivalent of the paper’s deconvolved `events`, and used `iscell` as the explicit curated-cell mask present in the data.

## 2-b. How is the `neural` data processed?

i. The agent reconstructs a pooled accepted-cell matrix across planes, preserves the native frame sampling, slices it per trial, applies the same per-frame validity mask used for behavior, and transposes each trial to neuron-by-time format. It does not recompute dF/F, smoothing, or deconvolution.

ii. 
```python
for plane in planes:
    ...
    plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
    ...
    deconv[:, dest_cols] = plane_data

valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The agent’s notes state that recomputing dF/F and deconvolution would add unnecessary divergence because the release already provides frame-aligned deconvolved traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level QC is the Suite2p `iscell` mask. At the frame level, samples are removed when any neuron is non-finite. No additional putative-interneuron or speed-correlation exclusion is applied.

ii. 
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
accepted_idx = np.flatnonzero(accepted_mask)
...
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```

iii. The notes describe `iscell` as the “primary shared-data curation” available in NWB. The agent acknowledged the paper’s extra speed-correlation exclusion, but did not implement it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from `trial_start` to the following `teleport` and by using the same frame indices as the trial-relative time input. Metadata explicitly records `temporal_alignment_event = "trial_start"` and `off_start = 0.0`.

ii. 
```python
trial_slice = slice(start, stop)
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
...
"temporal_alignment_event": "trial_start",
"off_start": 0.0,
```

iii. The notes repeatedly state that the conversion should be “trial-aligned” and that trial start is the reliable event shared by neural and behavioral streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the original frame grid, with session bin size estimated as the median difference between behavioral timestamps, about 64.5 ms. No temporal rebinning is applied.

ii. 
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
...
median_bin_ms = float(
    np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
)
```

iii. The notes say the aligned sample grid should come from behavior timestamps rather than trusting NWB rate attributes, especially in multi-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii. 
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
...
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
```

iii. The agent justified using behavior timestamps as the authoritative aligned time base shared with the neural data.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each kept frame in a trial, the absolute timestamp is shifted so that the `trial_start` frame equals zero. The resulting vector is cast to `float32`.

ii. 
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
...
time_trial.astype(np.float32)
```

iii. The notes describe this as a direct frame-aligned representation of “time from trial start” with no interpolation or resampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same `trial_slice` and the same `frame_mask` as the neural trial, so every neural frame has a matching trial-relative time sample.

ii. 
```python
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The agent’s stated goal was to preserve the shared frame-aligned sampling already present in the released data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. 
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
...
env_bin = env_to_binary(environment[start:stop])
```

iii. The notes map this variable to the reference code’s binary `morph`/environment identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the agent removes non-finite and negative values, takes the median remaining value, thresholds it at `0.5`, converts it to `0` or `1`, and repeats that value across all timepoints in the trial.

ii. 
```python
def env_to_binary(values: np.ndarray) -> int:
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    return int(np.round(np.median(values)) > 0.5)

np.full(time_trial.shape, meta["environment"], dtype=np.float32)
```

iii. The notes say the per-trial modal/median value was used to avoid invalid pre-sync sentinel samples and to produce a stable per-trial label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/trial number/data`.

ii. 
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
...
trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The notes mention trial IDs in the reference code and describe the NWB `trial number` series as the direct source.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent removes invalid and negative values, rounds to integers, takes the most frequent value within the trial, and repeats that scalar across all frames of the trial.

ii. 
```python
def modal_trial_number(values: np.ndarray) -> int:
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    counts = np.bincount(values)
    return int(np.argmax(counts))

np.full(time_trial.shape, trial_num, dtype=np.float32)
```

iii. The notes frame this as using the per-trial integer identity already embedded in the aligned behavior stream.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from the previous trial’s computed `reward_outcome`, which itself is based on `Reward/timestamps` and the `reward_zone/data` segment for each trial.

ii. 
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
...
reward_by_trial_number[trial_num] = reward_outcome
...
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. The agent justified this as matching the requested “previous trial outcome” input and using within-session trial history, with `0` for the first trial as the least-assumptive sentinel.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes a reward label for each complete trial, stores it in a dictionary keyed by trial number, then looks up `trial_num - 1` when building the current trial. Missing predecessors default to `0`, and the result is repeated across the trial.

ii. 
```python
reward_by_trial_number[trial_num] = reward_outcome
...
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
...
np.full(time_trial.shape, prev_outcome, dtype=np.float32)
```

iii. The notes explicitly call out this first-trial default and say they preferred a binary sentinel over `NaN` because the validator forbids missing values.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` together with reward-zone coordinates inferred from the session scene string in `identifier`.

ii. 
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
position = behavior_group["position/data"][()].astype(np.float32)
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. The notes say the sparse `reward_zone` timeseries only marks zone-entry events, so full zone extent had to come from the paper/code reward-zone semantics and scene metadata.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest edge of the current reward zone: negative before entering the zone, zero inside it, positive after leaving it.

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

iii. The notes describe this as matching the intended reward-relative position semantics from the paper while adapting it to the requested categorical decoder output.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into 7 bins exactly following the task specification: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `> 50`.

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

iii. The notes explicitly list these bins as the chosen mapping from continuous signed distance to the required categorical output.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from the same trial-sliced, frame-masked position samples that define the neural trial, so it is frame-wise aligned with neural activity.

ii. 
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. The notes emphasize preserving the shared frame grid between behavior and neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. 
```python
position = behavior_group["position/data"][()].astype(np.float32)
...
discretize_absolute_position(position_trial)
```

iii. The notes describe absolute position as a direct behavioral stream from the aligned VR data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips positions to `[0, 450)` cm, divides by 90 cm, floors to integers, and caps the result at bin index 4.

ii. 
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```

iii. The notes state that the 450 cm track was discretized into 5 equal-width bins to match the task instructions.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 equal-sized corridor bins of width 90 cm.

ii. 
```python
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. The agent’s mapping plan in the notes explicitly says “discretize absolute track position on `[0,450]` into 5 equal bins.”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position uses the same frame-selected `position_trial` array that is indexed by the same trial bounds and frame mask as `neural_trial`.

ii. 
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
output_trial = np.vstack([
    ...,
    discretize_absolute_position(position_trial),
    ...
])
```

iii. The alignment rationale is the same shared frame-aligned time base used throughout the script.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. 
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
...
lick_trial = lick[trial_slice][frame_mask]
```

iii. The notes identify the NWB lick series as the reference-aligned licking variable and use it both for output construction and for lick-sensor QC.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first removes trials classified as lick-sensor failures, then binarizes the per-frame lick values with `lick_trial > 0` and stores the result as `int16`.

ii. 
```python
if meta["lick_error"]:
    dropped_lick += 1
    continue
...
(lick_trial > 0).astype(np.int16)
```

iii. The notes justify dropping failed lick trials entirely because the validator disallows missing values, and describe the lick output as a binary time-varying series.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is indexed with the same trial bounds and frame mask as neural activity, so it is one value per retained neural frame.

ii. 
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The notes state that all behavioral and neural time series should remain on the common frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session `identifier` string, parsed into environment and zone labels, plus the per-trial switch rule at trial 30.

ii. 
```python
identifier = decode_bytes(f["identifier"][()])
scene = identifier.split("/")[-1]
scene_info = parse_scene(scene)
...
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. The notes say this mirrors the reference code’s reward-zone lookup and avoids treating sparse `reward_zone` events as the zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses several scene-name formats with regexes, chooses the pre-switch zone for trials before 30 and the post-switch zone from trial 30 onward, maps `A/B/C` to `0/1/2`, and repeats the code across the trial.

ii. 
```python
SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")

def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        return scene_info.after_zone
    return scene_info.before_zone

np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16)
```

iii. The notes explicitly cite the paper’s “switch after 30 trials” rule and the need to infer A/B/C from scene metadata.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` together with the trial’s `reward_zone/data` segment.

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

iii. The notes say reward should be determined from reward deliveries within the trial and mention using the reward-zone signal to guard against spurious timestamp-only detections.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code finds reward timestamps between trial start and trial stop, checks whether the trial contains any `reward_zone > 0` sample, returns `1` only if both conditions are true, and repeats that scalar across the trial.

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

iii. The notes frame this as matching per-trial reward outcome while keeping labels categorical and time-aligned for the decoder format.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles minor data issues mostly by dropping bad data rather than imputing. It removes invalid behavior frames using finite-value checks and a position sentinel cutoff, removes frames with any non-finite neural value, drops incomplete or very short trials, drops lick-sensor-error trials, defaults missing previous-trial outcome to `0`, and raises exceptions for unsupported scene names or trials with no valid environment/trial-number samples.

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
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. The notes repeatedly say the validator disallows `NaN`s and that complete-trial dropping was preferred to “fabricating” values for malformed segments.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are opening every NWB file, reading and reconstructing the accepted-cell deconvolved matrix across planes, iterating over all trials twice per session, and writing the final multi-gigabyte pickle. Diagnostic plotting is extra overhead when enabled.

ii. 
```python
with h5py.File(path, "r") as f:
    ...
    for plane in planes:
        ...
        deconv[:, dest_cols] = plane_data
...
for start, stop in trial_bounds:
    ...
for meta in trial_meta:
    ...
pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly call out full-session deconvolved loading as a main inefficiency and report full conversion runtime as a few minutes for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-plane reconstruction loop, the first pass over trials that builds metadata, and the second pass that re-slices arrays and builds `input_trial`/`output_trial` could all be reduced or partially vectorized. The per-trial `np.full` allocations also repeat scalar expansion work.

ii. 
```python
for plane in planes:
    ...

for start, stop in trial_bounds:
    ...

for meta in trial_meta:
    ...
    input_trial = np.vstack([... np.full(...), ...])
    output_trial = np.vstack([... np.full(...), ...])
```

iii. In `CONVERSION_NOTES.md`, the agent itself noted that the implementation still loads full-session matrices and performs repeated per-trial allocations.

## 13-c. What processing does the code repeat multiple times?

i. The code makes two trial-level passes: one to compute trial metadata and reward lookup tables, and a second to actually build the trial arrays. It also repeatedly indexes the same raw arrays trial by trial and repeatedly expands per-trial constants into full-length vectors.

ii. 
```python
trial_meta: list[dict] = []
for start, stop in trial_bounds:
    ...
    trial_meta.append({...})

for meta in trial_meta:
    ...
    position_trial = position[trial_slice][frame_mask]
    speed_trial = speed[trial_slice][frame_mask]
    lick_trial = lick[trial_slice][frame_mask]
```

iii. The notes describe this as a straightforward but not maximally efficient implementation and mention avoiding intermediate files rather than fully optimizing the inner loops.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script constructs `kept_examples` for plotting, generates optional diagnostic plots, stores detailed `session_summary` metadata, and computes `trial_reward_times` for example plots only. These are useful for debugging but are not required for downstream decoder training on the saved arrays.

ii. 
```python
kept_examples: list[dict] = []
...
if len(kept_examples) < 3:
    trial_reward_times = reward_timestamps[...] - timestamps[start]
    kept_examples.append({...})

session_summary = {
    ...
    "trial_numbers_kept": [int(np.rint(x[0, 0])) for x in input_trials] if input_trials else [],
}

if args.show_processing and plots_made < 2:
    build_trial_plot(...)
```

iii. The notes explicitly separate these as sanity-check and visualization support, not as core converted data needed by the decoder.
