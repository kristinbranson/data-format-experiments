# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file under `data/sub-*/sub-*_behavior+ophys.nwb` with `h5py`, then processes each file session-by-session with `load_session()`. In `--sample` mode it truncates this list to the first two files.

ii.
```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files

def load_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        ...
```

iii. In `CONVERSION_NOTES.md`, the agent justified direct HDF5 access as a speed optimization and treated the NWB files as the complete exported dataset.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from each NWB file’s `general/subject/subject_id`, and `build_dataset()` creates a unique `subjects` list plus a per-session `subject_idx`.

ii.
```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
...
subject_idx.append(subject_to_idx[subject])
```

iii. The notes describe one folder per subject, but the implemented split uses the file metadata as the authoritative subject label.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is read from `general/session_id`, and one output session is appended per successfully converted file.

ii.
```python
session_id = decode_h5_scalar(handle["general/session_id"])
...
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(
        path=path,
        show_processing=show_processing and session_number < 2,
    )
    ...
    data["neural"].append(session_data["neural_trials"])
```

iii. The agent’s notes state that the dataset contains one NWB file per session, so file boundaries define session boundaries.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavioral signals: every `trial_start > 0.5` becomes a trial start, and the next `teleport > 0.5` becomes the trial end. Trial slices are half-open intervals `[start:stop)`.

ii.
```python
def reconstruct_trials(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0.5)
    teleports = np.flatnonzero(teleport > 0.5)
    trials = []
    tp_ptr = 0

    for start in starts:
        while tp_ptr < len(teleports) and teleports[tp_ptr] <= start:
            tp_ptr += 1
        if tp_ptr >= len(teleports):
            break
        stop = teleports[tp_ptr]
        if stop > start:
            trials.append((int(start), int(stop)))
        tp_ptr += 1
```

iii. In Step 5 of the notes, the agent justified this as matching the paper’s trial semantics: trial-aligned frames from trial start to teleport onset, excluding teleport frames.

## 1-e. How are trials filtered based on quality controls?

i. The agent skips degenerate trial windows, drops trials with fewer than 2 samples, removes “bad lick” trials when more than 35% of samples have `lick > 2`, and drops whole sessions if fewer than 2 usable trials remain.

ii.
```python
LICK_ERROR_FRAC = 0.35

def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC

for trial_idx, (start, stop) in enumerate(trials):
    if stop <= start:
        continue
    ...
    if pos_trial.size < 2:
        continue
    if is_bad_lick_trial(lick_trial):
        dropped_bad_lick += 1
        continue
...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The notes say this was done because lick is a required decoder output, NaN lick labels were undesirable, and the decoder needs at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are taken from the NWB `processing/ophys/Deconvolved` arrays, with ROIs filtered by `iscell`. For multi-plane sessions, plane-specific deconvolved matrices are reassembled using `planeIdx`.

ii.
```python
segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)

deconv_group = handle["processing/ophys/Deconvolved"]
plane_keys = sorted(deconv_group.keys(), key=lambda key: int(key.replace("plane", "")))
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    ...
    deconvolved = all_deconvolved[:, curated_idx]
```

iii. In the notes, the agent argued that NWB-exported deconvolved activity was the closest native match to the paper’s `events` signal and avoided recomputing dF/F.

## 2-b. How is the `neural` data processed?

i. The agent does not recompute dF/F or OASIS events. It reads stored deconvolved activity, optionally reconstructs multi-plane sessions into one ROI matrix, casts to `float16`, and then slices each trial and transposes it to `(neurons, time)`.

ii.
```python
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    n_frames = deconv_group[plane_keys[0]]["data"].shape[0]
    n_rois = plane_idx.shape[0]
    all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
    for plane_key in plane_keys:
        ...
        all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]
...
neural_trial = deconvolved[start:stop].T
```

iii. The notes justify this as a memory/runtime optimization and as using the closest available exported representation of the paper’s deconvolved events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is limited to Suite2p manual curation via `iscell[:, 0] > 0.5`. The agent does not implement the paper’s additional putative interneuron exclusion.

ii.
```python
iscell = segmentation["iscell"][:]
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconvolved = all_deconvolved[:, curated_idx]
```

iii. Step 5 of the notes explicitly says the initial neuron filter is `iscell[:,0] == 1`, while additional interneuron exclusions were left as a later consistency check rather than part of the final conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from the `trial_start` frame up to the `teleport` frame. No extra temporal shift is applied.

ii.
```python
trials = reconstruct_trials(trial_start, teleport)
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes say the conversion uses “trial-aligned frames” and that trials span `trial_start` to teleport onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed frame rate of 15.5078125 Hz, corresponding to `1000 / 15.5078125` ms per sample. No temporal rebinning or resampling is applied.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
"frame_rate_hz": FRAME_RATE_HZ,
```

iii. The notes say the agent intentionally kept the native imaging-frame resolution and only applied task-required categorical binning, not temporal rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`, together with the reconstructed trial start/end indices.

ii.
```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    time_trial = position_t[start:stop] - position_t[start]
```

iii. The agent treated the behavior timestamps as already synchronized to imaging frames, so elapsed time within the trial could be computed directly from them.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp is subtracted from all timestamps in that trial, and the result is stored as `float32`.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
...
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        ...
    ]
)
```

iii. The notes describe this as the required “time from trial start” input aligned to the chosen trial boundaries.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time is aligned to neural data by using the same `[start:stop)` frame indices for both the behavior timestamps and the neural matrix.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
...
neural_trial = deconvolved[start:stop].T
```

iii. The agent’s assumption, stated in the notes, is that the NWB behavior streams are already synchronized to the imaging frame grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The implemented environment label is derived from the NWB `identifier` string by `parse_scene()` and `zone_for_trial()`. The raw `environment` behavior stream is loaded but not used in the final input.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
environment = behavior["environment/data"][:].astype(np.float32)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
...
np.full(n_time, env_code, dtype=np.float32),
```

iii. In the notes, the agent justified this by saying that environment and reward-zone identity should come from the scene string plus the paper’s 30-trial switch rule.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses several scene-name formats, determines whether the session is a switch session, chooses the pre- or post-switch environment based on whether `trial_idx >= 30`, converts `Env1/Env2` to `0/1`, and repeats the value across the trial.

ii.
```python
def parse_scene(identifier: str) -> dict:
    ...
    match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    ...
    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    ...

def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```

iii. The notes explicitly cite the scene identifier and “switch-after-30” rule as the intended source of both environment and reward-zone identity.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the reconstructed within-session trial index `trial_idx` produced by iterating over the reconstructed trial list.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. The notes describe this as a reconstructed within-session trial index rather than using the raw `trial number` stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is applied beyond casting the trial index to float and repeating it across all time bins of the trial.

ii.
```python
input_trial = np.vstack(
    [
        ...,
        np.full(n_time, float(trial_idx), dtype=np.float32),
        ...
    ]
)
```

iii. The agent treated trial number as a per-trial constant represented in time-varying format.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward delivery timestamps (`Reward/timestamps`) and the reconstructed trial time windows defined on the behavioral timestamp grid.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The notes say the agent wanted to use actual delivered rewards within each trial, then shift that result back by one trial for the previous-outcome input.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes a per-trial binary reward outcome, then sets the current trial’s previous-outcome value to the previous trial’s binary reward label, with trial 0 forced to 0. The value is repeated across the trial.

ii.
```python
def reward_outcomes_from_timestamps(reward_times: np.ndarray, trial_times: np.ndarray, trials: list[tuple[int, int]]) -> np.ndarray:
    outcomes = np.zeros(len(trials), dtype=np.int8)
    reward_ptr = 0
    for trial_idx, (start, stop) in enumerate(trials):
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
            reward_ptr += 1
        outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The justification in the notes is that this matches the task specification `omitted = 0, rewarded = 1` for the previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from framewise `position` and the per-trial reward-zone boundaries inferred from the scene identifier plus switch logic. The raw `reward_zone` behavior series is not used.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
...
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. In the notes, the agent explicitly says the NWB `reward_zone` stream is not the A/B/C label and that active zone identity should be parsed from the session scene metadata instead.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest reward-zone boundary: negative before the zone, zero inside it, positive after it. It then maps the continuous distance to 7 categorical bins.

ii.
```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
    bins = np.full(distance.shape, 6, dtype=np.int16)
    bins[distance < -50.0] = 0
    bins[(distance >= -50.0) & (distance < -10.0)] = 1
    bins[(distance >= -10.0) & (distance < 0.0)] = 2
    bins[distance == 0.0] = 3
    bins[(distance > 0.0) & (distance <= 10.0)] = 4
    bins[(distance > 10.0) & (distance <= 50.0)] = 5
    return bins
```

iii. The notes say this follows the decoder task definition of distance relative to the active reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Thresholding is hard-coded with the instruction-specified boundaries: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
bins = np.full(distance.shape, 6, dtype=np.int16)
bins[distance < -50.0] = 0
bins[(distance >= -50.0) & (distance < -10.0)] = 1
bins[(distance >= -10.0) & (distance < 0.0)] = 2
bins[distance == 0.0] = 3
bins[(distance > 0.0) & (distance <= 10.0)] = 4
bins[(distance > 10.0) & (distance <= 50.0)] = 5
```

iii. The agent’s notes describe this as task-required output discretization rather than a paper-specific transformation.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by taking `position[start:stop]` over the same frame range used to slice `deconvolved[start:stop]`.

ii.
```python
pos_trial = position[start:stop]
...
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The justification is the same alignment assumption used throughout the converter: behavior and neural streams share the same imaging-frame index after trial reconstruction.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes treat this as direct track position in the 450 cm VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The position is clipped to the track interval `[0, 450)` and then discretized into five equal-width bins.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The notes say absolute position should use the 450 cm track interval only, with teleport frames excluded entirely.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code uses five equal bins defined by evenly spacing the 0–450 cm corridor into 90 cm segments.

ii.
```python
TRACK_START_CM = 0.0
TRACK_END_CM = 450.0
...
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. This follows the decoder task requirement for five equal-sized position bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position bins are computed from the same `[start:stop)` frame slice as the neural trial matrix.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes say all outputs are built on the same trial-aligned imaging frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived directly from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
lick_bin = binarize_licks(lick_trial)
```

iii. The notes refer to the raw lick stream as the source for the binary lick decoder target.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first filters out bad-lick trials, then binarizes lick values by rounding and clipping them into `{0, 1}`.

ii.
```python
def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC

def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. The notes justify this as matching the paper’s bad-lick QC while still producing a binary lick output for the decoder.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same trial slice `lick[start:stop]` used for neural and other behavioral outputs.

ii.
```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
lick_bin = binarize_licks(lick_trial)
```

iii. The converter assumes the lick stream is already sampled on the same frame grid as the neural data.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the session `identifier` string, interpreted with scene parsing and switch logic; the reward-zone label does not come from the raw `reward_zone` timeseries.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```

iii. The notes explicitly argue that the raw `reward_zone` stream is not the A/B/C identity, so the session scene metadata should determine reward-zone location.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses supported scene-name patterns, chooses the pre- or post-switch zone after trial 30, maps `A/B/C` to `0/1/2`, and repeats that categorical value across the whole trial.

ii.
```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
...
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
...
np.full(n_time, zone_code, dtype=np.int16),
```

iii. The notes say this mirrors the paper/code convention that switch sessions change condition after 30 trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps` together with trial start/stop times on the behavior timestamp grid.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
reward_code = int(reward_outcomes[trial_idx])
```

iii. The notes state that the output should reflect actual delivered reward rather than a planned trial type.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code marks the trial as rewarded if any reward timestamp falls between that trial’s start and stop times. The resulting binary label is repeated across the trial.

ii.
```python
def reward_outcomes_from_timestamps(reward_times: np.ndarray, trial_times: np.ndarray, trials: list[tuple[int, int]]) -> np.ndarray:
    outcomes = np.zeros(len(trials), dtype=np.int8)
    reward_ptr = 0
    for trial_idx, (start, stop) in enumerate(trials):
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
            reward_ptr += 1
        outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes
...
np.full(n_time, reward_code, dtype=np.int16),
```

iii. The agent justified this in the notes as using actual reward delivery within reconstructed trial intervals.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mainly handles errors by skipping invalid trials or failing fast. It skips empty/degenerate trials and bad-lick trials, skips sessions with too few remaining trials, and raises an error for unrecognized scene identifiers or inconsistent multi-plane ROI mapping. It does not implement special repair for neural/behavior length mismatches or missing reward-zone labels.

ii.
```python
raise ValueError(f"Unrecognized scene format: {scene}")
...
if plane_data.shape[1] != cols.size:
    raise ValueError(
        f"{path.name}: {plane_key} has {plane_data.shape[1]} ROIs but planeIdx maps {cols.size}"
    )
...
if stop <= start:
    continue
...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The notes frame this as keeping only clean trials for the decoder and treating structural metadata inconsistencies as hard failures.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session I/O and array materialization in `load_session()`, especially loading deconvolved matrices, reconstructing multi-plane ROI matrices, iterating over all trials to slice/build outputs, and finally writing the large pickle.

ii.
```python
with h5py.File(path, "r") as handle:
    ...
    deconv_group = handle["processing/ophys/Deconvolved"]
    ...
    for plane_key in plane_keys:
        ...
        all_deconvolved[:, cols] = plane_data
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
    ...
with out_path.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly identify full-session deconvolved loading as a memory/runtime bottleneck and add per-session timing output to track it.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest vectorization opportunities are the loops over trials in `reward_outcomes_from_timestamps()` and `load_session()`, plus the loop that reconstructs multi-plane deconvolved matrices.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...

for trial_idx, (start, stop) in enumerate(trials):
    ...
    input_trial = np.vstack(...)
    output_trial = np.vstack(...)

for plane_key in plane_keys:
    ...
    all_deconvolved[:, cols] = plane_data
```

iii. The notes mention that the implementation favors bounded memory and direct session-by-session processing over more aggressive vectorization or parallelism.

## 13-c. What processing does the code repeat multiple times?

i. Within each trial, the code repeatedly recomputes per-trial constants and allocations such as `zone_for_trial()`, `np.full(...)`, and `np.vstack(...)`. In `--show-processing` mode it also re-slices raw trial data for plotting after already slicing the same trials for conversion.

ii.
```python
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
...
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        np.full(n_time, env_code, dtype=np.float32),
        np.full(n_time, float(trial_idx), dtype=np.float32),
        np.full(n_time, float(prev_reward), dtype=np.float32),
    ]
)
output_trial = np.vstack(
    [
        dist_bin,
        pos_bin,
        speed_bin,
        lick_bin,
        np.full(n_time, zone_code, dtype=np.int16),
        np.full(n_time, reward_code, dtype=np.int16),
    ]
)
```

iii. The notes focus more on I/O optimizations than on eliminating these smaller repeated per-trial allocations.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the raw `environment` stream but never uses it, creates a local `brain_region_idx` array that is never consumed, and optionally builds plotting-only `raw_examples`. It also loads full-session deconvolved arrays before discarding trials that fail QC.

ii.
```python
environment = behavior["environment/data"][:].astype(np.float32)
...
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
...
raw_examples = []
...
if show_processing and len(raw_examples) < 2:
    raw_examples.append(...)
```

iii. These are side effects of the agent’s implementation strategy: it preferred simpler session-level loading and scene-based labeling over minimizing every discarded intermediate.
