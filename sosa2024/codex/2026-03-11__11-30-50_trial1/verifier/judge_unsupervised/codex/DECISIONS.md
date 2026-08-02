# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all sessions by globbing every `sub-*/sub-*_behavior+ophys.nwb` file under `data/`, sorting the file list, and then opening each NWB file directly with `h5py`. Within each file it reads behavioral streams from `processing/behavior/BehavioralTimeSeries` and imaging/segmentation streams from `processing/ophys`.

ii. ```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files

with h5py.File(path, "r") as handle:
    behavior = handle["processing/behavior/BehavioralTimeSeries"]
    segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. In `CONVERSION_NOTES.md`, the agent justified direct NWB loading by treating the NWB files as an exported `sess`-equivalent and using the already aligned behavior plus NWB `Deconvolved` data instead of reconstructing the original paper pipeline from raw scanbox/VR intermediates.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by reading `general/subject/subject_id` from each NWB file, collecting unique subject IDs in encounter order, and storing one `subject_idx` entry per session.

ii. ```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The notes explicitly say the dataset consists of one NWB per session and that subject identity should come from NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The outer list dimension of `neural`, `input`, and `output` is populated one file at a time.

ii. ```python
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(...)
    data["neural"].append(session_data["neural_trials"])
    data["input"].append(session_data["input_trials"])
    data["output"].append(session_data["output_trials"])
```

iii. The agent’s notes describe the native organization as “one NWB file per session” and carry that organization straight into the converted dataset.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavioral impulses: every `trial_start > 0.5` is paired with the next later `teleport > 0.5`, and each trial spans `[start:stop)` frames, excluding the teleport frame itself.

ii. ```python
def reconstruct_trials(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0.5)
    teleports = np.flatnonzero(teleport > 0.5)
    ...
    for start in starts:
        while tp_ptr < len(teleports) and teleports[tp_ptr] <= start:
            tp_ptr += 1
        ...
        stop = teleports[tp_ptr]
        if stop > start:
            trials.append((int(start), int(stop)))
```

iii. In the notes, the agent says this was chosen to mimic the reference semantics “trial start until teleport onset, excluding teleport frames.”

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials with fewer than 2 position samples and drops entire trials when the lick trace looks corrupted: more than `LICK_ERROR_FRAC = 0.35` of samples satisfy `lick > 2`. Sessions with fewer than 2 usable trials are skipped.

ii. ```python
LICK_ERROR_FRAC = 0.35

def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC

if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue

if len(session_data["neural_trials"]) < 2:
    ... continue
```

iii. The notes say the agent removed bad-lick trials because lick is a required decoder output and they did not want to carry missing labels. The notes also cite the paper’s lick-sensor-failure exclusion, although the code uses `0.35` while the notes summarize the paper as `>30%`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from NWB `processing/ophys/Deconvolved/.../data`, after selecting ROIs with `iscell[:,0] > 0.5`. For multiplane sessions, plane datasets are stitched together using `planeIdx`.

ii. ```python
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)

deconv_group = handle["processing/ophys/Deconvolved"]
...
deconvolved = all_deconvolved[:, curated_idx]
...
neural_trial = deconvolved[start:stop].T
```

iii. The notes explicitly state the agent’s key decision: “Neural signal = deconvolved activity,” because the paper’s decoder uses `events` and NWB already exports `Deconvolved`.

## 2-b. How is the `neural` data processed?

i. Processing is limited to ROI filtering, optional multiplane concatenation, casting the arrays to `float16`, slicing by reconstructed trial bounds, and transposing to `(neurons, time)`. The agent does not recompute dF/F, neuropil subtraction, maximin baseline, Gaussian smoothing, or deconvolution.

ii. ```python
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
    for plane_key in plane_keys:
        ...
        plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
        all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]

neural_trial = deconvolved[start:stop].T
```

iii. The notes acknowledge the original pipeline computes dF/F and `events` from fluorescence and neuropil, but the agent decided the exported NWB `Deconvolved` signal was “the closest native match.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level QC in code is `iscell[:,0] > 0.5`. There is no additional interneuron exclusion, no speed-based sample masking, and no explicit NaN masking of the neural timeseries.

ii. ```python
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconvolved = all_deconvolved[:, curated_idx]
```

iii. The notes say this was an “initial neuron filter” matching Suite2p manual curation, while also acknowledging that additional interneuron-style exclusion remained an unresolved consistency check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing every trial from the reconstructed `trial_start` frame through the frame before `teleport`. The timebase used elsewhere in the trial is anchored to `position_t[start]`.

ii. ```python
trials = reconstruct_trials(trial_start, teleport)
...
time_trial = position_t[start:stop] - position_t[start]
neural_trial = deconvolved[start:stop].T
```

iii. The notes repeatedly state the intended temporal alignment event is “trial start” and that trials should exclude teleport frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the imaging frame rate: `15.5078125 Hz`, corresponding to `64.483627... ms` per sample. No temporal rebinning is applied.

ii. ```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
"frame_rate_hz": FRAME_RATE_HZ,
```

iii. The notes say the reference data are already synchronized at imaging-frame resolution and that no rebinning should be introduced unless validation forced it.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position/timestamps` vector in `processing/behavior/BehavioralTimeSeries`.

ii. ```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The notes map “position timestamps” directly to `time_from_trial_start_s` because behavior was already aligned to imaging frames.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent slices the timestamps over `[start:stop)` and subtracts the first timestamp so the trial begins at `0.0`.

ii. ```python
time_trial = position_t[start:stop] - position_t[start]
...
time_trial.astype(np.float32)
```

iii. The notes describe this as a direct trial-relative time coordinate required by the decoder task.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is computed from the exact same start/stop indices used for `neural_trial`, so each timestamp vector is sample-for-sample aligned with each neural frame in the trial.

ii. ```python
time_trial = position_t[start:stop] - position_t[start]
neural_trial = deconvolved[start:stop].T
input_trial = np.vstack([time_trial.astype(np.float32), ...])
```

iii. The notes emphasize that the behavior streams are already on the imaging-frame grid, so shared frame slices were the agent’s alignment strategy.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In code, environment type is not taken from the raw `environment` timeseries even though that variable is loaded. Instead, it is inferred from the session `identifier` string parsed by `parse_scene()` and `zone_for_trial()`.

ii. ```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
environment = behavior["environment/data"][:].astype(np.float32)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```

iii. The notes initially discuss using the valid `environment` values directly, but the implemented code follows the separate “scene identifier drives reward-zone identity” decision and derives the environment from the parsed scene metadata.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses scene strings like `Env1_LocationA_to_B` or `Env1_A_to_Env2_C`, applies the hard-coded switch at trial 30, maps `Env1/Env2` to `0/1`, and repeats the resulting constant across all timepoints in the trial.

ii. ```python
SWITCH_TRIAL = 30
ENV_TO_INT = {"Env1": 0, "Env2": 1}

def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]

np.full(n_time, env_code, dtype=np.float32)
```

iii. The notes justify this by saying scene strings encode environment transitions and should be consistent with the paper’s 30-trial switch rule.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not read from the raw `trial number` behavior variable. It is derived from the order of reconstructed trial intervals, using `enumerate(trials)`.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32)
```

iii. The notes say trial number should come from reconstructed trial order rather than raw `trial number` values during teleport or pre-sync periods.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent assigns a 0-based index to each reconstructed trial and repeats that scalar across the whole trial.

ii. ```python
input_trial = np.vstack(
    [
        ...,
        np.full(n_time, float(trial_idx), dtype=np.float32),
        ...,
    ]
)
```

iii. The notes justify repeating per-trial constants across time so every `input_trial` has a uniform `(n_input, n_timepoints)` shape.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `Reward/timestamps`, combined with the reconstructed trial start/stop times from `position/timestamps` and trial boundaries.

ii. ```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The notes describe this as a per-trial binary variable derived from actual reward delivery, with the first trial defaulting to `0`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent marks each trial as rewarded if any reward timestamp falls between that trial’s start and stop timestamps, then shifts that vector by one trial when constructing the decoder input.

ii. ```python
def reward_outcomes_from_timestamps(reward_times, trial_times, trials):
    ...
    outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)

prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
np.full(n_time, float(prev_reward), dtype=np.float32)
```

iii. The notes say this choice matches the task definition “omitted = 0, rewarded = 1, per trial.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the raw `position` timeseries plus reward-zone coordinates inferred from the parsed scene string (`A/B/C` mapped to fixed cm ranges).

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
ZONE_COORDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The notes say the NWB `reward_zone` stream is not the A/B/C identity and therefore reward-zone identity should be inferred from scene metadata and paper-defined coordinates.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the code computes signed distance to the nearest point in the active reward zone: negative before the zone, zero inside the zone, and positive after the zone.

ii. ```python
distance = np.where(
    position_cm < zone_start,
    position_cm - zone_start,
    np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
)
```

iii. The notes describe this as the decoder-task-required adaptation of the paper’s reward-relative position concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii. ```python
bins = np.full(distance.shape, 6, dtype=np.int16)
bins[distance < -50.0] = 0
bins[(distance >= -50.0) & (distance < -10.0)] = 1
bins[(distance >= -10.0) & (distance < 0.0)] = 2
bins[distance == 0.0] = 3
bins[(distance > 0.0) & (distance <= 10.0)] = 4
bins[(distance > 10.0) & (distance <= 50.0)] = 5
```

iii. This follows the explicit discretization bins in the task instructions, which the notes call out as one of the places where adaptation from the paper was required.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[start:stop)` frame slice as `neural_trial`, so each distance bin is aligned one-to-one with each neural timepoint.

ii. ```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The notes say all behavior outputs were to be kept on the shared imaging-frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the `position` behavior stream.

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes list raw `position` as the source for both absolute and reward-relative position outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to the `[0, 450)` track interval and then digitized into 5 equal-width bins spanning the corridor.

ii. ```python
clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The notes justify this with the task instruction “5 equal-sized bins” on a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 equal bins over the 450 cm track, i.e. nominally 0-90, 90-180, 180-270, 270-360, 360-450 cm.

ii. ```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The notes describe this as a task-specific discretization rather than one copied from the paper’s 10 cm spatial binning.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is generated from the same frame slice used for the neural data in each trial.

ii. ```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes again say the agent kept all variables on the aligned imaging-frame sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. ```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
lick_bin = binarize_licks(lick_trial)
```

iii. The notes identify `sess.timeseries['licks']` in the paper code as the analogous source stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick trace is rounded to the nearest integer and clipped to binary `0/1`. Separately, entire trials are dropped if the raw lick values suggest sensor corruption (`mean(lick > 2) > 0.35`).

ii. ```python
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)

if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
```

iii. The notes justify binary clipping as matching the reference licks stream after correction, and justify dropping corrupted trials because the decoder output should not contain NaNs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same trial indices as the neural data and therefore stays frame-aligned.

ii. ```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
lick_bin = binarize_licks(lick_trial)
```

iii. The notes say lick is one of the aligned behavior timeseries that should stay on the imaging frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB `identifier` scene string, not from the raw `reward_zone` behavior stream.

ii. ```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```

iii. The notes explicitly justify this by saying the raw `reward_zone` stream was occupancy-like and not the A/B/C identity used in the paper.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into pre/post environment and reward-zone labels, a hard switch after trial 30 is applied for switch sessions, and `A/B/C` are mapped to `0/1/2` and repeated across the trial.

ii. ```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
...
zone_code = int(ZONE_TO_INT[zone_name])
...
np.full(n_time, zone_code, dtype=np.int16)
```

iii. The notes cite the paper’s “switch after 30 trials” rule and the fixed reward-zone coordinates as the basis for this conversion.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, together with reconstructed trial intervals anchored by `position/timestamps`.

ii. ```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. The notes state that reward outcome should reflect actual delivered reward events, not intended reward-zone identity.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if at least one reward timestamp falls inside that trial’s time interval; the resulting 0/1 label is repeated across every frame in the trial.

ii. ```python
outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
...
np.full(n_time, reward_code, dtype=np.int16)
```

iii. The notes describe this as a direct mapping to the task’s per-trial “Reward outcome” output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles issues by skipping data rather than repairing it: empty lick trials and trials with fewer than 2 position samples are dropped; corrupted lick trials are dropped; sessions with fewer than 2 remaining trials are skipped; unknown scene strings raise `ValueError`. There is no imputation or explicit masking for NaNs, invalid environment samples, or invalid pre-sync position values.

ii. ```python
if lick_trial.size == 0:
    return True
...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    ... continue
...
raise ValueError(f"Unrecognized scene format: {scene}")
```

iii. The notes frame this as a pragmatic choice to avoid passing malformed labels into the decoder, but they do not describe any attempt to preserve partially valid samples within corrupted trials.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are likely opening all 152 NWB files, materializing deconvolved arrays (especially multiplane sessions), reconstructing per-trial outputs for every frame, and writing the large pickle. The code itself records per-session elapsed time around `load_session()`.

ii. ```python
session_t0 = time.perf_counter()
session_data, stats = load_session(...)
elapsed = time.perf_counter() - session_t0
print(... f"time={elapsed:.2f}s")

for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
    ...
```

iii. The notes identify direct NWB access and framewise trial extraction as the main work, and the code’s logging supports that interpretation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are scalar or per-trial and could be vectorized: pairing starts with teleports in `reconstruct_trials`, scanning rewards in `reward_outcomes_from_timestamps`, the full per-trial loop that repeatedly builds `np.full` arrays, and the multiplane stitching loop.

ii. ```python
for start in starts:
    ...

for trial_idx, (start, stop) in enumerate(trials):
    ...

for plane_key in plane_keys:
    ...
```

iii. The notes do not emphasize optimization, but the implementation clearly favors straightforward loops over vectorized batch operations.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly parses per-trial constants into full-length arrays (`environment`, `trial_number`, `previous_trial_outcome`, `reward_zone_location`, `reward_outcome`), repeatedly calls reward-zone lookup per trial, and recomputes frame-aligned derived outputs separately for every trial.

ii. ```python
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
...
np.full(n_time, env_code, dtype=np.float32)
np.full(n_time, float(trial_idx), dtype=np.float32)
np.full(n_time, float(prev_reward), dtype=np.float32)
np.full(n_time, zone_code, dtype=np.int16)
np.full(n_time, reward_code, dtype=np.int16)
```

iii. The notes explicitly defend repeated per-trial constants as a formatting choice needed to keep all inputs and outputs time-varying.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `environment` from the NWB behavior group but never uses it. It also allocates a local `brain_region_idx` array inside `load_session()` that is never returned, and optionally builds plotting artifacts that are not part of the converted dataset. More broadly, storing per-trial constants as full time series is redundant for downstream analyses that only need one value per trial.

ii. ```python
environment = behavior["environment/data"][:].astype(np.float32)
...
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
...
if show_processing and neural_trials:
    make_processing_plot(...)
```

iii. The notes never justify loading the unused raw `environment` stream or the unused local `brain_region_idx`; those appear to be leftovers from exploration. The notes do justify repeated per-frame constants, but that still creates redundant data that many downstream analyses would collapse immediately.
