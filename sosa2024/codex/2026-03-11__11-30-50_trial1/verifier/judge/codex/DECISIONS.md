# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed all NWB files under `data/sub-*`, then processed each file as one session. Within each file it read behavior and ophys arrays directly with `h5py`, then later reconstructed trials from `trial_start` and `teleport`.

ii. ```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files

with h5py.File(path, "r") as handle:
    behavior = handle["processing/behavior/BehavioralTimeSeries"]
    ...
    deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. The notes say the NWB export already contains aligned behavior plus deconvolved activity, so the AI treated each NWB as the released per-session container and used direct HDF5 reads for speed.

## 1-b. How are the data split into subjects?

i. Subjects are implicitly split by NWB file path and by the stored subject metadata inside each file. During dataset assembly, each session’s `subject` is mapped into `data["subjects"]` and `subject_idx`.

ii. ```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
...
subject_idx.append(subject_to_idx[subject])
```

iii. The notes describe one subject folder per mouse and state that subject IDs in the NWB metadata match those folders, so the AI used the metadata value when assembling the final dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The converter loops over the sorted list of NWB file paths and appends one entry per file to `neural`, `input`, `output`, and metadata.

ii. ```python
files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(path=path, ...)
    ...
    data["neural"].append(session_data["neural_trials"])
```

iii. The notes say the released dataset has one NWB file per session, matching the subject/day organization in the paper.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from each `trial_start` pulse to the next `teleport` pulse, excluding the teleport frame itself. The converter stores each trial as the slice `start:stop`.

ii. ```python
def reconstruct_trials(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0.5)
    teleports = np.flatnonzero(teleport > 0.5)
    ...
    for start in starts:
        ...
        stop = teleports[tp_ptr]
        if stop > start:
            trials.append((int(start), int(stop)))

for trial_idx, (start, stop) in enumerate(trials):
    pos_trial = position[start:stop]
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes say this was chosen to match the repo’s “trial start to teleport onset” slicing and the task’s requested alignment to trial start.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with fewer than 2 position samples and drops any trial where more than 35% of lick samples exceed 2, treating those as bad lick-sensor trials. Sessions with fewer than 2 remaining trials are skipped entirely.

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
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The notes explicitly say the AI imported the paper’s bad-lick trial rule because lick is a required decoder output and it preferred dropping those trials over carrying missing lick labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal comes from the NWB `processing/ophys/Deconvolved` data, with ROI curation from `ImageSegmentation/PlaneSegmentation/iscell` and `planeIdx` for multi-plane sessions.

ii. ```python
segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)

deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. The notes say the reference decoder uses deconvolved calcium activity (`events`), and the NWB export already stores that signal directly.

## 2-b. How is the `neural` data processed?

i. The AI reconstructs a whole-session deconvolved matrix, reassembles multi-plane recordings into segmentation-table order when needed, filters to curated ROIs, slices trials, transposes to neuron-by-time, and stores the result as `float16`.

ii. ```python
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

neural_trial = deconvolved[start:stop].T
```

iii. The notes justify this as matching the released NWB packaging while fixing a multi-plane bug and reducing output size by writing neural data as `float16`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered solely by the Suite2p curation flag `iscell[:, 0] > 0.5`. No additional interneuron or speed-correlation exclusion is implemented.

ii. ```python
iscell = segmentation["iscell"][:]
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconvolved = all_deconvolved[:, curated_idx]
```

iii. The notes say the AI regarded `iscell` as the primary curation exported into NWB and checked, but did not add, the paper’s extra speed-correlation interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by reconstructing each trial from `trial_start` to the next `teleport` and then slicing the deconvolved matrix on that interval. No additional time shift is applied.

ii. ```python
trials = reconstruct_trials(trial_start, teleport)
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes repeatedly state that trial start is the alignment event requested by the instructions and that the converter therefore uses track-only slices starting at `trial_start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are kept at the native imaging-frame resolution, hard-coded as `15.5078125 Hz`, or about `64.5 ms` per bin. No temporal rebinning or resampling is applied.

ii. ```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "frame_rate_hz": FRAME_RATE_HZ,
```

iii. The notes say the reference analyses operate at the native imaging rate and that behavior is already synchronized to that grid, so the AI kept the raw frame resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` timestamps, specifically `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The notes say the behavior streams are aligned at imaging-frame resolution, so any synchronized behavior timestamp stream would work; the AI used position timestamps directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp is subtracted from all timestamps in that trial so time starts at `0`.

ii. ```python
time_trial = position_t[start:stop] - position_t[start]
...
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        ...
    ]
)
```

iii. The notes justify this as the direct implementation of “time from start of trial.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned by using the same reconstructed trial slice indices for both timestamps and neural activity. The AI assumes the position timestamps and deconvolved frames share the same sample grid.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    time_trial = position_t[start:stop] - position_t[start]
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes say all behavioral and neural time series in the release are already synchronized at imaging-frame resolution, so indexing them with the same `start:stop` slice should preserve alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from the session `identifier` string by parsing scene names such as `Env1_LocationA_to_B` or `Env1_B_to_Env2_C`. The raw `environment` time series is loaded but not used for the final input.

ii. ```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
environment = behavior["environment/data"][:].astype(np.float32)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```

iii. The notes say the repo uses scene-based labels and that the NWB `environment` field is an encoded behavior stream rather than the higher-level session label, so the AI chose the scene metadata as the authoritative source.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `identifier` string is parsed into pre-switch and post-switch environment labels, and then `zone_for_trial` chooses the appropriate environment based on whether the trial index is before or after trial 30. That per-trial constant is then repeated across all frames in the trial.

ii. ```python
def parse_scene(identifier: str) -> dict:
    ...
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]

input_trial = np.vstack(
    [
        ...,
        np.full(n_time, env_code, dtype=np.float32),
        ...
    ]
)
```

iii. The notes justify this with the paper’s “switch after 30 trials” rule and by treating the identifier as the clean source of session context.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from a stored NWB variable. It is derived from the reconstructed within-session trial index `trial_idx` during the trial loop.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. The notes say trial order should come from the `trial_start`/`teleport` reconstruction rather than trusting the raw `trial number` stream when the two disagree.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The only processing is to use the 0-based trial-loop index and repeat it across all frames of the trial.

ii. ```python
input_trial = np.vstack(
    [
        ...,
        np.full(n_time, float(trial_idx), dtype=np.float32),
        ...
    ]
)
```

iii. The notes justify this as the natural session-local trial counter once trial boundaries have been reconstructed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `Reward/timestamps` and the reconstructed trial time intervals from `position/timestamps`.

ii. ```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The notes say reward delivery is the correct source for outcome, and they preferred using raw reward timestamps directly against trial intervals.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The converter first computes a per-trial binary `reward_outcomes` array by checking whether any reward timestamp falls between each trial’s start and stop time. It then shifts that vector by one trial, using `0` for the first trial, and repeats the value across time.

ii. ```python
def reward_outcomes_from_timestamps(reward_times, trial_times, trials):
    outcomes = np.zeros(len(trials), dtype=np.int8)
    ...
    outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes

prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
...
np.full(n_time, float(prev_reward), dtype=np.float32),
```

iii. The notes justify this as matching the task definition “omitted = 0, rewarded = 1” while avoiding an unnecessary intermediate framewise reward vector.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the `position` behavior stream plus reward-zone identity inferred from the session `identifier` string and the 30-trial switch rule. The raw `reward_zone` behavior stream is not used.

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
...
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The notes say the NWB `reward_zone` stream is not an A/B/C label, so the AI treated scene metadata plus the fixed paper coordinates as the cleaner source of trial reward-zone identity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each position sample, the converter computes signed distance to the active zone: negative before the zone, `0` inside it, and positive after it.

ii. ```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
```

iii. The notes tie this directly to the paper’s fixed reward-zone coordinates and the decoder task’s requested signed distance variable.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI manually thresholds the continuous signed distance into seven categories matching the task bins: `<-50`, `-50:-10`, `-10:0`, `0`, `0:10`, `10:50`, `>50`.

ii. ```python
bins = np.full(distance.shape, 6, dtype=np.int16)
bins[distance < -50.0] = 0
bins[(distance >= -50.0) & (distance < -10.0)] = 1
bins[(distance >= -10.0) & (distance < 0.0)] = 2
bins[distance == 0.0] = 3
bins[(distance > 0.0) & (distance <= 10.0)] = 4
bins[(distance > 10.0) & (distance <= 50.0)] = 5
```

iii. The notes say these thresholds were chosen to match the decoder specification exactly rather than introducing any paper-specific alternative.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position samples and neural samples are sliced with the same `start:stop` trial interval, then distance bins are computed from that per-trial position vector.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    pos_trial = position[start:stop]
    ...
    neural_trial = deconvolved[start:stop].T
    dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The notes say the behavior streams are already aligned to imaging frames in the NWB release, so shared trial indexing is sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the `position` behavior time series.

ii. ```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
...
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes treat raw VR corridor position as already the desired continuous source variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips positions to the track interval `[0, 450)` and then bins them into five equal-width bins using `np.digitize`.

ii. ```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The notes justify this with the paper’s 450 cm track length and the task’s request for five equal-sized bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The converter uses five equal 90 cm bins over the 0 to 450 cm track after clipping values into that interval.

ii. ```python
TRACK_START_CM = 0.0
TRACK_END_CM = 450.0
...
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The notes say this was chosen because the instructions asked for “5 equal-sized bins” and the track length is 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned to neural activity by taking the same `start:stop` trial slice from the position vector and the deconvolved activity matrix.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    pos_trial = position[start:stop]
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes say no extra alignment step was needed because the NWB behavior streams are already synchronized to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the `lick` behavior time series.

ii. ```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
...
lick_bin = binarize_licks(lick_trial)
```

iii. The notes identify the NWB lick stream as the direct source of the lick output, subject to the bad-lick QC rule.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first rejects entire trials with too many corrupted lick samples (`>2`), then binarizes the remaining lick values by rounding and clipping them into `0/1`.

ii. ```python
def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    ...
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC

def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. The notes say this mirrors the paper’s lick-sensor failure handling and yields the required binary decoder target.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same reconstructed trial interval from the lick series and the deconvolved neural matrix.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    lick_trial = lick[start:stop]
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes say the aligned NWB release lets the converter use shared trial indices without an additional interpolation step.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the session `identifier` string, parsed into environment and reward-zone labels, then combined with the fixed switch-after-30-trials rule. The raw `reward_zone` stream is not used as the label source.

ii. ```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```

iii. The notes say the AI regarded scene metadata as the authoritative A/B/C zone identity because the sampled `reward_zone` field was only useful as within-zone occupancy.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The converter parses the scene string, decides whether the trial is pre- or post-switch, maps `A/B/C` to `0/1/2`, and repeats that code across the trial.

ii. ```python
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]

np.full(n_time, zone_code, dtype=np.int16)
```

iii. The notes explicitly cite the paper’s switch rule and the identifier examples as the basis for this processing.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, evaluated against each reconstructed trial’s start and stop time.

ii. ```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
reward_code = int(reward_outcomes[trial_idx])
```

iii. The notes say actual delivered reward events are the clean source for trial outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI computes a per-trial binary array indicating whether any reward timestamp lies within that trial’s interval, then repeats the current trial’s value across all frames.

ii. ```python
def reward_outcomes_from_timestamps(reward_times, trial_times, trials):
    ...
    outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes

output_trial = np.vstack(
    [
        ...,
        np.full(n_time, reward_code, dtype=np.int16),
    ]
)
```

iii. The notes justify the repeated per-trial label as matching the decoder task specification even though the paper also uses time-varying reward-state variables in other analyses.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases procedurally: empty lick trials are treated as bad, very short trials are skipped, sessions with too few remaining trials are skipped, and multi-plane ROI ordering mismatches raise errors. It does not implement the reference solution’s behavior/neural length cropping or reward-zone inference from missing within-zone observations.

ii. ```python
if lick_trial.size == 0:
    return True
...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
...
if plane_data.shape[1] != cols.size:
    raise ValueError(...)
...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The notes frame these as pragmatic release-data checks, especially to avoid propagating bad lick labels and to fail loudly if multi-plane reconstruction is inconsistent.

## 13-a. What are the most time-consuming steps of the code?

i. The costly parts are session-by-session file I/O, loading whole-session deconvolved matrices, reconstructing multi-plane sessions, looping over every trial to slice arrays and build outputs, and optional plotting. The notes also flag that loading full-session deconvolved activity dominates memory and runtime.

ii. ```python
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(...)

all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
for plane_key in plane_keys:
    plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
    all_deconvolved[:, cols] = plane_data

for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The notes explicitly say “Full-session deconvolved activity is currently loaded into memory per session before trial slicing” and that there is no parallel file processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that repeatedly slices behavior/neural arrays and constructs many `np.full` arrays could be partly vectorized, and the multi-plane reassembly loop could be replaced with a more direct column gather if the ROI ordering were precomputed. The top-level session loop could also be parallelized, though that is coarse-grained rather than vectorization.

ii. ```python
for plane_key in plane_keys:
    ...
    all_deconvolved[:, cols] = plane_data

for trial_idx, (start, stop) in enumerate(trials):
    ...
    input_trial = np.vstack([... np.full(n_time, ...), ...])
    output_trial = np.vstack([... np.full(n_time, ...), ...])
```

iii. The notes mention the lack of parallel file processing and identify whole-session loading plus repeated per-trial work as the main inefficiencies.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly computes per-trial constants with `np.full`, repeatedly parses trial-specific zone/environment through `zone_for_trial`, and repeatedly slices aligned arrays for each variable within the trial loop. Optional plotting also reuses and re-derives several already-computed trial signals.

ii. ```python
for trial_idx, (start, stop) in enumerate(trials):
    pos_trial = position[start:stop]
    speed_trial = speed[start:stop]
    lick_trial = lick[start:stop]
    ...
    env_name, zone_name = zone_for_trial(scene_info, trial_idx)
    ...
    np.full(n_time, env_code, dtype=np.float32)
    np.full(n_time, float(trial_idx), dtype=np.float32)
    np.full(n_time, float(prev_reward), dtype=np.float32)
```

iii. The notes emphasize repeated per-trial construction of time-varying matrices from constants because the final format requires every trial to have `(d, T)` arrays.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter loads the raw `environment` time series but never uses it, creates a local `brain_region_idx` array inside `load_session` that is never returned, and does extra plotting/statistics work that is only diagnostic. It also stores verbose `session_info` metadata that the downstream decoder does not need.

ii. ```python
environment = behavior["environment/data"][:].astype(np.float32)
...
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
...
raw_examples = []
...
stats = {
    ...
}
data["metadata"]["session_info"].append(stats)
```

iii. The notes frame these as convenience and validation features rather than outputs required by the decoder, and the unused `environment` stream is a side effect of the scene-based environment decision.
