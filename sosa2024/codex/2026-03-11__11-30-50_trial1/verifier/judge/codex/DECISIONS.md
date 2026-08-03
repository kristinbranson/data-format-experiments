# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script gathers every NWB file matching `data/sub-*/sub-*_behavior+ophys.nwb`, sorts that flat file list, and processes each file as one session. It opens each file directly with `h5py.File` rather than `pynwb`, and it does not run a separate survey/discovery pass before conversion.

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

iii. In Step 6 of `CONVERSION_NOTES.md`, the agent says it chose direct `h5py` reads because they were faster than NWB object loading and reduced I/O overhead.

## 1-b. How are the data split into subjects?

i. Subjects are not discovered from directory names. Instead, each session file is read and the subject label is taken from `general/subject/subject_id`; `build_dataset()` appends each new subject the first time it is encountered in the sorted session list.

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

iii. No separate justification is recorded beyond using the NWB metadata directly while iterating through all session files.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The final dataset order is simply the sorted pathname order returned by the glob.

ii.
```python
files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(path=path, ...)
```

iii. This follows the agent’s Step 2 dataset survey notes, which describe one NWB file per session under each subject directory.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from `trial_start` and `teleport`. The script finds all indices where `trial_start > 0.5`, then pairs each start with the first later index where `teleport > 0.5`. The trial slice is `[start:stop]`, so teleport frames are excluded.

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

iii. Step 5 says the agent wanted trial-aligned frames from `trial_start` to teleport onset because that best matched the paper/reference semantics for in-trial samples.

## 1-e. How are trials filtered based on quality controls?

i. The script does not use the human reference’s minimum-length filter. Instead, it drops trials with fewer than 2 position samples and drops trials whose lick stream looks corrupted, defined as more than `0.35` of samples having `lick > 2`.

ii.
```python
LICK_ERROR_FRAC = 0.35

def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC
...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
```

iii. Step 5 and the README justify this as a way to preserve a binary `lick` output without missing labels; Step 10 says bad-lick trial exclusion was intentionally kept and compared against the paper’s reported lick-QC fraction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is taken from `processing/ophys/Deconvolved`, with ROI curation information coming from `processing/ophys/ImageSegmentation/PlaneSegmentation`.

ii.
```python
segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)

deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. Step 5 states that NWB-exported deconvolved calcium activity is the closest available equivalent to the reference code’s `events` signal.

## 2-b. How is the `neural` data processed?

i. The script reorders multi-plane recordings into segmentation-table ROI order, filters to curated ROIs, then slices trial windows and transposes each trial to `(neurons, time)`. It also downcasts the deconvolved matrix to `float16`.

ii.
```python
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    n_frames = deconv_group[plane_keys[0]]["data"].shape[0]
    n_rois = plane_idx.shape[0]
    all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
    for plane_key in plane_keys:
        plane_number = int(plane_key.replace("plane", ""))
        cols = np.flatnonzero(plane_idx == plane_number)
        plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
        ...
        all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]
...
neural_trial = deconvolved[start:stop].T
```

iii. Step 6 says the agent viewed full-session deconvolved loading as a bottleneck and explicitly chose `float16` plus direct HDF5 reads to reduce memory and output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered by the first column of `iscell`; only entries with `iscell[:, 0] > 0.5` are kept.

ii.
```python
iscell = segmentation["iscell"][:]
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconvolved = all_deconvolved[:, curated_idx]
```

iii. Step 5 says this was meant to match Suite2p manual curation and was the agent’s initial neuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start implicitly by slicing each trial from the reconstructed `trial_start` index to the matching teleport index and treating the first sample in that slice as time zero.

ii.
```python
trials = reconstruct_trials(trial_start, teleport)
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. The Step 5 notes say the desired alignment event was trial start and that no further re-alignment was needed once in-trial slices were reconstructed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script hard-codes a frame rate of `15.5078125 Hz` and therefore a time bin size of `1000 / 15.5078125 = 64.483... ms`. No temporal rebinning or interpolation is applied.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
```

iii. Step 3 and Step 5 repeatedly cite the paper’s approximately 15.5 Hz imaging rate and native framewise alignment, so the agent chose to preserve that sampling grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` timestamps, not from the `trial number` timestamps used by the human reference code.

ii.
```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. No explicit written justification appears in the final notes; the code simply uses the position timestamp stream as the session time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, it subtracts the first timestamp in the slice so the first kept sample is `0.0`.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
...
time_trial.astype(np.float32),
```

iii. This matches the agent’s trial-start alignment plan from Step 5.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector and neural data use the same `[start:stop]` indices from the reconstructed trial list, so they are aligned sample-for-sample by shared slicing.

ii.
```python
pos_trial = position[start:stop]
time_trial = position_t[start:stop] - position_t[start]
...
neural_trial = deconvolved[start:stop].T
```

iii. The agent’s Step 5 plan was to keep behavior and neural data on the native imaging-frame grid and align them by common trial indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The final script derives environment type from the NWB `identifier` scene string, parsed by `parse_scene()`, then switched at trial 30 for switch sessions. It loads `BehavioralTimeSeries/environment` but does not use it. This contradicts the earlier Step 5 notes, which had planned to use the behavior stream directly.

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

iii. Step 5 and the README justify scene-based parsing as matching the paper/code switch rule, and Step 10 records a specific edge-case check for a cross-environment switch at trial 30. The internal notes are inconsistent with the final implementation.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `parse_scene()` extracts pre-switch and post-switch environment labels from the session identifier; `zone_for_trial()` chooses the pre- or post-switch environment depending on whether `trial_idx >= 30`; the chosen value is then repeated across all samples in the trial.

ii.
```python
def parse_scene(identifier: str) -> dict:
    ...
    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    if match:
        pre_env, pre_zone, post_env, post_zone = match.groups()
        return {..., "switch": True}

def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```

iii. The justification recorded in Step 5 is that scene metadata should determine switch identity; Step 10 says the agent manually checked the trial-29/trial-30 transition in one switch session.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the reconstructed trial order itself, specifically the `enumerate(trials)` loop index, not from the stored `trial number` timeseries.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. This matches the agent’s Step 5 decision to use reconstructed trial order rather than rely on the raw `trial number` stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is converted to a constant vector repeated across the whole trial.

ii.
```python
np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. No further justification is recorded beyond using a per-trial constant in the decoder input matrix.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `BehavioralTimeSeries/Reward/timestamps`, together with the trial start/stop times taken from the position timestamp stream and the reconstructed trial boundaries.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. Step 5 says trial reward outcome should come from actual delivered reward events, then previous trial outcome should be the prior trial’s reward label.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The helper `reward_outcomes_from_timestamps()` assigns each trial a binary reward outcome based on whether any reward timestamp falls inside that trial’s time span. Then the input for trial `t` is set to the reward outcome of trial `t-1`, with trial 0 defaulting to 0, and that scalar is repeated across time.

ii.
```python
def reward_outcomes_from_timestamps(reward_times, trial_times, trials):
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

iii. Step 5 explicitly says first trial should default to 0 and later trials should repeat the previous trial’s rewarded/omitted status.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance-to-zone is derived from trial position samples plus reward-zone identity inferred from the session identifier and 30-trial switch rule. The final code does not use the `reward_zone` behavior timeseries for this output.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. Step 4 and Step 5 say the agent believed the NWB `reward_zone` samples were occupancy-like rather than A/B/C identity, so it chose scene parsing as the cleaner source of per-trial zone identity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The script computes signed distance to the nearest edge of the active reward zone: negative before the zone, zero inside, positive after the zone. It then converts those distances to category indices.

ii.
```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
    ...
```

iii. The Step 5 mapping notes say this was intended to match the paper’s reward-relative position concept while still obeying the decoder task’s requested discrete bins.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into 7 categories using explicit inequality masks that implement the instruction bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

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

iii. The bins are justified by the decoder-task specification quoted in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position, reward-zone identity, and neural data are all taken from the same trial slice `[start:stop]`, so the distance-bin timeseries shares the neural trial’s sample index exactly.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The Step 5 plan says behavior and neural data should stay on the same imaging-frame grid, making shared slicing the alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from `BehavioralTimeSeries/position/data`.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_bin = discretize_absolute_position(pos_trial)
```

iii. No separate justification is recorded beyond using the raw VR position stream directly.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The script clips positions to the nominal corridor range `[0, 450)` and then bins them into 5 equal-width 90 cm bins.

ii.
```python
TRACK_START_CM = 0.0
TRACK_END_CM = 450.0

def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. Step 5 says the agent wanted bins that covered the 450 cm corridor itself, not the wider observed value range including invalid or out-of-trial positions.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are the 5 equal-width track bins implied by `np.linspace(0, 450, 6)`, namely `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, `[360,450)`, with clipping into the range first.

ii.
```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The justification comes from the instructions’ request for 5 equal-sized bins across the corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same `[start:stop]` indices as the neural data for each trial.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. No separate justification is recorded beyond the general trial-slice alignment strategy.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from `BehavioralTimeSeries/lick/data`.

ii.
```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
```

iii. The Step 5 mapping notes identify the aligned lick timeseries as the source variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script first rejects entire trials whose lick signal appears corrupted (`>35%` of samples with `lick > 2`). For kept trials, it rounds the lick values to the nearest integer, clips them to `[0, 1]`, and stores the result as a binary vector.

ii.
```python
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
...
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. Step 5 and the README justify dropping bad lick trials so the converted lick output can stay binary with no missing entries.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural signals are aligned by using the same trial slice `[start:stop]`.

ii.
```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
lick_bin = binarize_licks(lick_trial)
```

iii. This follows the same shared-slice alignment used for all time-varying behavior outputs.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB `identifier` scene string, parsed into pre/post environment and zone labels, together with the fixed 30-trial switch rule. The final code does not read the `reward_zone` behavior stream for the A/B/C label.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```

iii. Step 4 says the agent treated the behavior `reward_zone` series as occupancy-like and chose scene metadata as the intended source of A/B/C identity, following the paper/code’s switch-session descriptions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The script parses the scene string, applies the trial-30 switch rule for switch sessions, maps `A/B/C` to `0/1/2`, and repeats that scalar across every timepoint in the trial.

ii.
```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
...
np.full(n_time, zone_code, dtype=np.int16),
```

iii. The recorded justification is the same as for environment/reward-zone identity generally: the scene string plus the paper’s 30-trial switch rule were treated as the authoritative per-trial label source.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `BehavioralTimeSeries/Reward/timestamps`, interpreted relative to each trial’s start and stop times.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. Step 5 says reward outcome should reflect actual delivered rewards rather than nominal task structure.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The script scans forward through the sorted reward timestamps and marks a trial as rewarded if the next reward timestamp falls inside that trial’s time span. That binary value is then repeated across all samples in the trial.

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
```

iii. Step 5 explicitly maps reward outcome to actual delivered reward events per trial, then repeats the result across time for decoder formatting.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles only a few cases. It skips empty or degenerate trials (`stop <= start` or `<2` position samples), drops bad-lick trials, and raises an error if multi-plane ROI counts do not match `planeIdx`. It does not implement the human reference’s cropping for neural/behavior length mismatches, reward-time alignment assertions, or NaN-tolerant reward-zone inference.

ii.
```python
if stop <= start:
    continue
...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
...
if plane_data.shape[1] != cols.size:
    raise ValueError(
        f"{path.name}: {plane_key} has {plane_data.shape[1]} ROIs but planeIdx maps {cols.size}"
    )
```

iii. The recorded justification focuses on keeping the lick output label clean and on catching multi-plane reconstruction mistakes; there is no broader robustness policy in the final notes.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s main expensive steps are session I/O, reading the deconvolved matrices, reconstructing multi-plane session matrices, trial-by-trial slicing/stacking, and writing the final pickle. The AI intentionally removed the separate survey pass that appears in the human reference.

ii.
```python
with h5py.File(path, "r") as handle:
    ...
    all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
    for plane_key in plane_keys:
        ...
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(...)
...
with out_path.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 explicitly calls out full-session deconvolved loading as the main bottleneck and says the switch to direct `h5py` reads and `float16` storage was done for speed and memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining serial loops are the trial loop in `load_session()`, the per-plane reconstruction loop for multi-plane sessions, and the per-trial reward-timestamp scan in `reward_outcomes_from_timestamps()`. The code does not attempt to vectorize or parallelize those parts.

ii.
```python
for plane_key in plane_keys:
    ...

for trial_idx, (start, stop) in enumerate(trials):
    ...

for trial_idx, (start, stop) in enumerate(trials):
    start_t = trial_times[start]
    stop_t = trial_times[stop]
    ...
```

iii. The only explicit justification in the notes is pragmatic: Step 6 says sessions are processed sequentially to keep peak memory bounded, and it notes that no parallel file processing was added.

## 13-c. What processing does the code repeat multiple times?

i. Relative to the human reference, the AI removed the large duplicated survey/conversion pass. Within the final script, the main repeated work is per-trial re-creation of constant input/output vectors, repeated parsing use of `zone_for_trial()`, and repeated HDF5 reads for every session during a full run.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    env_name, zone_name = zone_for_trial(scene_info, trial_idx)
    ...
    input_trial = np.vstack([... np.full(n_time, env_code, ...), ...])
    output_trial = np.vstack([... np.full(n_time, zone_code, ...), np.full(n_time, reward_code, ...)])
```

iii. Step 6 frames the overall design as a deliberate tradeoff: process one session at a time, avoid a separate survey pass, and accept the per-trial loop as the cost of variable-length trial extraction.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work in the final script is loading the behavioral `environment` array and never using it, plus constructing a local `brain_region_idx` array inside `load_session()` that is never returned or consumed. The plotting-only `raw_examples` bookkeeping is also discarded outside `--show-processing`.

ii.
```python
environment = behavior["environment/data"][:].astype(np.float32)
...
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
...
raw_examples = []
```

iii. No explicit justification is recorded for those discarded computations; they appear to be leftovers from earlier plans and plotting support.
