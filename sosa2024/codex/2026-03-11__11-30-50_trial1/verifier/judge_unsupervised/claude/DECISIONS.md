# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from NWB (HDF5) files in the `data/` directory. It uses `h5py` to directly access HDF5 groups and datasets. The function `get_session_files()` finds all files matching the glob pattern `data/sub-*/sub-*_behavior+ophys.nwb`, sorts them alphabetically, and returns the list. Each file is then loaded sequentially via `load_session()`. The `build_dataset()` function iterates over all session files and aggregates results.

ii.
```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```
```python
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(path=path, show_processing=show_processing and session_number < 2)
```

iii. The AI identified that the data is stored as NWB/HDF5 files, one per session, organized by subject directories. It chose to use `h5py` directly rather than the slower NWB Python API for performance reasons. This is documented in CONVERSION_NOTES Step 6: "Uses direct HDF5 access with h5py instead of slower NWB object loading."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by reading the `general/subject/subject_id` field from each NWB file. A dictionary `subject_to_idx` maps subject names to indices. As sessions are processed, new subjects are added to the `subjects` list when first encountered. Each session is assigned a subject index via `subject_idx`.

ii.
```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
# ...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The AI identified 11 subjects (mice) from the NWB files, matching the paper's statement of "n = 11 mice" for the switch task. Subject IDs are extracted directly from the NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. The files are sorted alphabetically by path, giving a deterministic ordering by subject then session number. The total is 152 sessions (11 mice x 14 days, minus 2 missing early sessions for m11).

ii.
```python
files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
```

iii. CONVERSION_NOTES Step 4 documents: "11*14 = 154 planned switch sessions, minus 2 missing early m11 sessions = 152." The session ID is also extracted from `general/session_id` in the NWB file for logging purposes.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from two binary behavioral time series: `trial_start` and `teleport`. The function `reconstruct_trials()` finds frame indices where `trial_start > 0.5` (trial onset impulses) and `teleport > 0.5` (teleport impulses). For each trial-start frame, it finds the next teleport frame to define the trial interval `[start, stop)`. The teleport frame itself is excluded from the trial data by Python slice semantics.

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
    return trials
```

iii. CONVERSION_NOTES Step 5: "Reference code keeps samples from trial start until teleport onset. I will reconstruct trials from trial_start impulses to the next teleport impulse, excluding teleport frames." This matches the reference code's `trial_start_inds` to `teleport_inds` segmentation.

## 1-e. How are trials filtered based on quality controls?

i. Three trial-level quality filters are applied: (1) trials where `stop <= start` are skipped, (2) trials with fewer than 2 frames (`pos_trial.size < 2`) are skipped, (3) trials with bad lick sensor data are dropped. The lick QC rule drops trials where >35% of frames have lick counts > 2 (`LICK_ERROR_FRAC = 0.35`). Additionally, sessions with fewer than 2 usable trials after QC are entirely skipped.

ii.
```python
if stop <= start:
    continue
# ...
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
```
```python
def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC
```
```python
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. CONVERSION_NOTES Step 5: "Bad lick trials will be removed: Because lick is a required decoder output and NaNs are undesirable, trials meeting the paper's lick-sensor failure rule will be excluded." The threshold of 0.35 is cited as "Reference code threshold in glmUtils.get_timeseries_data."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the deconvolved calcium activity stored in `processing/ophys/Deconvolved/plane*/data` in the NWB files. Cell identity is filtered using the `iscell` column from `processing/ophys/ImageSegmentation/PlaneSegmentation`. For multi-plane sessions (mice m17, m18), `planeIdx` is used to reconstruct a full-session matrix from separate plane data.

ii.
```python
deconv_group = handle["processing/ophys/Deconvolved"]
plane_keys = sorted(deconv_group.keys(), key=lambda key: int(key.replace("plane", "")))
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    # Multi-plane reconstruction...
    all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]
```

iii. CONVERSION_NOTES Step 5: "Neural signal = deconvolved activity: The reference decoder and place-cell pipeline use deconvolved calcium activity (events). NWB already exports Deconvolved, so this is the closest native match." This maps to `sess.timeseries['events']` in the reference code.

## 2-b. How is the `neural` data processed?

i. The deconvolved activity matrix is read from NWB, filtered to keep only curated ROIs (iscell[:,0] == 1), stored as float16, and then sliced per trial from the trial-start frame to the teleport frame. Each trial's neural data is transposed to shape (n_neurons, n_timepoints).

ii.
```python
iscell = segmentation["iscell"][:]
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
# ...
deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
# ...
neural_trial = deconvolved[start:stop].T
```

iii. The AI uses the pre-computed deconvolved data from the NWB export rather than recomputing dF/F and OASIS deconvolution from raw fluorescence. CONVERSION_NOTES Step 4: "Treat NWB as an exported sess-equivalent. Use NWB aligned behavior streams and map Deconvolved/plane0/data to reference events."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the Suite2p manual curation flag `iscell[:,0] == 1`. The AI does NOT apply the additional interneuron exclusion based on speed-dF/F correlation (> 0.5 threshold) described in the paper. The AI tested a deconvolved-speed correlation proxy and found 0 neurons above the threshold, but this used deconvolved activity instead of dF/F.

ii.
```python
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
```

iii. CONVERSION_NOTES Step 5: "Initial neuron filter = iscell[:,0] == 1: This matches Suite2p manual curation." Step 10: "A deconvolved-speed proxy check on the largest session found 0 neurons above r > 0.5, so this is not the source of the main count mismatch." However, the paper's method uses dF/F correlation, not deconvolved activity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. For each trial, the deconvolved matrix is sliced from the trial_start frame index to the teleport frame index: `deconvolved[start:stop]`. Since behavioral and neural data are already synchronized at the imaging frame rate in the NWB file, no additional alignment is needed. The time axis starts at 0 for each trial (the trial_start frame).

ii.
```python
neural_trial = deconvolved[start:stop].T
```

iii. CONVERSION_NOTES Step 5: "Trial-aligned frames exclude teleport period: Reference code keeps samples from trial start until teleport onset." The instructions specify "Temporally align based on start of the trial."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5078125 Hz, corresponding to a time bin of ~64.48 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```
In metadata:
```python
"time_bin_size": TIME_BIN_MS,
```

iii. CONVERSION_NOTES Step 5: "Use native imaging-frame resolution: The reference data are sampled at ~15.5 Hz and behavior is already synchronized to this grid. No rebinning in time unless a later validation forces it."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps` in the NWB file. These are the frame-wise timestamps of the behavioral/imaging data.

ii.
```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
# ...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The AI uses position timestamps as the common time base, subtracting the trial-start timestamp to get time relative to trial onset. This leverages the fact that behavioral and neural data are already synchronized to imaging frames.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamps array is sliced from start to stop, and the first timestamp is subtracted to get time from trial start in seconds. The result is cast to float32.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
# ...
input_trial = np.vstack([
    time_trial.astype(np.float32),
    # ...
])
```

iii. This is a straightforward subtraction to convert absolute timestamps to trial-relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time array uses the same frame indices as the neural data (both sliced `[start:stop]`), so they are inherently aligned frame-by-frame. Time starts at 0.0 at the first frame of each trial.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]  # same [start:stop] as neural
neural_trial = deconvolved[start:stop].T
```

iii. Since all data streams are already at imaging-frame resolution in the NWB, no interpolation or resampling is needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the NWB `identifier` field, which contains a scene string (e.g., "Env1_LocationA", "Env1_B_to_Env2_C"). The environment is parsed from this string using `parse_scene()` and applied per trial based on the 30-trial switch rule.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
# ...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```
```python
ENV_TO_INT = {"Env1": 0, "Env2": 1}
```

iii. CONVERSION_NOTES Step 5: "Cross-check against scene identifier; encode ENV1=0, ENV2=1." The AI chose to parse environment from the session identifier rather than the behavioral `environment` stream, which contains invalid pre-sync values (-1).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string is parsed with regex to extract environment labels (Env1/Env2). For switch sessions, `zone_for_trial()` determines whether to use pre-switch or post-switch environment based on the 30-trial rule. The environment is encoded as binary: Env1=0, Env2=1. It is a per-trial constant repeated across all time frames.

ii.
```python
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```
```python
np.full(n_time, env_code, dtype=np.float32),
```

iii. The 30-trial switch is consistent with the paper: "Each switch occurred after 30 trials."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the reconstructed within-session trial index (0-based), which comes from enumerating the reconstructed trial list from `trial_start` and `teleport` signals.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    # ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5: "Use reconstructed trial order from trial_start events, not raw trial number during teleport."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is the 0-based index from the enumeration of reconstructed trials. It is a per-trial constant repeated across all frames. Note that this is the raw trial index before any filtering -- if trial 5 is dropped due to lick QC, the next trial still has index 6, not 5.

ii.
```python
np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. The AI uses the raw enumeration index rather than renumbering after filtering. This preserves the original trial ordering information.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` (reward delivery timestamps) and the trial time boundaries. Reward outcomes are computed per trial, then for trial t, the outcome of trial t-1 is used.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
# ...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
# ...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The first trial of each session defaults to 0 (no previous trial). This is a reasonable convention since there is no preceding trial information.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, per-trial reward outcomes are computed by checking whether any reward timestamp falls within each trial's time window [trial_start_time, teleport_time). Then for each trial, the previous trial's outcome is looked up. For the first trial in a session, the default is 0. The result is binary (0=omitted/unrewarded, 1=rewarded) and repeated across all frames.

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
        outcomes[trial_idx] = int(
            reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t
        )
    return outcomes
```

iii. The reward detection uses a sweep-pointer approach over sorted reward timestamps. The use of `stop_t = trial_times[stop]` accesses the teleport frame's timestamp as the upper bound (exclusive), which bounds the reward within the trial period.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from two sources: (1) position data from `processing/behavior/BehavioralTimeSeries/position/data`, and (2) reward zone coordinates derived from the NWB `identifier` scene string (zones A=80-130, B=200-250, C=320-370 cm).

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
# ...
pos_trial = position[start:stop]
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The zone coordinates match the paper: "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if before the zone (position - zone_start), zero if inside [zone_start, zone_end], positive if past the zone (position - zone_end). This is the minimum signed distance to the nearest edge of the reward zone.

ii.
```python
def discretize_distance_to_zone(position_cm, zone_start, zone_end):
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
```

iii. The signed distance preserves directional information (before vs. after zone), consistent with the bin structure in the instructions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 bins matching the instruction specification: 0 (< -50), 1 (-50 to -10), 2 (-10 to < 0), 3 (0, in zone), 4 (>0 to +10), 5 (+10 to +50), 6 (> +50).

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

iii. The bin boundaries match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from the same trial-sliced position data (`position[start:stop]`) that shares frame indices with the neural data. Both have the same number of timepoints per trial, ensuring frame-by-frame alignment.

ii.
```python
pos_trial = position[start:stop]
# ...
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
# both pos_trial and deconvolved[start:stop] use same [start:stop]
```

iii. Alignment is inherent because behavioral and neural data are already synchronized to imaging frames in the NWB.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from `processing/behavior/BehavioralTimeSeries/position/data` in the NWB file.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
pos_trial = position[start:stop]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The raw position is in centimeters on the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to the track range [0, 450) and digitized into 5 equal-width bins using `np.linspace(0, 450, 6)` to create bin edges, then `np.digitize` with the interior edges.

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The 5 equal bins each span 90 cm (0-90, 90-180, 180-270, 270-360, 360-450), consistent with the instruction "discretized into 5 equal-sized bins."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal-width bins are created with edges at 0, 90, 180, 270, 360, 450 cm. `np.digitize` assigns each position to bins 0-4 based on the interior edges [90, 180, 270, 360].

ii.
```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)  # [0, 90, 180, 270, 360, 450]
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The clipping to `np.nextafter(TRACK_END_CM, TRACK_START_CM)` ensures position exactly at 450 cm maps to bin 4 rather than bin 5.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same as distance to reward zone: position is sliced from the same `[start:stop]` frame range as neural data, ensuring frame-by-frame alignment.

ii.
```python
pos_trial = position[start:stop]  # same indices as deconvolved[start:stop]
```

iii. Inherent alignment from shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick data is derived from `processing/behavior/BehavioralTimeSeries/lick/data` in the NWB file.

ii.
```python
lick = behavior["lick/data"][:].astype(np.float32)
lick_trial = lick[start:stop]
lick_bin = binarize_licks(lick_trial)
```

iii. The raw lick data contains frame-wise lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are rounded to the nearest integer and then clipped to the range [0, 1], producing a binary output (0 = no lick, 1 = lick).

ii.
```python
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. This is consistent with the reference code's approach of clipping lick values > 1 to 1. The paper states licks are binary after correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced from the same `[start:stop]` frame range as neural data. Frame-by-frame alignment is inherent.

ii.
```python
lick_trial = lick[start:stop]  # same indices as deconvolved[start:stop]
```

iii. Same alignment mechanism as all other behavioral variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB `identifier` field (session scene string), parsed via `parse_scene()`. The scene string encodes the reward zone identity (A, B, or C) and any switches.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```
```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
```

iii. CONVERSION_NOTES Step 5: "Scene identifier drives reward-zone identity: NWB reward_zone is not the A/B/C label. Active reward-zone identity will be parsed from the session identifier." The zone_for_trial function handles the 30-trial switch rule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regex patterns to extract pre- and post-switch zone identities. For each trial, `zone_for_trial()` checks if the session has a switch and whether the trial index is >= 30 (the switch point). The zone is encoded as 0=A, 1=B, 2=C and repeated across all frames as a per-trial constant.

ii.
```python
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```
```python
np.full(n_time, zone_code, dtype=np.int16),
```

iii. SWITCH_TRIAL = 30, matching the paper: "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` in the NWB file. These are the actual timestamps of reward delivery events.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
reward_code = int(reward_outcomes[trial_idx])
```

iii. The AI uses reward delivery timestamps rather than a binary reward signal to determine per-trial reward outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward timestamp falls within the trial's time window [trial_start_time, teleport_time). If yes, reward_outcome = 1; otherwise 0. The result is a per-trial binary constant repeated across all frames.

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
        outcomes[trial_idx] = int(
            reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t
        )
    return outcomes
```

iii. The reward rate is ~84.2% rewarded, consistent with the paper's ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Trials with too few frames (< 2) are silently skipped. (2) Trials with corrupted lick sensor data are dropped via the lick QC rule. (3) Sessions with fewer than 2 usable trials after filtering are skipped entirely. (4) The last trial in a session may be skipped if there is a trial_start with no subsequent teleport. (5) Pre-sync invalid values (position = -500, environment = -1) are implicitly handled by only processing in-trial frames. (6) Multi-plane data reconstruction handles varying ROI counts per plane.

ii.
```python
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
# ...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. CONVERSION_NOTES Step 10 documents edge cases: "sub-m11_ses-03 has 81 nonnegative trial numbers but only 80 trial_start and teleport events. Converter follows trial_start/teleport and therefore excludes one partial/non-aligned trial."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the full deconvolved activity matrix from each NWB file. For large multi-plane sessions, this involves reading and reconstructing the full (frames x ROIs) matrix from disk. The full conversion processes 152 sessions at approximately 1.5-2.5 seconds per session, totaling ~4-7 minutes.

ii.
```python
deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
```

iii. CONVERSION_NOTES Step 7: "Large multi-plane benchmark (m18 ses-03): ~2.49 s/session."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main trial-processing loop iterates over each trial sequentially. The `reward_outcomes_from_timestamps` function loops over trials to match reward timestamps. The `reconstruct_trials` function uses a two-pointer loop. These could potentially be vectorized with numpy operations (e.g., using `np.searchsorted` for reward matching, vectorized slicing for trial extraction).

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    # ... per-trial processing
```
```python
for trial_idx, (start, stop) in enumerate(trials):
    start_t = trial_times[start]
    stop_t = trial_times[stop]
    while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
        reward_ptr += 1
    # ...
```

iii. The AI noted this in CONVERSION_NOTES Step 6: "Full-session deconvolved activity is currently loaded into memory per session before trial slicing." However, the overall runtime (~4-7 minutes) was deemed acceptable.

## 13-c. What processing does the code repeat multiple times?

i. The session loading function reads the entire deconvolved matrix into memory once per session, then slices it per trial. Position, speed, lick, and timestamps are similarly loaded fully then sliced. The `zone_for_trial` function is called for each trial but always returns the same result for trials in the same switch segment. The `decode_h5_scalar` function is called multiple times for different HDF5 fields.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
# ... loaded once, sliced multiple times per trial
pos_trial = position[start:stop]
```

iii. The per-session loading approach means no data is loaded more than once per session, but per-trial constants (environment, zone) are recomputed for each trial rather than precomputed once per segment.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores per-trial constants (environment, trial number, previous outcome, reward zone location, reward outcome) as full time-varying arrays repeated across every frame. While this conforms to the target format specification, it is storage-inefficient since these values don't change within a trial. The processing plots are only generated for up to 2 sessions but the code structure handles the flag check per session. The `raw_examples` collection for visualization is a minor overhead. Speed is loaded and processed for all sessions even though it is primarily a straightforward discretization step.

ii.
```python
np.full(n_time, env_code, dtype=np.float32),
np.full(n_time, float(trial_idx), dtype=np.float32),
np.full(n_time, float(prev_reward), dtype=np.float32),
# ...
np.full(n_time, zone_code, dtype=np.int16),
np.full(n_time, reward_code, dtype=np.int16),
```

iii. The repeated per-trial constants are required by the target format specification "(n_input, n_timepoints) or (n_input)" and "(n_output, n_timepoints) or (n_output)". The AI chose the time-varying representation for uniformity.
