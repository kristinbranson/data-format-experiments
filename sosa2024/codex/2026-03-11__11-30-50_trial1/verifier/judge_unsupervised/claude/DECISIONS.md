# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by globbing all NWB files matching `data/sub-*/sub-*_behavior+ophys.nwb`, sorting them alphabetically, and processing each file sequentially with `h5py`. Each NWB file represents one session. Within each session, behavioral timeseries (position, speed, lick, trial_start, teleport, environment, reward timestamps) and optical physiology data (deconvolved activity, iscell, planeIdx) are read directly from the HDF5 groups.

ii.
```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```
```python
def load_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        identifier = decode_h5_scalar(handle["identifier"])
        subject = decode_h5_scalar(handle["general/subject/subject_id"])
        session_id = decode_h5_scalar(handle["general/session_id"])
        region = decode_h5_scalar(handle["general/optophysiology/ImagingPlane/location"])
        ...
        behavior = handle["processing/behavior/BehavioralTimeSeries"]
        position = behavior["position/data"][:].astype(np.float32)
        ...
```

iii. The AI justified this by noting that the NWB files are exported versions of the reference code's `sess` objects and can be read directly with `h5py` for efficiency, avoiding the slower NWB object loading. This is documented in CONVERSION_NOTES Step 6.

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file contains a `general/subject/subject_id` field identifying the mouse. The AI reads this field and uses a `subject_to_idx` dictionary to assign a unique integer index to each subject. All 11 subjects from the data directory are included.

ii.
```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
```

iii. The AI documented 11 switch-task mice in the dataset, matching the paper's description of n=11 mice. Subject m11 has 12 sessions (starting on day 3), all others have 14, totaling 152 sessions.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes them one at a time via `load_session()`. Sessions with fewer than 2 usable trials after quality control are skipped.

ii.
```python
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(path=path, ...)
    if len(session_data["neural_trials"]) < 2:
        print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
        continue
```

iii. The AI justified the minimum 2-trial requirement based on the target format specification which states "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-d. How are the data split into trials?

i. Trials are reconstructed from binary frame-wise signals `trial_start` and `teleport` in the NWB behavioral timeseries. The AI finds frame indices where `trial_start > 0.5` and pairs each start with the next `teleport > 0.5` frame, creating `(start, stop)` tuples. The trial spans from the trial_start frame to the teleport frame (exclusive of teleport).

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

iii. The AI documented that the reference code uses `trial_start_inds` to `teleport_inds` for trial boundaries, with in-trial samples being from trial start until teleport onset. The NWB stores these as frame-wise binary signals rather than index arrays, so reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) `stop <= start` trials are skipped, (2) trials with fewer than 2 frames are skipped, and (3) trials with lick sensor errors are dropped. Lick sensor error is detected when >35% of frames in a trial have cumulative lick count > 2, matching the reference code's `get_timeseries_data` threshold.

ii.
```python
LICK_ERROR_FRAC = 0.35

def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC
```
```python
for trial_idx, (start, stop) in enumerate(trials):
    if stop <= start:
        continue
    ...
    if pos_trial.size < 2:
        continue
    if is_bad_lick_trial(lick_trial):
        dropped_bad_lick += 1
        continue
```

iii. The AI justified the 0.35 threshold by referencing the code: "Reference code threshold in glmUtils.get_timeseries_data." The paper states 30% while the code uses 35%; the AI followed the code. The AI noted 70 trials were dropped (~0.57% of 12,217), close to the paper's reported 81/12,376 (0.65%).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` group in the NWB ophys processing hierarchy (`processing/ophys/Deconvolved/plane{N}/data`), filtered by the `iscell` column from `ImageSegmentation/PlaneSegmentation`.

ii.
```python
segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)

deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. The AI justified using deconvolved activity as the neural signal because "The reference decoder and place-cell pipeline use deconvolved calcium activity (`events`). NWB already exports `Deconvolved`, so this is the closest native match."

## 2-b. How is the `neural` data processed?

i. The deconvolved data is read from NWB, filtered to curated ROIs (iscell[:,0] == 1), and sliced per trial from trial_start to teleport frame. For multi-plane sessions, data from multiple planes are reconstructed into a single matrix using `planeIdx` before applying the iscell filter. The data is stored as float16 and transposed to (n_neurons, n_timepoints) per trial.

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
        all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]
...
neural_trial = deconvolved[start:stop].T
```

iii. The AI noted this is a direct read of pre-computed deconvolved activity from NWB, which corresponds to the reference code's `sess.timeseries['events']`. No additional dF/F computation or deconvolution is performed because the NWB already stores the result.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered solely by the Suite2p `iscell[:,0] > 0.5` criterion, which selects manually curated ROIs. The AI did not implement the reference code's additional interneuron exclusion (correlation between dF/F and running speed > 0.5).

ii.
```python
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
```

iii. The AI acknowledged the missing interneuron exclusion in CONVERSION_NOTES Step 4 and Step 10, noting it as a known discrepancy. In Step 10, the AI attempted a proxy check using deconvolved-speed correlation on the largest session and found 0 neurons above r > 0.5, concluding this was "not the source of the main count mismatch." However, the reference uses dF/F-speed correlation, not deconvolved-speed correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data spans from the `trial_start` frame to the `teleport` frame (exclusive). The time axis starts at 0 for each trial.

ii.
```python
neural_trial = deconvolved[start:stop].T
```
Where `start` is the frame index of `trial_start > 0.5` and `stop` is the next frame index of `teleport > 0.5`.

iii. The AI noted: "Trial-aligned frames exclude teleport period... Reference code keeps samples from trial start until teleport onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5078125 Hz, giving a time bin of ~64.48 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The AI justified: "Use native imaging-frame resolution... The reference data are sampled at ~15.5 Hz and behavior is already synchronized to this grid. No rebinning in time unless a later validation forces it."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the NWB behavioral timeseries.

ii.
```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The AI noted that behavior timestamps are aligned to imaging frames and provide the time axis for all behavioral data.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamps at the trial window (`start:stop`) are extracted and the first timestamp is subtracted to make time relative to trial start (starting at 0).

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
...
input_trial = np.vstack([
    time_trial.astype(np.float32),
    ...
])
```

iii. No additional justification needed; this directly implements the instruction "Time from start of trial in seconds."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural data and time data use the same frame indices (`start:stop`), so they are inherently aligned frame-by-frame.

ii.
```python
pos_trial = position[start:stop]
time_trial = position_t[start:stop] - position_t[start]
neural_trial = deconvolved[start:stop].T
```

iii. The AI justified: "Neural and behavioral data are analyzed at the imaging frame rate (~15.5 Hz, ~64.5 ms). Reference code aligns Unity VR data to imaging frames before any later analyses."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the NWB `identifier` field (the scene string, e.g. `Env1_LocationA_to_B`) via the `parse_scene` function, combined with the 30-trial switch rule.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```

iii. The AI noted the NWB `environment` timeseries contains valid values 0/1 plus pre-sync invalid -1, but chose to derive environment from the scene identifier for reliability, cross-checking against the switch rule.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string from the NWB identifier is parsed with regex to extract pre/post environment labels. For switch sessions (e.g., `Env1_B_to_Env2_C`), the environment changes at trial 30. The environment is encoded as 0 (ENV1) or 1 (ENV2) and repeated across all frames in the trial.

ii.
```python
def parse_scene(identifier: str) -> dict:
    scene = identifier.rstrip("/").split("/")[-1]
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    ...
    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    ...

def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```

iii. The AI justified using the scene identifier because the reference code similarly infers environment from scene metadata. ENV_TO_INT maps Env1->0, Env2->1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-based index from the reconstructed trial list within each session, not from the NWB `trial number` timeseries.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. The AI noted that the NWB `trial number` includes invalid pre-sync -1 values and decided to use the reconstructed within-session trial index, which is "consistent with `glmUtils.get_timeseries_data` (`trials`)."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the loop index from iterating over the reconstructed trial list. It is cast to float32 and repeated across all frames in the trial.

ii.
```python
np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. No complex processing needed; this is a straightforward enumeration of trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the `Reward/timestamps` array in NWB behavioral timeseries, which records the times of reward delivery events.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The AI determines reward outcome per trial by checking whether any reward timestamp falls within the trial's time window. The previous trial's outcome is then used for the current trial, with the first trial defaulting to 0 (no previous outcome).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward outcomes are computed by iterating through trials and checking if any reward timestamp falls within each trial's time window (`start_t` to `stop_t`). Previous trial outcome for trial `t` uses the outcome of raw trial `t-1` (even if that trial was later dropped by QC). The first trial defaults to 0.

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
```

iii. The AI noted this is "Binary, 0=omitted/unrewarded, 1=rewarded" and relates to the reference code's `behavior.get_trial_types`. The reference code checks both `reward > 0` AND `rzone > 0`, while the AI only checks reward timestamps.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the NWB `position/data` timeseries and the reward zone coordinates parsed from the session identifier scene string.

ii.
```python
pos_trial = position[start:stop]
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
...
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The AI derived reward zone identity from the scene string and the 30-trial switch rule, using paper-defined zone coordinates: A=(80,130), B=(200,250), C=(320,370).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the reward zone is computed: negative if the animal is before the zone, zero if inside, positive if past. Then discretized into 7 bins.

ii.
```python
def discretize_distance_to_zone(position_cm, zone_start, zone_end):
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

iii. The AI derived this from the paper's reward zone definitions and the decoder task specification's bin boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is thresholded into 7 bins: 0 (< -50 cm), 1 (-50 to -10), 2 (-10 to <0), 3 (0, in zone), 4 (>0 to +10), 5 (+10 to +50), 6 (>+50 cm). This matches the instruction specification exactly.

ii. See code snippet in 7-b above.

iii. The bin boundaries are taken directly from the decoder task specification in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance to reward zone uses the same frame indices (`start:stop`) as the neural data, so alignment is frame-by-frame at the native imaging rate.

ii.
```python
pos_trial = position[start:stop]
...
neural_trial = deconvolved[start:stop].T
```

iii. Both position and neural data are sampled at the same imaging frame rate and indexed identically.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the NWB `position/data` timeseries.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. Position data is directly available in the NWB behavioral timeseries.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to the track range [0, 450) cm and digitized into 5 equal-width bins of 90 cm each using `np.linspace(0, 450, 6)` edges.

ii.
```python
def discretize_absolute_position(position_cm):
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The AI uses 5 equal-sized bins across the 450 cm track, as specified in the instructions: "Absolute position in corridor, discretized into 5 equal-sized bins."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm track is divided into 5 equal bins of 90 cm each: bin 0 (0-90 cm), bin 1 (90-180 cm), bin 2 (180-270 cm), bin 3 (270-360 cm), bin 4 (360-450 cm). `np.digitize` with `right=False` puts values at bin edges into the higher bin.

ii. See code snippet in 8-b above.

iii. This matches "5 equal-sized bins" from the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-by-frame alignment as all other variables, using the same `start:stop` indices.

ii. Position and neural data are both sliced with `position[start:stop]` and `deconvolved[start:stop].T`.

iii. Alignment is inherent from the NWB structure where behavioral and neural data share the same frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the NWB `lick/data` timeseries in the behavioral data group.

ii.
```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
```

iii. The lick signal is a cumulative lick count per imaging frame, as stored in the NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The cumulative lick count is binarized: values are rounded and clipped to [0, 1]. Trials with lick sensor errors (>35% of frames with lick > 2) are dropped entirely rather than included with missing data.

ii.
```python
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. The AI justified binarization because the decoder task spec requires "Lick, time-varying. 0 = no, 1 = yes." The reference code similarly clips licks to binary (`licks[licks > 1] = 1`). The AI does not apply the Gaussian smoothing (sigma=2) that the reference code applies for GLM/decoder analyses.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-by-frame alignment using identical `start:stop` indices.

ii.
```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. All behavioral and neural data share the same frame indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field (session scene string) and the 30-trial switch rule, not from the NWB `reward_zone` timeseries.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```

iii. The AI noted that the NWB `reward_zone` values are not A/B/C labels but rather occupancy signals, so the scene identifier is the reliable source for zone identity. The reference code's `get_reward_zones` similarly derives zones from scene names.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed to extract the pre- and post-switch zone labels (A/B/C). For trials before trial 30, the pre-switch zone is used; for trials >= 30, the post-switch zone is used. Zone labels are mapped to integers: A=0, B=1, C=2. The value is repeated across all frames in the trial.

ii.
```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
SWITCH_TRIAL = 30

def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```

iii. The 30-trial switch boundary matches the paper's "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` array in the NWB behavioral timeseries.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
reward_code = int(reward_outcomes[trial_idx])
```

iii. Reward timestamps record the actual time of reward delivery, allowing binary reward outcome determination per trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward timestamp falls within the trial's time window (from `trial_times[start]` to `trial_times[stop]`). If yes, reward_outcome = 1; otherwise 0. The value is repeated across all frames in the trial.

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
```

iii. The AI noted the ~84.2% reward rate in the converted data matches the paper's ~85% (with ~15% omissions).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of data issues: (1) Lick sensor errors: trials with >35% bad lick frames are dropped entirely. (2) Very short trials (< 2 frames): skipped. (3) Invalid trial boundaries (stop <= start): skipped. (4) Pre-sync invalid values (-1 in environment, -500 in position): avoided by only processing within trial boundaries. (5) Sessions with < 2 usable trials: skipped. (6) Multi-plane session mismatch: plane data reconstructed using planeIdx mapping. (7) Missing m11 sessions (days 1-2): naturally absent from NWB files.

ii.
```python
if stop <= start:
    continue
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

iii. The AI documented these edge cases in CONVERSION_NOTES Steps 4, 10, and noted: "Trials span trial_start to teleport onset, excluding teleport frames" and "Per-trial constants are repeated across timepoints."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the full deconvolved activity matrix from each NWB file via h5py. For large multi-plane sessions (e.g., m18 with 2341 neurons), this takes ~2.5 seconds. The full conversion of 152 sessions takes approximately 4-7 minutes total.

ii.
```python
deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
```

iii. The AI benchmarked conversion times: small sessions ~1s, large multi-plane sessions ~2.5s, with an estimated total of 4-7 minutes for all 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops could be vectorized: (1) `reconstruct_trials` uses a pointer-based loop to match trial starts with teleports; this could use `np.searchsorted`. (2) `reward_outcomes_from_timestamps` uses a similar pointer-based loop to check reward events within trial windows; this could also use `np.searchsorted`. (3) The per-trial processing loop in `load_session` iterates over each trial for slicing and discretization, though vectorizing this would require handling variable-length trials.

ii.
```python
# reconstruct_trials - could use np.searchsorted
for start in starts:
    while tp_ptr < len(teleports) and teleports[tp_ptr] <= start:
        tp_ptr += 1
    ...

# reward_outcomes_from_timestamps - could use np.searchsorted
for trial_idx, (start, stop) in enumerate(trials):
    while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
        reward_ptr += 1
    ...
```

iii. The AI noted these as relatively minor bottlenecks compared to the I/O-dominated NWB reading.

## 13-c. What processing does the code repeat multiple times?

i. The code does not significantly repeat processing. Each session is processed once. The only repeated computation is that `zone_for_trial` is called per trial (trivial cost). The scene identifier is parsed once per session. The deconvolved data is loaded once and sliced per trial.

ii. No significant code repetition observed.

iii. The AI designed the code to process sessions sequentially with no redundant reloading.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads the `environment` timeseries from NWB (`behavior["environment/data"][:]`) but does not use it for any computation - environment is instead derived from the scene identifier. Additionally, the code stores neural data as float16, which requires the downstream decoder to cast back to float32 for training.

ii.
```python
environment = behavior["environment/data"][:].astype(np.float32)  # Read but unused
```

iii. The AI read the environment timeseries during development but ultimately derived environment from the scene identifier for reliability. The unused read remains in the code.
