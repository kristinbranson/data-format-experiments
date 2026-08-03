# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `data/` directory using `h5py` (not `pynwb`). It discovers files by globbing `data/sub-*/sub-*_behavior+ophys.nwb`, sorted by path. Each NWB file corresponds to one session. All 152 NWB files across 11 subjects are loaded.

ii.
```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```

Loading with h5py:
```python
with h5py.File(path, "r") as handle:
    identifier = decode_h5_scalar(handle["identifier"])
    subject = decode_h5_scalar(handle["general/subject/subject_id"])
    ...
    behavior = handle["processing/behavior/BehavioralTimeSeries"]
    position = behavior["position/data"][:].astype(np.float32)
    ...
```

iii. The AI chose `h5py` for direct HDF5 access instead of `pynwb` for speed. The CONVERSION_NOTES.md documents this as a deliberate performance optimization: "Uses direct HDF5 access with h5py instead of slower NWB object loading."

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading the `general/subject/subject_id` field from each NWB file. A dynamic `subject_to_idx` dictionary maps subject IDs to indices as sessions are processed. Subjects are added to the list as new ones are encountered.

ii.
```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
```

iii. The AI reads subject identity from NWB metadata rather than parsing directory names. This is functionally equivalent since the NWB files are organized by subject directory.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from the NWB file's `general/session_id` field. All 152 NWB files are processed.

ii.
```python
session_id = decode_h5_scalar(handle["general/session_id"])
...
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(path=path, ...)
```

iii. The AI uses file-level session identification from NWB metadata, which is consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from `trial_start` and `teleport` behavioral time series. For each `trial_start` impulse (>0.5), the next `teleport` impulse (>0.5) after it is found to define the trial end. Trial boundaries are `(start, stop)` tuples where `stop` is exclusive.

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

iii. The CONVERSION_NOTES.md documents: "Reconstructs trials from trial_start impulse to teleport impulse." The AI chose to match each trial_start to the immediately following teleport event using a pointer-based approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on two criteria: (1) trials with fewer than 2 timepoints are skipped, and (2) trials with bad lick sensor data are removed. Bad lick trials are identified when >35% of frames have lick values >2, following the reference code's `LICK_ERROR_FRAC = 0.35` threshold.

ii.
```python
LICK_ERROR_FRAC = 0.35  # Reference code threshold in glmUtils.get_timeseries_data.

def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC

# In load_session:
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
```

Also, sessions with fewer than 2 usable trials after QC are skipped:
```python
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The AI identified the lick QC rule from the reference code (`glmUtils.get_timeseries_data`) and applied it. The CONVERSION_NOTES.md notes: "81 out of 12,376 trials removed across 11 switch mice" from the paper, and the converted data dropped 70 bad-lick trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field in `processing/ophys/Deconvolved/` of the NWB files.

ii.
```python
deconv_group = handle["processing/ophys/Deconvolved"]
plane_keys = sorted(deconv_group.keys(), key=lambda key: int(key.replace("plane", "")))
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    # multi-plane handling
    ...
```

iii. The CONVERSION_NOTES.md documents: "Uses NWB Deconvolved/plane0/data as neural activity" and "Candidate equivalent of reference sess.timeseries['events']."

## 2-b. How is the `neural` data processed?

i. For single-plane sessions, deconvolved data is directly indexed by curated ROI indices. For multi-plane sessions, data from each plane is loaded and assembled into a full matrix using `planeIdx` to assign columns, then curated ROIs are selected. Data is stored as `float16`.

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
    n_frames = deconv_group[plane_keys[0]]["data"].shape[0]
    n_rois = plane_idx.shape[0]
    all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
    for plane_key in plane_keys:
        plane_number = int(plane_key.replace("plane", ""))
        cols = np.flatnonzero(plane_idx == plane_number)
        plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
        all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]
```

iii. The AI documented the multi-plane handling fix: "Multi-plane ROI loading bug: Initial converter assumed a single plane0 response matrix and failed on m17/m18. Fixed by reconstructing a full session matrix from plane0 and plane1 using planeIdx."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered using the `iscell` column from the `PlaneSegmentation` table. Only ROIs where `iscell[:,0] > 0.5` (i.e., `iscell[:,0] == 1`) are retained.

ii.
```python
iscell = segmentation["iscell"][:]
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
```

iii. The AI documented in CONVERSION_NOTES.md: "Initial neuron filter = iscell[:,0] == 1: This matches Suite2p manual curation."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing the deconvolved matrix from the trial start frame to the teleport frame. Since neural and behavioral data share the same frame-based time axis in the NWB files, no additional temporal alignment is needed.

ii.
```python
neural_trial = deconvolved[start:stop].T
```

iii. The CONVERSION_NOTES.md documents: "Temporal alignment: converter uses trial_start -> teleport, matching the reference in-trial slicing logic."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5 Hz (~64.5 ms per frame). No temporal rebinning is applied. The time bin size is hardcoded as a constant.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The AI documented: "Use native imaging-frame resolution: The reference data are sampled at ~15.5 Hz and behavior is already synchronized to this grid. No rebinning in time unless a later validation forces it."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavioral time series.

ii.
```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The timestamps are from the position behavioral time series, which shares the same time base as other behavioral streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from all timestamps within the trial to get time relative to trial start.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
...
input_trial = np.vstack([
    time_trial.astype(np.float32),
    ...
])
```

iii. Straightforward subtraction of the first timestamp in the trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame-based indexing in the NWB file. Both are sliced using the same `start:stop` indices, so they are inherently aligned.

ii.
```python
pos_trial = position[start:stop]
...
neural_trial = deconvolved[start:stop].T
time_trial = position_t[start:stop] - position_t[start]
```

iii. The AI verified alignment through sanity checks: "Input checks for the same trials: reconstructed time, environment, trial number, and previous reward outcome matched converted inputs exactly."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the NWB `identifier` field, which contains a scene string like `Env1_LocationA` or `Env1_LocationA_to_B`. The environment (Env1/Env2) is parsed from this string.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```

```python
def parse_scene(identifier: str) -> dict:
    scene = identifier.rstrip("/").split("/")[-1]
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    if match:
        env, zone = match.groups()
        return {"scene": scene, "pre_env": env, "post_env": env, ...}
    ...
```

iii. The AI chose to parse environment from the scene identifier rather than from the `environment` behavioral time series. The CONVERSION_NOTES.md notes: "Derives environment and reward-zone identity from the NWB identifier scene string and the 30-trial switch rule."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string is parsed with regex to extract the environment name (Env1 or Env2). For switch sessions, the environment may change after trial 30. The environment is encoded as 0 (Env1) or 1 (Env2) and repeated across all timepoints in the trial.

ii.
```python
ENV_TO_INT = {"Env1": 0, "Env2": 1}
...
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
...
env_code = float(ENV_TO_INT[env_name])
input_trial = np.vstack([
    ...
    np.full(n_time, env_code, dtype=np.float32),
    ...
])
```

iii. The scene parsing handles three formats: simple (`Env1_LocationA`), within-env switch (`Env1_LocationA_to_B`), and cross-env switch (`Env1_A_to_Env2_C`).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter `trial_idx` during trial iteration.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    input_trial = np.vstack([
        ...
        np.full(n_time, float(trial_idx), dtype=np.float32),
        ...
    ])
```

iii. The trial number is the 0-based index of the trial within the session, derived from the enumeration of reconstructed trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop index. The value is constant across all timepoints within a trial. Note that the trial index counts all reconstructed trials, including those later dropped by QC, so the trial number in the output may skip values.

ii.
```python
np.full(n_time, float(trial_idx), dtype=np.float32)
```

iii. Straightforward assignment from the enumeration index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` in the behavioral time series. Reward timestamps are compared to trial time boundaries to determine per-trial reward outcomes.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
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

iii. The AI uses a pointer-based approach to match reward timestamps to trial time windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's outcome is used. For the first trial (index 0), the value defaults to 0. The value is constant across all timepoints within a trial.

ii.
```python
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
...
input_trial = np.vstack([
    ...
    np.full(n_time, float(prev_reward), dtype=np.float32),
])
```

iii. The AI correctly checks `trial_idx > 0` for the first trial edge case.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone coordinates. The reward zone identity (A, B, or C) is determined from the NWB session identifier scene string and the 30-trial switch rule.

ii.
```python
ZONE_COORDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The reward zone coordinates match the paper's definitions: A=80-130, B=200-250, C=320-370 cm.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The signed distance from position to the nearest edge of the reward zone is computed: negative before the zone, zero inside the zone, positive after the zone. This distance is then discretized into 7 bins.

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

iii. The distance computation is vectorized and matches the expected signed-distance logic from the paper.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit comparisons:
- 0: < -50 cm
- 1: -50 to -10 cm (inclusive lower, exclusive upper)
- 2: -10 to < 0 cm
- 3: exactly 0 cm (in zone)
- 4: > 0 to 10 cm (inclusive upper)
- 5: 10 to 50 cm (inclusive upper)
- 6: > 50 cm (default)

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

iii. The bin edges match the instruction specification. The AI uses explicit boolean indexing rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame-based indexing as neural data. Both use the same `start:stop` slice.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. Inherently aligned because both share the same NWB frame index.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. Direct use of the raw position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the track range [0, 450) cm, then discretized into 5 equal-sized bins of 90 cm each using `np.linspace(0, 450, 6)` which gives edges `[0, 90, 180, 270, 360, 450]`.

ii.
```python
TRACK_START_CM = 0.0
TRACK_END_CM = 450.0

def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The AI chose to use the track length (0-450 cm) divided into 5 equal 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is clipped to [0, 450) and digitized using edges from `np.linspace(0, 450, 6)` = `[0, 90, 180, 270, 360, 450]`. The internal edges used for `np.digitize` are `[90, 180, 270, 360]`, producing 5 bins:
- 0: [0, 90)
- 1: [90, 180)
- 2: [180, 270)
- 3: [270, 360)
- 4: [360, 450]

ii.
```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The AI uses 90 cm equal-width bins on the track range [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-based indexing as neural data.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. Inherently aligned through shared frame index.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
lick_bin = binarize_licks(lick_trial)
```

iii. Direct use of the raw lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are rounded to the nearest integer, then clipped to [0, 1].

ii.
```python
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. The AI uses rounding followed by clipping, rather than a simple threshold of >0.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-based indexing as neural data.

ii.
```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. Inherently aligned through shared frame index.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB session `identifier` field, which contains a scene string encoding the reward zone identity (A, B, or C) and any switch information.

ii.
```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
output_trial = np.vstack([
    ...
    np.full(n_time, zone_code, dtype=np.int16),
    ...
])
```

iii. The AI uses the NWB identifier to parse reward zone identity, avoiding reliance on the noisy `reward_zone` behavioral time series. Switch sessions change zone after trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regex to extract pre/post zone identities. For switch sessions (`trial_idx >= 30`), the post-switch zone is used; otherwise, the pre-switch zone. Zone labels are mapped: A=0, B=1, C=2.

ii.
```python
SWITCH_TRIAL = 30
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}

def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```

iii. The switch-at-30-trials rule comes from the paper: "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. Reward events are identified by their timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window (between `start_t` and `stop_t`). The result is binary (0/1) and constant across all timepoints in the trial.

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

```python
output_trial = np.vstack([
    ...
    np.full(n_time, reward_code, dtype=np.int16),
])
```

iii. A pointer-based approach efficiently scans sorted reward timestamps against sorted trial boundaries.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Short trials**: Trials with fewer than 2 timepoints are skipped.
- **Bad lick trials**: Trials where >35% of frames have lick values >2 are excluded (matching the reference code's QC criterion).
- **Sessions with too few trials**: Sessions with fewer than 2 usable trials after QC are skipped entirely.
- **Multi-plane sessions**: Multi-plane ROI data is reconstructed using `planeIdx` to handle sessions with more than one imaging plane.
- **Neural/behavior alignment**: Both use the same frame indices from the NWB file, so misalignment is handled implicitly.

ii.
```python
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

iii. The AI documented these checks in CONVERSION_NOTES.md under multiple steps.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files with `h5py` and reading the large deconvolved data arrays. The AI reported ~1.0-2.5 seconds per session, with the full conversion taking ~4-7 minutes for 152 sessions.

ii. N/A

iii. The CONVERSION_NOTES.md documents timing estimates: "Large multi-plane benchmark (m18 ses-03): ~2.49 s/session. Conservative upper bound ~6.3 min for 152 sessions."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `load_session` iterates over each trial sequentially. Operations like distance computation and discretization are already vectorized within each trial, but the trial-level loop could potentially be avoided by working with padded/masked arrays.

ii. N/A

iii. The per-trial loop is a natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The AI's code processes each NWB file only once (no separate survey step). All data loading and processing happens in the single `load_session` function. This is more efficient than the reference which does a survey pass followed by a conversion pass.

ii. N/A

iii. Using `h5py` directly and processing in a single pass avoids the double-loading that would occur with a separate survey step.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `speed` output is computed but is not required by the decoder task specification. The instructions list "Speed, time-varying" as a decoder output, and the AI includes it. However, speed is used in the reference code primarily as a filter (excluding <2 cm/s samples), not necessarily as a decoded output. The AI does not apply speed-based sample exclusion.

ii.
```python
speed_bin = discretize_speed(speed_trial)
```

iii. The instructions explicitly list speed as a decoder output, so including it is consistent with the task specification.
