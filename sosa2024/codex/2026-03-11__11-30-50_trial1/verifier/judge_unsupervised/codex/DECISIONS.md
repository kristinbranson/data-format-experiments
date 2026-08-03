# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all sessions by globbing every NWB file under `data/sub-*/sub-*_behavior+ophys.nwb`, sorting the paths, and then opening each file with `h5py`. Within each file it reads behavior streams from `processing/behavior/BehavioralTimeSeries`, ophys data from `processing/ophys`, and subject/session metadata from NWB header fields.

ii. `convert_data.py`:

```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files

def load_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        identifier = decode_h5_scalar(handle["identifier"])
        subject = decode_h5_scalar(handle["general/subject/subject_id"])
        session_id = decode_h5_scalar(handle["general/session_id"])
        ...
        behavior = handle["processing/behavior/BehavioralTimeSeries"]
        ...
        segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        ...
        deconv_group = handle["processing/ophys/Deconvolved"]
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the agent justified this by treating the NWB files as an exported `sess`-equivalent of the reference pipeline and by noting that direct NWB reads were the practical way to process the full 152-session release.

## 1-b. How are the data split into subjects (mice)?

i. The agent treats the NWB `general/subject/subject_id` field as the authoritative mouse identifier. It builds a unique `subjects` list in first-seen session order and a per-session `subject_idx` vector.

ii. `convert_data.py`:

```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
subject_to_idx = {}
subject_idx = []
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
...
subject_idx.append(subject_to_idx[subject])
...
data["subject_idx"] = np.asarray(subject_idx, dtype=np.int16)
```

iii. Step 2 of the notes records that the release is organized one folder per subject and one NWB file per session. The trajectory also shows the agent explicitly counted 11 subjects and used the NWB metadata rather than inferring subject identity from filenames alone.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are processed in sorted file order, and the resulting session-level trial lists are appended one session at a time into `data['neural']`, `data['input']`, and `data['output']`.

ii. `convert_data.py`:

```python
files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for session_number, path in enumerate(session_files):
    session_data, stats = load_session(
        path=path,
        show_processing=show_processing and session_number < 2,
    )
    ...
    data["neural"].append(session_data["neural_trials"])
    data["input"].append(session_data["input_trials"])
    data["output"].append(session_data["output_trials"])
```

iii. In Step 2 the agent documented that each subject folder contains one NWB per session. In the trajectory, it repeatedly refers to “152 session files” and uses that as the unit of conversion, consistent with the reference code’s per-session processing.

## 1-d. How are the data split into trials?

i. The agent reconstructs trials from framewise behavioral pulses: every `trial_start > 0.5` is paired to the next later `teleport > 0.5`, and the per-trial data slice is `[start:stop]`, excluding the teleport sample itself. Those trial boundaries are then used for neural, input, and output extraction.

ii. `convert_data.py`:

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
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    pos_trial = position[start:stop]
    ...
    neural_trial = deconvolved[start:stop].T
```

iii. Step 4 and Step 5 in the notes say the agent chose this specifically to mirror the reference `trial_start_inds`/`teleport_inds` semantics from `glmUtils.get_timeseries_data`, while adapting from the NWB’s framewise signals instead of a prebuilt `sess` object.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters out trials with fewer than 2 position samples and drops entire trials when the lick trace appears corrupted, defined as more than 35% of samples having `lick > 2`. It also skips sessions with fewer than 2 usable trials after filtering.

ii. `convert_data.py`:

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
...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The notes explicitly tie the `0.35` threshold to `glmUtils.get_timeseries_data`. The trajectory shows the agent deciding to drop bad-lick trials entirely because `lick` was required as a decoder output and it wanted to avoid NaN labels in the final trial-structured dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes directly from the NWB `processing/ophys/Deconvolved/plane*/data` arrays, with ROIs filtered by the Suite2p-style `iscell` table from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii. `convert_data.py`:

```python
segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconv_group = handle["processing/ophys/Deconvolved"]
plane_keys = sorted(deconv_group.keys(), key=lambda key: int(key.replace("plane", "")))
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    ...
    deconvolved = all_deconvolved[:, curated_idx]
```

iii. In Step 5, the agent explicitly chose “NWB `Deconvolved` activity” as the neural signal because the reference decoder uses deconvolved `events`, and the trajectory shows it treating the exported NWB `Deconvolved` arrays as the closest available equivalent.

## 2-b. How is the `neural` data processed?

i. The agent does not recompute dF/F or deconvolution. Instead it selects curated ROIs, optionally reconstructs a multi-plane ROI matrix, casts the signal to `float16`, slices each trial from `start` to `stop`, and transposes each trial to `(n_neurons, n_timepoints)`.

ii. `convert_data.py`:

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

iii. The notes justify this as a pragmatic substitution for the reference `events` timeseries: Step 4 says to use NWB `Deconvolved` unless later checks showed a mismatch, and the trajectory confirms the agent consciously chose not to recompute dF/F from `Fluorescence` and `Neuropil`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neuron-level quality filter in the converter is `iscell[:, 0] > 0.5`, which keeps manually curated Suite2p ROIs. The code does not implement the reference paper’s additional putative-interneuron exclusion and does not apply the reference samplewise speed/lick masking to the neural traces themselves.

ii. `convert_data.py`:

```python
iscell = segmentation["iscell"][:]
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
...
deconvolved = all_deconvolved[:, curated_idx]
```

iii. Step 5 of the notes says “Initial neuron filter = `iscell[:,0] == 1`” and describes the extra interneuron filter as a planned consistency check rather than a conversion rule. The trajectory later says the agent believed the remaining count mismatch was likely a release-version issue rather than a missing converter step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start. Each trial uses the same `[start:stop]` frame range defined from the `trial_start` and `teleport` signals, and metadata states the temporal alignment event is `"trial start"` with `off_start = 0.0`.

ii. `convert_data.py`:

```python
trials = reconstruct_trials(trial_start, teleport)
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trial = deconvolved[start:stop].T
...
"metadata": {
    ...
    "temporal_alignment_event": "trial start",
    "off_start": 0.0,
    "off_end": None,
```

iii. The notes repeatedly emphasize that the target decoder requested trial-start alignment, and the trajectory explicitly states: “`trial_start` is the correct aligner, and trials should run from `trial_start` up to but excluding the first `teleport` frame.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native imaging-frame resolution of the NWB data: `FRAME_RATE_HZ = 15.5078125`, corresponding to `TIME_BIN_MS = 1000 / FRAME_RATE_HZ`, about 64.48 ms per sample. No temporal rebinning is applied.

ii. `convert_data.py`:

```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "frame_rate_hz": FRAME_RATE_HZ,
```

iii. Step 3 and Step 5 of the notes cite the paper’s “~15.5 Hz” sampling and state the decision “Use native imaging-frame resolution; no rebinning in time unless a later validation forces it.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the behavior timestamp vector attached to the `position` timeseries, specifically `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii. `convert_data.py`:

```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. The notes say the behavior streams are already aligned to imaging frames and that trial-relative time should be constructed directly from those timestamps to match the requested decoder input.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent slices the frame timestamps over the trial interval and subtracts the first frame timestamp, so every trial begins at 0.0 seconds and then increases at the frame rate.

ii. `convert_data.py`:

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

iii. Step 5 of the notes describes this as a direct adaptation to the decoder task: use trial-aligned time in seconds, continuous and time-varying, while preserving the imaging-frame sampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector and the neural matrix use the exact same per-trial `[start:stop]` frame slice, so they have the same number of timepoints and share framewise alignment.

ii. `convert_data.py`:

```python
pos_trial = position[start:stop]
time_trial = position_t[start:stop] - position_t[start]
...
neural_trial = deconvolved[start:stop].T
...
n_time = neural_trial.shape[1]
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        ...
    ]
)
```

iii. The notes and trajectory both frame the entire converter around using one common trial frame grid for all signals, with behavior already synchronized to the imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In the implemented code, environment type is derived from the NWB `identifier` scene string, not from the raw `environment/data` stream that is loaded earlier. The parser extracts pre-switch and post-switch environment labels from strings like `Env1_LocationA`, `Env1_LocationA_to_C`, or `Env1_B_to_Env2_C`.

ii. `convert_data.py`:

```python
identifier = decode_h5_scalar(handle["identifier"])
...
environment = behavior["environment/data"][:].astype(np.float32)
...
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
```

iii. Step 5 of the notes says the agent planned to cross-check native `environment` values against the scene identifier, but the trajectory shows it ultimately chose the scene parser and switch rule as the source of per-trial environment labels.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses the session identifier into pre-switch and post-switch states, applies a hard switch at trial 30 for switch sessions, maps `Env1 -> 0` and `Env2 -> 1`, and repeats the per-trial environment code across all timepoints in the trial.

ii. `convert_data.py`:

```python
ENV_TO_INT = {"Env1": 0, "Env2": 1}
SWITCH_TRIAL = 30
...
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])
...
np.full(n_time, env_code, dtype=np.float32),
```

iii. The notes justify this by saying reward-zone and environment identity should be inferred from the scene metadata and the paper’s “switch after 30 trials” rule, rather than from the numeric `reward_zone` samples.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the raw `trial number/data` series. It is derived from the reconstructed chronological trial index produced by pairing `trial_start` and `teleport` events.

ii. `convert_data.py`:

```python
trial_start = behavior["trial_start/data"][:].astype(np.float32)
teleport = behavior["teleport/data"][:].astype(np.float32)
...
trials = reconstruct_trials(trial_start, teleport)
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. Step 5 of the notes explicitly says “Use reconstructed trial order from `trial_start` events, not raw `trial number` during teleport,” matching the reference code’s repeated loop index rather than the raw framewise counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent enumerates the reconstructed trials starting from 0 and repeats that scalar index at every timepoint within the trial.

ii. `convert_data.py`:

```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    input_trial = np.vstack(
        [
            ...,
            np.full(n_time, float(trial_idx), dtype=np.float32),
            ...
        ]
    )
```

iii. The notes justify this as aligning with the reference logic in `glmUtils.get_timeseries_data`, where `trial_ids` are assigned by trial-loop index for every in-trial sample.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from raw reward timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, together with the reconstructed trial start and stop times from `position/timestamps` and the `trial_start`/`teleport`-defined trial intervals.

ii. `convert_data.py`:

```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
trials = reconstruct_trials(trial_start, teleport)
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. Step 5 of the notes says the agent wanted “previous trial reward outcome” to be a per-trial binary variable consistent with the decoder specification, and it therefore built trial outcomes from actual reward delivery events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes a binary reward outcome for every trial, then assigns each trial the outcome of the immediately preceding trial; the first trial gets a default 0. That scalar is repeated across the whole trial.

ii. `convert_data.py`:

```python
def reward_outcomes_from_timestamps(reward_times: np.ndarray, trial_times: np.ndarray, trials: list[tuple[int, int]]) -> np.ndarray:
    outcomes = np.zeros(len(trials), dtype=np.int8)
    ...
    for trial_idx, (start, stop) in enumerate(trials):
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        ...
        outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes
...
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
...
np.full(n_time, float(prev_reward), dtype=np.float32),
```

iii. The notes and trajectory show no alternative considered here; the agent treated this as a straightforward task-specific derived variable based on per-trial reward delivery.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the raw `position/data` timeseries plus the active reward-zone coordinates inferred from the scene identifier and switch status. The raw framewise `reward_zone/data` values are not used for the categorical distance target.

ii. `convert_data.py`:

```python
position = behavior["position/data"][:].astype(np.float32)
...
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
...
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. Step 4 and Step 5 of the notes say the agent concluded the NWB `reward_zone` samples are occupancy-like codes rather than A/B/C labels, so it chose the paper’s fixed zone coordinates and scene metadata instead.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes a signed distance to the nearest point in the active reward zone: negative before the zone, zero inside the zone, positive after the zone. It does this frame by frame within each trial.

ii. `convert_data.py`:

```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
    ...
```

iii. The notes say this was chosen to satisfy the decoder task’s “distance to any location in the reward zone” definition while remaining anchored to the paper’s reward-zone coordinates.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is binned into 7 categories using the exact thresholds from the task instructions: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii. `convert_data.py`:

```python
bins = np.full(distance.shape, 6, dtype=np.int16)
bins[distance < -50.0] = 0
bins[(distance >= -50.0) & (distance < -10.0)] = 1
bins[(distance >= -10.0) & (distance < 0.0)] = 2
bins[distance == 0.0] = 3
bins[(distance > 0.0) & (distance <= 10.0)] = 4
bins[(distance > 10.0) & (distance <= 50.0)] = 5
```

iii. The notes describe this as one of the explicit task-driven discretizations where the converter should follow the decoder specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same trial slice `[start:stop]` used for `neural_trial`, so each framewise distance bin lines up one-to-one with a neural sample.

ii. `convert_data.py`:

```python
pos_trial = position[start:stop]
...
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
...
output_trial = np.vstack(
    [
        dist_bin,
        ...
    ]
)
```

iii. The trajectory repeatedly states that all time-varying variables would share the same reconstructed imaging-frame trial grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the raw `processing/behavior/BehavioralTimeSeries/position/data` stream.

ii. `convert_data.py`:

```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
...
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes identify raw position as one of the core decoder-relevant behavior variables already aligned to imaging frames in the NWB export.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position to the valid track interval `[0, 450)` and then discretizes it into 5 equal-width bins across the corridor.

ii. `convert_data.py`:

```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. Step 5 of the notes says absolute position should use the “track-only interval” and the paper’s 450 cm corridor length, with no teleport-specific bin because teleport frames are excluded.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm track is divided into 5 equal-sized bins by `np.linspace(..., 6)` and `np.digitize`, producing integer classes 0 through 4.

ii. `convert_data.py`:

```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. The notes justify this as following the decoder task specification, which asked for 5 equal-sized absolute-position bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Like the other time-varying outputs, absolute-position bins are computed from the exact trial slice used for `neural_trial`, so the two are frame aligned.

ii. `convert_data.py`:

```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The agent’s Step 5 plan was to keep all decoder variables on the common in-trial imaging-frame axis.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The lick output is derived from the raw `processing/behavior/BehavioralTimeSeries/lick/data` signal.

ii. `convert_data.py`:

```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
...
lick_bin = binarize_licks(lick_trial)
```

iii. The notes identify `sess.timeseries['licks']` / NWB `lick` as the source signal and specifically mention the reference lick-sensor QC threshold.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first excludes entire trials that fail the lick-sensor corruption rule, then converts the remaining per-frame lick values into binary labels by rounding to the nearest integer and clipping to `[0, 1]`.

ii. `convert_data.py`:

```python
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
...
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. The trajectory says the agent chose this because the decoder required a binary lick output and it did not want to carry NaNs from the reference lick-sensor handling into the final dataset.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick values are sliced over the same `[start:stop]` trial interval as the neural data, then binarized frame by frame, so they remain one-to-one aligned with neural samples.

ii. `convert_data.py`:

```python
lick_trial = lick[start:stop]
neural_trial = deconvolved[start:stop].T
lick_bin = binarize_licks(lick_trial)
```

iii. The notes treat lick like the other behavior streams already synchronized to imaging frames in the NWB file.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB `identifier` scene string, parsed into reward-zone labels before and after a possible switch, then resolved per trial using the hard-coded switch trial.

ii. `convert_data.py`:

```python
identifier = decode_h5_scalar(handle["identifier"])
scene_info = parse_scene(identifier)
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_code = int(ZONE_TO_INT[zone_name])
```

iii. Step 4 and Step 5 of the notes explicitly justify using the scene identifier because the numeric NWB `reward_zone` stream was interpreted as occupancy-like rather than as the A/B/C identity needed for this variable.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses one of several scene-string formats, applies the trial-30 switch rule where needed, maps `A/B/C` to `0/1/2`, and repeats that per-trial class across all frames in the trial.

ii. `convert_data.py`:

```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
...
def parse_scene(identifier: str) -> dict:
    scene = identifier.rstrip("/").split("/")[-1]
    ...
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
...
np.full(n_time, zone_code, dtype=np.int16),
```

iii. The notes say this was intended to mirror `behavior.get_reward_zones`, which also derives reward-zone identity from the scene metadata and switch day structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from raw reward delivery timestamps (`Reward/timestamps`) together with the trial boundaries and the behavior timestamp axis used to determine whether a reward occurred during each reconstructed trial.

ii. `convert_data.py`:

```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
...
reward_code = int(reward_outcomes[trial_idx])
```

iii. The notes map this output to actual delivered reward events and explicitly distinguish it from reward-zone identity.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The agent marks a trial as rewarded if any reward timestamp falls between the trial start time and trial stop time, otherwise unrewarded. That per-trial binary label is then repeated across all timepoints in the trial.

ii. `convert_data.py`:

```python
def reward_outcomes_from_timestamps(reward_times: np.ndarray, trial_times: np.ndarray, trials: list[tuple[int, int]]) -> np.ndarray:
    outcomes = np.zeros(len(trials), dtype=np.int8)
    ...
    outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes
...
np.full(n_time, reward_code, dtype=np.int16),
```

iii. The notes justify this as matching the decoder task’s per-trial reward-outcome variable and using “actual delivered reward.”

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mostly handles irregularities by dropping or skipping data rather than imputing it. It drops empty lick trials and bad-lick trials, ignores any `trial_start` without a following teleport, skips degenerate trials with fewer than 2 samples, and skips sessions left with fewer than 2 usable trials. It also checks multi-plane ROI counts and raises an error if `planeIdx` and plane-specific deconvolved matrices disagree.

ii. `convert_data.py`:

```python
if tp_ptr >= len(teleports):
    break
...
if lick_trial.size == 0:
    return True
...
if pos_trial.size < 2:
    continue
...
if plane_data.shape[1] != cols.size:
    raise ValueError(
        f"{path.name}: {plane_key} has {plane_data.shape[1]} ROIs but planeIdx maps {cols.size}"
    )
...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue
```

iii. The notes and trajectory show the agent preferred exclusion over imputation because the decoder validator expected consistent per-trial matrices without NaN labels, especially for `lick`.

## 13-a. What are the most time-consuming steps of the code?

i. The main cost is session-by-session NWB I/O and full-session neural data assembly. The heaviest operations are loading `Deconvolved` arrays, reconstructing multi-plane ROI matrices, and then slicing/stacking trial matrices for every session. Plot generation is an additional cost when `--show-processing` is enabled.

ii. `convert_data.py`:

```python
for session_number, path in enumerate(session_files):
    session_t0 = time.perf_counter()
    session_data, stats = load_session(...)
    elapsed = time.perf_counter() - session_t0
    print(...)
...
all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
for plane_key in plane_keys:
    ...
    all_deconvolved[:, cols] = plane_data
...
for trial_idx, (start, stop) in enumerate(trials):
    ...
    neural_trials.append(neural_trial)
```

iii. The trajectory explicitly mentions that full-file aggregation over 152 NWBs was expensive enough that the agent switched from `pynwb` to `h5py` for speed.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `reconstruct_trials`, `reward_outcomes_from_timestamps`, and the main trial-building loop could be partially vectorized. The multi-plane reconstruction loop could also be reduced if plane data were concatenated in a more direct way. Repeated per-trial `np.full(...)` creation for constant inputs/outputs is also vectorizable.

ii. `convert_data.py`:

```python
for start in starts:
    ...

for trial_idx, (start, stop) in enumerate(trials):
    ...

for plane_key in plane_keys:
    ...
```

iii. There is no explicit optimization plan in the notes beyond the switch to `h5py`, so this is mostly an evaluation of the written code rather than a stated agent rationale.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly resolves per-trial scene-derived metadata (`zone_for_trial`), repeatedly slices the same trial interval across multiple arrays (`position`, `speed`, `lick`, `timestamps`, `deconvolved`), and repeatedly allocates full-length constant vectors for environment, trial number, previous reward, reward-zone location, and reward outcome.

ii. `convert_data.py`:

```python
for trial_idx, (start, stop) in enumerate(trials):
    pos_trial = position[start:stop]
    speed_trial = speed[start:stop]
    lick_trial = lick[start:stop]
    time_trial = position_t[start:stop] - position_t[start]
    ...
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
```

iii. The notes do not present these repetitions as deliberate scientific choices; they are implementation consequences of building time-varying trial matrices.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unused work is that the raw `environment/data` stream is loaded but never used. The converter also spends time constructing verbose `stats`, metadata session summaries, timing prints, and optional processing plots that are not needed by the downstream decoder itself.

ii. `convert_data.py`:

```python
environment = behavior["environment/data"][:].astype(np.float32)
...
stats = {
    "path": str(path),
    "subject": subject,
    "session_id": session_id,
    ...
}
...
if show_processing and neural_trials:
    make_processing_plot(...)
```

iii. The trajectory shows the agent loaded the `environment` stream while exploring alternative mappings, but the final code uses scene parsing instead. The extra stats and plots were justified as sanity-check tooling rather than decoder inputs.
