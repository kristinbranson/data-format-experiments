# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing every `sub-*/sub-*_behavior+ophys.nwb` file under `/app/data`, sorting them by subject/session parsed from the pathname, and then opening each NWB file directly with `h5py`. Trial data are produced inside `build_session()` after the file is opened.

ii.
```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session

def build_full_dataset(data_dir: Path):
    session_records = []
    for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
        print(f"Converting {path.relative_to(data_dir.parent)}")
        record = build_session(path)
        session_records.append(record)
```
```python
def build_session(path: Path):
    with h5py.File(path, "r") as f:
        ...
```

iii. `CONVERSION_NOTES.md` says the conversion follows the public NWB export while mirroring the paper/code logic. The notes also justify this as complete coverage by reporting `11` converted mice and `152` converted sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified per file from the NWB metadata field `general/subject/subject_id`, then deduplicated in file order to build `subjects` and `subject_idx`.

ii.
```python
with h5py.File(path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"][()])
```
```python
subjects = []
subject_to_idx = {}
for record in session_records:
    subject = record["subject"]
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subjects)
        subjects.append(subject)
```

iii. The notes justify the subject split with manuscript-level sanity checks: `11` converted mice, matching the switch-task cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session ordering comes from sorting file paths by parsed subject ID and session number.

ii.
```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session
```
```python
for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
    record = build_session(path)
```

iii. The justification is implicit in the code structure and in the notes’ per-session reporting. The notes treat each converted NWB file as one imaging session and report per-session sanity statistics.

## 1-d. How are the data split into trials?

i. Trials are segmented from frames where `trial_start > 0`. For each start, the AI ends the trial at the last frame before the next start whose position remains on the 0 to 450 cm corridor. It does not use the `teleport` variable directly.

ii.
```python
def find_trial_segments(position: np.ndarray, trial_start: np.ndarray):
    starts = np.flatnonzero(trial_start > 0)
    next_starts = np.concatenate([starts[1:], [len(position)]])
    ends = []

    for start, next_start in zip(starts, next_starts):
        trial_pos = position[start:next_start]
        valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
        if len(valid) == 0:
            continue
        end = start + valid[-1] + 1
        ...
        ends.append(end)
```
```python
starts, ends = find_trial_segments(position, trial_start)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this choice: trials are aligned to `trial_start`, and only corridor frames in `[0, 450]` cm are kept because the AI considered that to better match the reference focus on the 450 cm track rather than the teleport period.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s explicit minimum-length trial filter. Instead, it drops candidate trial segments with no valid corridor frames, ignores zero-length segments, and raises an error if a session ends up with fewer than two trials.

ii.
```python
valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
if len(valid) == 0:
    continue
end = start + valid[-1] + 1
if end <= start:
    continue
```
```python
starts, ends = find_trial_segments(position, trial_start)
ntrials = len(starts)
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. The notes frame this as part of the trial-window definition: only corridor frames are retained and teleport-zone frames are excluded. No explicit justification for omitting a short-trial filter was recorded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final decoder neural signal is taken from `processing/ophys/Deconvolved`. Additional raw variables from `Fluorescence`, `Neuropil`, and `ImageSegmentation/PlaneSegmentation/iscell` are used only for cell filtering.

ii.
```python
plane_masks = get_curated_plane_masks(f)
...
events = load_curated_events(f, plane_masks, keep_mask, common_length)
```
```python
def load_curated_events(...):
    events = f["processing/ophys/Deconvolved"]
    ...
    plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` states that the decoder neural input uses the NWB `processing/ophys/Deconvolved` traces, and that fluorescence and neuropil are reconstructed only for the interneuron screen.

## 2-b. How is the `neural` data processed?

i. The AI concatenates curated cells across planes, crops all streams to a shared `common_length`, excludes putative interneurons, and then slices the deconvolved events into per-trial `(neurons, time)` matrices. It does not further smooth or bin the final neural signal.

ii.
```python
dense_lengths = [...]
for grp_name in ("Deconvolved", "Fluorescence", "Neuropil"):
    grp = f[f"processing/ophys/{grp_name}"]
    for plane_name in get_plane_names(grp):
        dense_lengths.append(grp[plane_name]["data"].shape[0])
common_length = min(dense_lengths)
```
```python
events = load_curated_events(f, plane_masks, keep_mask, common_length)
...
neural = events[start:end].T.astype(np.float32)
neural_trials.append(neural)
```

iii. The notes justify this as using the NWB deconvolved traces directly while matching the paper/code logic for curated-cell selection and extra interneuron exclusion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only manually curated ROIs (`iscell[:, 0] == 1`) and then applies an extra interneuron exclusion step: it reconstructs per-trial dF/F from fluorescence and neuropil and removes cells with `corr(dff, speed) > 0.5`.

ii.
```python
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
...
masks[plane_name] = mask
```
```python
trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
trial_dff = (trial_roi - baseline) / np.abs(baseline)
trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
...
all_is_int.append(corr > 0.5)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as matching the manuscript criterion and `dayData.py`, and reports that the removal fraction was small, consistent with the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start by segmenting each trial from `trial_start`, then taking the deconvolved event rows between each trial’s `start:end` indices.

ii.
```python
starts, ends = find_trial_segments(position, trial_start)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    neural = events[start:end].T.astype(np.float32)
    neural_trials.append(neural)
```

iii. The notes explicitly say trials are aligned to `trial_start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native frame resolution. The AI estimates the bin size from the median difference of the behavioral timestamps and stores it as milliseconds. No temporal rebinning is applied.

ii.
```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. The notes justify this with a timing sanity check: the median frame interval was `0.0644836` s (`64.4836` ms), matching the approximately `15.5 Hz` per-plane sampling described in the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives this input from the `position/timestamps` behavioral timestamp stream.

ii.
```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. No separate explicit justification was recorded beyond the general alignment logic. The code uses `position_ts` because all per-frame signals were handled on that shared timestamp grid after cropping.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, it subtracts the first timestamp in the trial from every timestamp in that trial.

ii.
```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The justification is implicit in the alignment requirement and the notes’ statement that trials are aligned to `trial_start`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `start:end` indices are used for timestamps and deconvolved events after all streams are cropped to a shared `common_length`.

ii.
```python
common_length = min(dense_lengths)
position_ts = position_ts[:common_length]
...
neural = events[start:end].T.astype(np.float32)
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The notes justify this with timing checks and with the explicit choice to align all trial streams to `trial_start`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The stored decoder input is derived from the session `identifier` string, specifically its scene name, not from the raw `environment` time series. The raw `environment` stream is only used as a validation check.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
```
```python
env_vals = environment[start:end]
env_vals = env_vals[env_vals >= 0]
behavior_env_by_trial.append(int(round(float(np.median(env_vals)))))
...
env_mismatch = int(np.sum(env_by_trial != np.asarray(behavior_env_by_trial, dtype=np.int64)))
```

iii. `CONVERSION_NOTES.md` justifies this by saying it reproduced the paper/code reward-zone schedule from scene names and verified that the parsed environment schedule matched the behavior stream on every trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses the scene name with regexes, maps `Env1 -> 0` and `Env2 -> 1`, applies a hard-coded switch split at trial `30` for switch sessions, and then fills a constant value across each trial’s timepoints.

ii.
```python
ENV_TO_IDX = {"Env1": 0, "Env2": 1}
CHANGE_TRIAL = 30
...
fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
```
```python
inputs = np.vstack([
    time_trial,
    np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
    ...
])
```

iii. The notes explicitly justify the `30`-trial split and special handling of day-8 environment-switch sessions as matching the manuscript/code conventions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the segmented-trial loop index, which itself comes from the `trial_start`-based trial segmentation.

ii.
```python
starts, ends = find_trial_segments(position, trial_start)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. The notes explicitly say `trial_number` is `0`-indexed within session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is applied beyond filling the per-trial loop index across all timepoints of that trial.

ii.
```python
np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` justifies this as matching the aligned behavior stream and being `0`-indexed within session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward-event timestamp stream `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, together with the per-trial time boundaries taken from `position_ts[start:end]`.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The notes justify reward outcome as derived from the sparse NWB reward timestamps within each trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, the AI computes a binary `reward_outcome` per trial by checking whether any reward timestamp falls inside the current trial. Then it assigns `previous_trial_outcome` from the previous imaged trial’s reward outcome, with the first trial set to `0`.

ii.
```python
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
...
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
```
```python
np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` explicitly says the first trial is set to `0` and that the previous-trial label uses the previous imaged trial in the same session.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The stored output is derived from the clipped per-trial position trace plus reward-zone bounds inferred from the scene name and session schedule. The raw `reward_zone` time series is used only to validate the inferred schedule, not to generate the label.

ii.
```python
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
zone_start, zone_end = zone_bounds[trial_idx]
outputs = np.vstack([
    discretize_distance_to_zone(pos_trial, zone_start, zone_end),
    ...
])
```
```python
zone_vals = reward_zone[start:end] > 0
if np.any(zone_vals):
    zone_pos = np.clip(position[start:end][zone_vals], 0.0, 450.0)
    center = float(np.mean(zone_pos))
    inferred = min(
        ZONE_BOUNDS_CM,
        key=lambda label: abs(center - np.mean(ZONE_BOUNDS_CM[label])),
    )
```

iii. The notes justify this by saying the AI reproduced the same reward-zone schedules from scene names as the manuscript/code and verified perfect agreement with observed reward-zone occupancy when it was present.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the active 50 cm reward zone: negative before the zone, `0` inside the zone, positive after the zone. It then discretizes that continuous distance.

ii.
```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
```

iii. `CONVERSION_NOTES.md` states this rule explicitly and says it matches the signed distance to the nearest point in the active reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI manually thresholds the signed distance into seven bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
bins[dist < -50.0] = 0
bins[(dist >= -50.0) & (dist < -10.0)] = 1
bins[(dist >= -10.0) & (dist < 0.0)] = 2
bins[dist == 0.0] = 3
bins[(dist > 0.0) & (dist <= 10.0)] = 4
bins[(dist > 10.0) & (dist <= 50.0)] = 5
bins[dist > 50.0] = 6
```

iii. The notes justify these bins as following the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by taking the same per-trial `start:end` slices from position as from the deconvolved event matrix.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    pos_trial = np.clip(position[start:end], 0.0, 450.0)
    neural = events[start:end].T.astype(np.float32)
```

iii. The notes justify this with the same trial-alignment logic used throughout the conversion.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the raw behavioral `position` time series, clipped to the 0 to 450 cm corridor before binning.

ii.
```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The notes justify this as focusing on the 450 cm track and excluding teleport frames.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position to `[0, 450]` cm and then places it into five equal 90 cm bins using `floor(position / 90)`, with values above the final bin clipped to bin `4`.

ii.
```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as “five equal 90 cm corridor bins.”

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholding is the same 90 cm corridor binning: `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm, encoded as integer categories `0` through `4`.

ii.
```python
clipped = np.clip(position_cm, 0.0, 450.0)
bins = np.floor(clipped / 90.0).astype(np.int64)
bins[bins > 4] = 4
```

iii. The notes justify this as matching the instruction for five equal-sized bins on the 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same `start:end` frame indices as the neural data for each trial.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
neural = events[start:end].T.astype(np.float32)
```

iii. The justification is the same trial-start alignment described in the notes.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says licks are binarized from the framewise lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI thresholds the lick trace at `> 0` to produce a binary no/yes time series.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The notes explicitly justify this as binarizing licks with `lick > 0`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same per-trial `start:end` frame indices as the neural data.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
neural = events[start:end].T.astype(np.float32)
```

iii. The justification is the same shared frame-index alignment described for the other time-varying signals.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The stored reward-zone-location output is derived from the NWB `identifier` scene string and the hard-coded schedule parser. The raw `reward_zone` time series is used only as a sanity check.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
```
```python
outputs = np.vstack([
    ...
    np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
    ...
])
```

iii. The notes justify this by saying the reward-zone schedules were parsed from scene names to mirror the manuscript/code conventions, and then checked against observed occupancy.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses fixed-session, same-environment switch, and environment-switch scene formats with regexes, applies a `30`-trial split when a switch occurs, maps `A/B/C` to `0/1/2`, and repeats that value across the trial.

ii.
```python
fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
...
split = min(change_trial, ntrials)
...
zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the schedule logic, including day-8 environment switches and the `30`-trial switch point.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse behavioral reward-event timestamps, `Reward/timestamps`, combined with per-trial time windows.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The notes explicitly say reward outcome is derived from the sparse NWB reward timestamps within each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI marks the outcome as rewarded if any reward timestamp falls between the trial’s first and last retained timestamps; otherwise it marks the trial as omitted. The label is then repeated across all timepoints in that trial.

ii.
```python
reward_trial = reward_outcome[trial_idx]
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` ties this to omission/reward trial labels and reports the resulting omission fraction as a sanity check against the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI crops all dense streams to the shortest available length across behavior and ophys arrays. It skips malformed trial candidates with no retained corridor frames, errors out if fewer than two trials or zero neurons remain, and treats missing reward-zone occupancy as non-fatal because labels come from the scene schedule; in that case it simply omits the agreement check for that trial.

ii.
```python
common_length = min(dense_lengths)
clipped_samples = max(dense_lengths) - common_length
position = position[:common_length]
position_ts = position_ts[:common_length]
speed = speed[:common_length]
lick = lick[:common_length]
environment = environment[:common_length]
reward_zone = reward_zone[:common_length]
trial_start = trial_start[:common_length]
```
```python
if len(valid) == 0:
    continue
...
if ntrials < 2:
    raise ValueError(...)
...
if n_neurons == 0:
    raise ValueError(...)
```

iii. The notes justify the cropping as an alignment fix and describe the scene-based schedule plus occupancy agreement checks as sanity checks rather than primary label sources.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work is the per-session dF/F reconstruction and correlation-based interneuron screen in `compute_interneuron_mask()`, followed by loading large NWB arrays and assembling per-trial matrices across all sessions.

ii.
```python
for plane_name in get_plane_names(fluorescence):
    ...
    for start, end in zip(starts, ends):
        trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
        ...
        trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
        dff[:, start:end] = trial_dff
```
```python
for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
    record = build_session(path)
```

iii. The trajectory explicitly says “the converter is CPU-bound” because of the “per-session dF/F reconstruction,” which is the clearest justification the AI recorded.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops inside `find_trial_segments()`, `compute_interneuron_mask()`, the session summary pass, and the final per-trial assembly pass are the main vectorization opportunities. The trial-wise dF/F reconstruction loop is especially expensive.

ii.
```python
for start, next_start in zip(starts, next_starts):
    trial_pos = position[start:next_start]
    valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
```
```python
for start, end in zip(starts, ends):
    trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
    ...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
```

iii. No explicit vectorization discussion was recorded in the notes. This answer is inferred from the code structure and the trajectory’s statement that the dF/F reconstruction dominates runtime.

## 13-c. What processing does the code repeat multiple times?

i. The code iterates over the same trial boundaries multiple times: once to build the valid mask for interneuron screening, once to compute per-trial summary variables such as environment and reward outcome, and once again to build the final neural/input/output trial arrays. It also repeatedly traverses plane groups for curation and event loading.

ii.
```python
for start, end in zip(starts, ends):
    valid_mask[start:end] = True
```
```python
for trial_idx, (start, end, next_start) in enumerate(zip(starts, ends, next_starts)):
    ...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
```

iii. No separate explicit justification was recorded. This is an inference from the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes several sanity-check-only quantities that are not part of the saved decoder dataset: `behavior_env_by_trial`, reward-zone agreement counts, `zone_counter`, `env_counter`, and the full dF/F arrays used only to derive the boolean interneuron mask. Those intermediates are discarded after summary statistics or filtering.

ii.
```python
behavior_env_by_trial = []
parsed_zone_matches = 0
parsed_zone_checks = 0
...
zone_counter = Counter()
env_counter = Counter()
```
```python
dff = np.full_like(f_roi, np.nan, dtype=np.float32)
...
all_is_int.append(corr > 0.5)
...
del f_roi, f_neu, dff, dff_valid, dff_centered
```

iii. The notes explicitly describe the environment/reward-zone checks as sanity checks, and the trajectory shows the dF/F reconstruction was done only to support the interneuron screen rather than to populate the saved neural data.
