# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file matching `/app/data/sub-*/sub-*_behavior+ophys.nwb`, opens each file with `h5py`, and reads the behavioral streams from `processing/behavior/BehavioralTimeSeries` plus the ophys streams from `processing/ophys`. It then constructs one session record per NWB file and later assembles the full dataset from all session records.

ii. 
```python
for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
    print(f"Converting {path.relative_to(data_dir.parent)}")
    record = build_session(path)
    ...
```

```python
with h5py.File(path, "r") as f:
    ...
    beh = f["processing/behavior/BehavioralTimeSeries"]
    position = np.asarray(beh["position/data"], dtype=np.float32)
    position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
    speed = np.asarray(beh["speed/data"], dtype=np.float32)
    lick = np.asarray(beh["lick/data"], dtype=np.float32)
    environment = np.asarray(beh["environment/data"], dtype=np.float32)
    reward_zone = np.asarray(beh["reward_zone/data"], dtype=np.float32)
    trial_start = np.asarray(beh["trial_start/data"], dtype=np.float32)
    reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
```

iii. `CONVERSION_NOTES.md` says the conversion follows the public NWB export while mirroring the paper/code logic. The trajectory shows the agent first inspected the paper, original repo docs, and NWB structure, then decided to read each NWB session directly.

## 1-b. How are the data split into subjects (mice)?

i. The agent treats each `sub-*` directory / NWB `subject_id` as one mouse. It stores one string per unique subject in first-encounter order and builds `subject_idx` so each session points back to that subject.

ii.
```python
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

iii. The notes say the converted mice should match the switch-task cohort from the paper. The trajectory shows the agent used the NWB `subject_id` and the subject folders after checking the original repository’s animal/session organization.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The agent orders sessions by a helper that parses mouse number from the parent directory and session number from the filename.

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

iii. The notes repeatedly describe the dataset as “list of sessions,” and the trajectory shows the agent recovered the original repository’s session organization from the file naming convention and docs.

## 1-d. How are the data split into trials?

i. The agent uses `trial_start` pulses to locate trial starts, then ends each trial at the last frame before the next start whose position is still on the 0 to 450.5 cm corridor. It does not use the explicit `teleport` signal to define trial ends.

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

iii. `CONVERSION_NOTES.md` says trials are aligned to `trial_start` and keep frames “until the last frame still on the corridor,” excluding teleport-zone frames. The trajectory shows the agent explicitly chose “keep only corridor frames from trial start to teleport onset.”

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. Trials with no valid corridor frames are dropped implicitly inside `find_trial_segments`; sessions with fewer than 2 remaining trials raise an error. There is no explicit removal of short trials, lick-fault trials, or other per-trial QC flags.

ii.
```python
if len(valid) == 0:
    continue
...
ntrials = len(starts)
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. The notes justify corridor-only trial windows and document sanity checks, but they do not describe any additional per-trial exclusion rule. The trajectory shows the agent found references to lick-sensor faults in the original notebook, but it did not implement a corresponding filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the NWB deconvolved ophys traces in `processing/ophys/Deconvolved/.../data`, restricted to manually curated ROIs using `iscell`.

ii.
```python
events = f["processing/ophys/Deconvolved"]
...
plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
```

```python
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
```

iii. The notes say the decoder neural input uses the NWB `processing/ophys/Deconvolved` traces and keeps only manually curated ROIs. The trajectory shows the agent found the original repo docs stating that deconvolved calcium `events` were the analysis timeseries.

## 2-b. How is the `neural` data processed?

i. After curation (and the separate interneuron screen), the agent concatenates kept neurons across imaging planes, crops all streams to a common length, slices the deconvolved event matrix per trial, and transposes each trial to `(n_neurons, n_timepoints)`. No temporal smoothing or rebinning is applied to the actual decoder neural signal.

ii.
```python
events = load_curated_events(f, plane_masks, keep_mask, common_length)
...
neural = events[start:end].T.astype(np.float32)
neural_trials.append(neural)
```

iii. The notes say the decoder neural signal is “suite2p OASIS deconvolved calcium events exported in the NWB files.” The trajectory shows the agent distinguished between reconstructed dF/F for screening and deconvolved `events` for the actual neural input.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies two neural QC filters: keep only manually curated cells (`iscell[:, 0] == 1`), then exclude putative interneurons whose reconstructed dF/F correlates with speed above 0.5.

ii.
```python
plane_masks = get_curated_plane_masks(f)
...
is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
keep_mask = ~is_int
events = load_curated_events(f, plane_masks, keep_mask, common_length)
```

```python
corr = np.divide(...)
all_is_int.append(corr > 0.5)
```

iii. `CONVERSION_NOTES.md` says this matches the manuscript criterion and `dayData.py`. The trajectory shows the agent found `spatial.is_putative_interneuron(...)`, `dayData.py`, and the `> 0.5` speed-correlation threshold in the original code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial starts at `trial_start` and is aligned so time 0 is the first frame of that trial. The per-trial neural matrix uses the same `start:end` slice as the behavioral inputs/outputs.

ii.
```python
starts, ends = find_trial_segments(position, trial_start)
...
neural = events[start:end].T.astype(np.float32)
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The notes state “Trials are aligned to `trial_start`,” and the metadata says the temporal alignment event is “trial start / entry onto the 0 cm corridor position.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the original imaging/behavior frame rate. The bin size is the median adjacent timestamp difference, about 64.48 ms. No temporal rebinning is applied.

ii.
```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. The notes say the median frame interval is 0.0644836 s, matching the approximately 15.5 Hz sampling described in the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamp stream attached to position, specifically `position/timestamps`.

ii.
```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The notes describe this input as time from trial start and the code uses the frame timestamps already aligned to the other streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent subtracts the trial’s first timestamp from every timestamp in that trial, producing a continuous time-since-start vector.

ii.
```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The notes describe the input as `time_from_trial_start_s`; no other transformation is documented.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same frame indices as the neural trial matrix (`start:end`) and therefore has exactly the same number of time bins per trial.

ii.
```python
neural = events[start:end].T.astype(np.float32)
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
inputs = np.vstack([
    time_trial,
    ...
])
```

iii. The notes say all decoder inputs are stored as time-varying `(4, T)` arrays for every trial, aligned to the same trial window.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The stored environment input is not taken directly from the raw `environment/data` stream. Instead, it is derived from the session `identifier` / scene name, parsed by `scene_schedule(...)` into an environment label per trial. The raw behavior `environment` stream is only used for a sanity check.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
```

```python
inputs = np.vstack([
    time_trial,
    np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
    ...
])
```

iii. The notes say the agent “parsed the session `scene` from the NWB identifier and reproduced the same schedule logic as `reward_relative.behavior.get_reward_zones`.” The trajectory shows the agent recovered the original `morph` / environment semantics but still chose scene parsing as the main source.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent regex-parses scene names such as fixed sessions, within-environment switches, and environment-switch sessions; then fills a constant environment code across each trial, using a hard split at trial 30 for switch sessions.

ii.
```python
fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
...
split = min(change_trial, ntrials)
```

iii. The notes explicitly justify this as reproducing the reward-zone schedule logic from the original code, including day-8 environment-switch sessions and 30 pre-switch trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial loop index `trial_idx`; it is not read from the raw `trial number/data` stream.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. The notes say trial number is “0-indexed within session,” matching the agent’s implementation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent assigns each trial a constant scalar equal to its 0-based index within the session and repeats that scalar across all time bins in the trial.

ii.
```python
inputs = np.vstack([
    time_trial,
    np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
    np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
    ...
])
```

iii. `CONVERSION_NOTES.md` explicitly documents that this variable is 0-indexed within session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from reward delivery timestamps. The agent first computes a per-trial `reward_outcome` list from `Reward/timestamps`, then uses the previous element of that list for the current trial.

ii.
```python
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

```python
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
```

iii. The notes say previous trial outcome uses “the previous imaged trial in the same session; the first trial is set to `0`.”

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward outcome for each trial is binarized to 0/1 based on whether any reward timestamp falls inside that trial window; previous-trial outcome is then repeated across all time bins of the next trial, with the first trial set to 0.

ii.
```python
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
...
np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
```

iii. This matches the prose in `CONVERSION_NOTES.md`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-frame position plus reward-zone bounds inferred from the parsed scene schedule (`zone_bounds`), not directly from the raw `reward_zone/data` time series.

ii.
```python
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
zone_start, zone_end = zone_bounds[trial_idx]
```

iii. The notes justify this by saying the reward-zone schedule was reconstructed from the scene name using the same logic as `reward_relative.behavior.get_reward_zones`, with occupancy-based sanity checks against the raw `reward_zone` stream.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes a signed distance to the active 50 cm reward zone: negative before the zone, zero inside, positive after.

ii.
```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
```

iii. `CONVERSION_NOTES.md` describes exactly this signed-distance rule.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is binned into 7 categories using the decoder-task thresholds.

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

iii. The notes present the same 7 bins and describe them as matching the requested decoder outputs.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `start:end` frame slice used for the neural data, producing one categorical value per neural time bin.

ii.
```python
neural = events[start:end].T.astype(np.float32)
...
outputs = np.vstack([
    discretize_distance_to_zone(pos_trial, zone_start, zone_end),
    ...
])
```

iii. The notes say the outputs are time-varying `(6, T)` arrays aligned to the same trial window as the neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the raw behavioral position stream `position/data`.

ii.
```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The notes describe absolute position as corridor position and state that corridor frames are kept while teleport frames are excluded.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips positions to the 0 to 450 cm corridor and then bins them by flooring `position / 90`, yielding five 90 cm bins.

ii.
```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. The notes say absolute position uses five equal 90 cm corridor bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into bins 0 to 4 covering the clipped corridor in five equal-width 90 cm intervals.

ii.
```python
bins = np.floor(clipped / 90.0).astype(np.int64)
bins[bins > 4] = 4
```

iii. This follows the decoder instructions as interpreted in `CONVERSION_NOTES.md`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is produced from the same frame indices as the neural trial matrix, one label per neural time bin.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
...
outputs = np.vstack([
    ...,
    discretize_position(pos_trial),
    ...
])
```

iii. The notes say all time-varying outputs are stored as `(6, T)` arrays aligned to the neural trial window.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw behavioral `lick/data` stream.

ii.
```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The notes say licks are binarized from framewise cumulative lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent thresholds the cumulative lick count at `> 0`, converting it to a binary no/yes timeseries.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says “Licks are binarized from the framewise cumulative lick counts with `lick > 0`.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick binary series is sliced with the same per-trial `start:end` indices as neural activity, so it is frame-aligned to neural data.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
...
outputs = np.vstack([
    ...,
    lick_trial,
    ...
])
```

iii. The notes say the aligned NWB behavior traces are used directly at the same frame rate.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The stored reward-zone location is derived from the session `identifier` / scene name via `scene_schedule(...)`, not directly from the raw `reward_zone/data` stream.

ii.
```python
scene = identifier.split("/")[-1]
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
```

```python
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. The notes explicitly justify this as reproducing the original reward-zone schedule logic from the scene name, then checking it against observed reward-zone occupancy.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses scene names into one of A/B/C, handles fixed and switch sessions, uses a hard switch after 30 trials, maps A/B/C to 0/1/2, and repeats the resulting label across all time bins in the trial.

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
...
zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
...
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` says fixed sessions use one reward zone; switch sessions use the first 30 trials before the switch and the remaining trials after the switch.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the sparse reward event timestamps in `Reward/timestamps`, together with the per-trial time window.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The notes say reward outcome is derived from the sparse NWB reward timestamps within each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The agent assigns 1 if any reward timestamp falls within the trial window, else 0, and repeats that scalar across all time bins of the trial.

ii.
```python
reward_trial = reward_outcome[trial_idx]
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```

iii. This is stated directly in the notes.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles mismatches and missingness pragmatically rather than with dataset-specific corrections. It crops all dense streams to a common minimum length, skips candidate trials with no valid corridor samples, clips position into corridor bounds, sets first-trial previous reward to 0, and tolerates missing reward-zone occupancy by relying on parsed scene schedules plus sanity checks. It does not implement the paper’s lick-sensor faulty-trial removal.

ii.
```python
common_length = min(dense_lengths)
...
position = position[:common_length]
...
if len(valid) == 0:
    continue
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
...
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
```

iii. The notes justify cropping as a way to align streams and describe scene/zone sanity checks. The trajectory shows the agent found the original lick-sensor-fault discussion but did not incorporate it into the conversion.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading full NWB arrays, reconstructing per-trial dF/F from fluorescence and neuropil inside `compute_interneuron_mask(...)`, and then iterating over every trial again to build neural/input/output tensors.

ii.
```python
for plane_name in get_plane_names(fluorescence):
    ...
    for start, end in zip(starts, ends):
        trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
        ...
        trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
```

```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    neural_trials.append(neural)
    input_trials.append(inputs)
    output_trials.append(outputs)
```

iii. No explicit prose justification is given for runtime; this is inferred from the code structure and from the notes’ emphasis on the extra interneuron-screen reconstruction step.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial dF/F reconstruction loop, the valid-mask construction loop, the per-trial environment/reward bookkeeping loop, and the final per-trial tensor-construction loop are all written imperatively and could be partially vectorized.

ii.
```python
for start, end in zip(starts, ends):
    valid_mask[start:end] = True
```

```python
for trial_idx, (start, end, next_start) in enumerate(zip(starts, ends, next_starts)):
    ...
```

```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
```

iii. No explicit justification is given; this is an implementation-property of `convert_data.py`.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial iteration several times: once to find segments, once to build session-level sanity summaries, again inside the interneuron dF/F reconstruction, and again to build final trial tensors. It also repeatedly parses/splits plane-level cell masks.

ii.
```python
starts, ends = find_trial_segments(position, trial_start)
...
for trial_idx, (start, end, next_start) in enumerate(zip(starts, ends, next_starts)):
    ...
is_int = compute_interneuron_mask(...)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
```

iii. No explicit justification is given beyond the agent wanting both sanity summaries and final tensors.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded computation is reconstructing dF/F only to compute the interneuron mask; the decoder never uses that dF/F afterward. The code also computes scene sanity-check statistics (`env_mismatch`, inferred zone matches, counters, summaries) that are saved only as metadata/logging and not used by downstream decoding.

ii.
```python
is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
keep_mask = ~is_int
events = load_curated_events(f, plane_masks, keep_mask, common_length)
```

```python
behavior_env_by_trial = []
parsed_zone_matches = 0
parsed_zone_checks = 0
...
summary = {
    ...
    "env_mismatch_trials": env_mismatch,
    "zone_match_fraction_when_observed": (...),
}
```

iii. `CONVERSION_NOTES.md` makes clear that the reconstructed dF/F exists only for the putative-interneuron screen, and the rest of the summary statistics are documented as sanity checks rather than decoder inputs.
