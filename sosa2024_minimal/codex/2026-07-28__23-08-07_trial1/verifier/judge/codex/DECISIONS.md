# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing every `sub-*/sub-*_behavior+ophys.nwb` file under `/app/data`, sorting them by subject and session number, then opening each file with `h5py`. Within each file it reads the dense behavior streams and the ophys groups (`Deconvolved`, `Fluorescence`, `Neuropil`, and segmentation metadata).

ii. ```python
for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
    record = build_session(path)

with h5py.File(path, "r") as f:
    beh = f["processing/behavior/BehavioralTimeSeries"]
    position = np.asarray(beh["position/data"], dtype=np.float32)
    ...
    for grp_name in ("Deconvolved", "Fluorescence", "Neuropil"):
        grp = f[f"processing/ophys/{grp_name}"]
```

iii. In the trajectory, the agent said it would "load NWB directly with `h5py`" and confirmed the dataset shape by inventorying all `152` NWB files across `11` mice before writing the converter. It justified this as matching the public NWB export structure it had inspected.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from each NWB file's subject metadata, and the final `subjects` list is built in first-seen order after sorting files by subject/session.

ii. ```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session

subject = decode_scalar(f["general/subject/subject_id"][()])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

iii. The trajectory shows the agent first confirming that the NWB files carry `general/subject/subject_id` and then using those file-level subject IDs to organize the conversion across the full dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session ordering comes from sorting the filenames by parsed subject number and `ses-XX` identifier.

ii. ```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session

for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
    record = build_session(path)
```

iii. In the trajectory, the agent explicitly checked that the NWB `session_id` and filename session number correspond to experiment days and used one file per session throughout the converter.

## 1-d. How are the data split into trials?

i. Trial starts are the indices where `trial_start > 0`. Trial ends are not taken from `teleport`; instead, for each start the AI searches forward until the last sample whose position is still on the corridor (`0` to `450.5` cm) before the next trial starts.

ii. ```python
def find_trial_segments(position: np.ndarray, trial_start: np.ndarray):
    starts = np.flatnonzero(trial_start > 0)
    next_starts = np.concatenate([starts[1:], [len(position)]])
    ends = []

    for start, next_start in zip(starts, next_starts):
        trial_pos = position[start:next_start]
        valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
        ...
        end = start + valid[-1] + 1
```

iii. The trajectory says the NWB export had no `trials` table, so trial boundaries had to be reconstructed from frame-aligned behavior. The agent justified its choice as "frame-aligned corridor extraction" and described the trial window as running from `trial_start` to the final corridor frame before teleport.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a per-trial minimum-length quality filter. It only raises an error if a session has fewer than two recovered trials.

ii. ```python
starts, ends = find_trial_segments(position, trial_start)
ntrials = len(starts)
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. The trajectory does not mention any deliberate short-trial exclusion. The agent focused on making sure sessions had enough trials for the decoder and did not describe additional trial-level QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` matrices are derived from the NWB-exported `processing/ophys/Deconvolved` traces after cell curation and interneuron removal. The raw `Fluorescence` and `Neuropil` arrays are used only to compute a dF/F-like signal for the interneuron screen.

ii. ```python
events = f["processing/ophys/Deconvolved"]
...
plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
...
fluorescence = f["processing/ophys/Fluorescence"]
neuropil = f["processing/ophys/Neuropil"]
```

iii. In the trajectory and final notes, the agent repeatedly stated that it would "use the exported deconvolved activity as the neural signal" while reconstructing per-trial dF/F only for the speed-correlation interneuron exclusion.

## 2-b. How is the `neural` data processed?

i. The AI keeps the stored deconvolved traces as the neural signal. Before that, it applies `iscell` curation, reconstructs trial-restricted dF/F with neuropil subtraction and maximin-like baseline filtering for interneuron detection, excludes putative interneurons, and then slices the surviving deconvolved traces into trials.

ii. ```python
trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
trial_dff = (trial_roi - baseline) / np.abs(baseline)
trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
...
neural = events[start:end].T.astype(np.float32)
```

iii. The trajectory says the converter would "reconstruct the paper’s per-trial dF/F only for the interneuron screen" and otherwise keep the NWB `Deconvolved` traces. It justified this as a simpler way to respect the paper’s interneuron filter while using the already exported event signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: first by manual Suite2p curation (`iscell[:, 0]`), then by excluding putative interneurons whose correlation with running speed exceeds `0.5`.

ii. ```python
iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
...
corr = np.divide(...)
all_is_int.append(corr > 0.5)
...
keep_mask = ~is_int
events = load_curated_events(f, plane_masks, keep_mask, common_length)
```

iii. The trajectory explicitly says the converter would use "curated-cell loading" and the manuscript’s "speed-correlation interneuron exclusion," and later reports the number of interneurons removed per session as a sanity check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned by splitting the session-level event trace into trial windows that begin at `trial_start`; no additional temporal shifting is applied after trial segmentation.

ii. ```python
starts, ends = find_trial_segments(position, trial_start)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    neural = events[start:end].T.astype(np.float32)
```

iii. The agent’s trajectory frames the conversion around "trial alignment" and a trial window beginning at `trial_start`, so the alignment decision is to let the trial segmentation itself define the start-aligned neural data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native framewise sampling resolution. The script estimates the bin size from the median difference of the behavior timestamps and does not apply any temporal rebinning.

ii. ```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. The trajectory describes the conversion as keeping the frame-aligned NWB signals and mentions that the timing check matched the expected `~15.5 Hz` per-plane sampling, with no mention of resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` timestamps array, specifically `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii. ```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The trajectory shows the agent inspecting the dense behavior timestamp streams and then using the aligned behavior timestamps to compute trial-relative time. It did not describe any need to distinguish among the behavior timestamp arrays.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp in that trial from every timestamp in the trial.

ii. ```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The trajectory justification is implicit: the task required time from trial start, so the agent used the standard start-subtracted timestamps within each trial window.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector uses the same `[start:end]` indices as the neural event matrix for each trial, after all dense streams are trimmed to a shared session length.

ii. ```python
common_length = min(dense_lengths)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
neural = events[start:end].T.astype(np.float32)
```

iii. In the trajectory, the agent identified a one-sample mismatch in some sessions and justified trimming every dense stream to the shortest shared length as "the safest way to preserve the original alignment semantics."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The decoded environment type is derived primarily from the NWB `identifier` string's scene name via `scene_schedule`. The framewise `environment` behavior series is read only to check for mismatches with the parsed schedule.

ii. ```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
environment = np.asarray(beh["environment/data"], dtype=np.float32)
```

iii. The trajectory says the scene names in the NWB identifiers "encode the same reward-zone and environment transitions used by the reference code," including the day-8 environment switches, and the agent chose to use that schedule directly.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses the scene name with regexes, maps `Env1` to `0` and `Env2` to `1`, applies a 30-trial split for switch sessions, and then fills each trial with a constant environment label.

ii. ```python
ENV_TO_IDX = {"Env1": 0, "Env2": 1}
CHANGE_TRIAL = 30
...
fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
...
np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32)
```

iii. The trajectory says the agent wanted the "exact scene schedule" and verified that the day-8 within-session environment-switch sessions were parsed correctly before trusting the schedule-derived labels.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial loop index after trial starts and ends have been recovered from `trial_start` and position-based corridor endpoints.

ii. ```python
starts, ends = find_trial_segments(position, trial_start)
...
np.full(time_trial.shape[0], trial_idx, dtype=np.float32)
```

iii. The trajectory did not give a separate justification beyond organizing the conversion around reconstructed trial windows. Trial number is simply the sequential index of those recovered trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No further processing is applied. The loop index is copied into a constant vector across all time bins in the trial.

ii. ```python
np.full(time_trial.shape[0], trial_idx, dtype=np.float32)
```

iii. The trajectory gives no extra rationale here; this follows directly from the decision to represent trial number as a per-trial quantity in the decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward/timestamps` stream, compared against each trial’s behavior timestamp interval.

ii. ```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The trajectory identifies reward outcome as something derived from the sparse NWB reward timestamps within each trial, and the same per-trial reward vector is later reused for previous-trial outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary reward outcome for each trial, then assigns each trial the previous trial’s binary reward outcome; the first trial is set to `0`.

ii. ```python
reward_trial = reward_outcome[trial_idx]
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
...
np.full(time_trial.shape[0], prev_reward, dtype=np.float32)
```

iii. The trajectory final notes say `previous_trial_outcome` uses the previous imaged trial in the same session and sets the first trial to `0`, matching the decoder requirement.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the framewise `position` series and a reward-zone label/boundary schedule parsed from the NWB scene identifier. The `reward_zone` behavior stream is only used for sanity checks, not as the primary source for the active zone.

ii. ```python
scene = identifier.split("/")[-1]
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
zone_start, zone_end = zone_bounds[trial_idx]
outputs = np.vstack([
    discretize_distance_to_zone(pos_trial, zone_start, zone_end),
```

iii. The trajectory says the scene names encode the reward-zone transitions used by the original analysis code, and the agent explicitly preferred the "exact scene schedule" over reconstructing reward zones from noisier framewise occupancy.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the active reward zone: negative before the zone, `0` inside it, positive after it. It then discretizes that value into seven categories.

ii. ```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
```

iii. The trajectory justification is that the reward-zone boundaries come from the parsed session schedule, and then distance is computed directly from position within the corridor trial window.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is thresholded manually into seven bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii. ```python
bins[dist < -50.0] = 0
bins[(dist >= -50.0) & (dist < -10.0)] = 1
bins[(dist >= -10.0) & (dist < 0.0)] = 2
bins[dist == 0.0] = 3
bins[(dist > 0.0) & (dist <= 10.0)] = 4
bins[(dist > 10.0) & (dist <= 50.0)] = 5
bins[dist > 50.0] = 6
```

iii. The trajectory does not justify the bin edges separately; they are taken directly from the decoder instructions and implemented explicitly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[start:end]` slices used for the neural event matrix.

ii. ```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    pos_trial = np.clip(position[start:end], 0.0, 450.0)
    neural = events[start:end].T.astype(np.float32)
```

iii. The trajectory consistently treats all trialwise variables as coming from the same recovered corridor-aligned trial window, so no extra alignment step was introduced.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the framewise `position` behavior series.

ii. ```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The trajectory shows the agent inspecting the NWB behavior streams and using `position` as the direct source for corridor location.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips trial position samples to `[0, 450]` cm, then maps them to five 90 cm bins using `floor(position / 90)` capped at `4`.

ii. ```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. The trajectory justifies this as part of the "corridor extraction" logic: the track is treated as the 450 cm corridor, excluding teleport, so positions are clipped to corridor limits before discretization.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are five equal-width 90 cm bins across the 450 cm corridor.

ii. ```python
bins = np.floor(clipped / 90.0).astype(np.int64)
bins[bins > 4] = 4
```

iii. The trajectory does not add a separate justification beyond following the requested five-bin discretization over the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slices as the neural data.

ii. ```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
neural = events[start:end].T.astype(np.float32)
```

iii. The trajectory’s alignment logic is uniform across variables: once `[start:end]` is chosen for a trial, every per-frame signal uses the same slice.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the framewise `lick` behavior series.

ii. ```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The trajectory shows the agent inventorying `lick` as one of the aligned behavior streams and then using it directly in the per-trial outputs.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick signal with `lick > 0`.

ii. ```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. The trajectory notes in the conversion notes that licks are binarized from the framewise lick values with `lick > 0` to satisfy the binary decoder output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced with the same `[start:end]` indices as the neural data.

ii. ```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
neural = events[start:end].T.astype(np.float32)
```

iii. The trajectory uses the same per-trial corridor-aligned windows for all time-varying streams, including licks.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB scene identifier, which is parsed into a per-trial reward-zone schedule. The framewise `reward_zone` series is used only as a sanity check.

ii. ```python
scene = identifier.split("/")[-1]
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64)
```

iii. The trajectory explicitly says the agent confirmed that scene names in the NWB identifiers encode the same reward-zone transitions as the reference code, so it used scene parsing as the primary reward-location source.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regexes, mapped onto reward labels `A/B/C`, and split at trial `30` for switch sessions. The resulting zone label is then broadcast across all time bins in each trial.

ii. ```python
same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
...
zone_labels[:split] = zone0
zone_labels[split:] = zone1
...
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64)
```

iii. The trajectory says the converter was built around the "exact scene schedule" and that the day-8 environment-switch sessions were tested explicitly before the agent proceeded with the full conversion.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps` compared against each trial’s timestamp interval.

ii. ```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The trajectory states that reward outcome is derived from the sparse NWB reward timestamps within each trial and reports overall rewarded fractions as a sanity check.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI sets reward outcome to `1` if any reward timestamp falls between the trial’s first and last timestamp, otherwise `0`. That scalar is then copied across the whole trial.

ii. ```python
reward_trial = reward_outcome[trial_idx]
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64)
```

iii. The trajectory justification is that reward outcome is a per-trial binary decoder target, so only whether any reward occurred in the trial matters.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly handles stream-length mismatches. It computes a `common_length` across all dense behavior and ophys streams, trims every stream to that shortest length, and records how many samples were clipped in the session summary. It also errors out if a session has fewer than two trials or no neurons after filtering.

ii. ```python
common_length = min(dense_lengths)
clipped_samples = max(dense_lengths) - common_length
position = position[:common_length]
position_ts = position_ts[:common_length]
speed = speed[:common_length]
...
if ntrials < 2:
    raise ValueError(...)
if n_neurons == 0:
    raise ValueError(...)
```

iii. The trajectory shows this decision very clearly: after a failed full run, the agent inspected a problematic session, found a one-sample mismatch between behavior and ophys, and justified trimming all dense streams to the shortest shared length as the safest way to preserve alignment semantics.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts of the AI code are session-by-session NWB reads, the nested per-trial dF/F reconstruction inside `compute_interneuron_mask`, loading the curated deconvolved event matrices, and writing the large pickle output.

ii. ```python
for plane_name in get_plane_names(fluorescence):
    ...
    for start, end in zip(starts, ends):
        trial_roi = ...
        baseline = gaussian_filter1d(...)
        baseline = minimum_filter1d(...)
        baseline = maximum_filter1d(...)
```

iii. In the trajectory, the agent said the run was "CPU-bound rather than stuck" and specifically attributed that to the "per-session dF/F reconstruction" used for the interneuron screen.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization targets are the nested per-plane/per-trial dF/F reconstruction loop in `compute_interneuron_mask`, the per-trial loop that builds `neural_trials`, `input_trials`, and `output_trials`, and the per-trial sanity-check loop that recomputes environment and reward statistics.

ii. ```python
for plane_name in get_plane_names(fluorescence):
    ...
    for start, end in zip(starts, ends):
        ...

for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
```

iii. The trajectory does not contain an explicit vectorization discussion, but its runtime commentary highlights the dF/F reconstruction as the performance bottleneck, which points directly to these nested loops.

## 13-c. What processing does the code repeat multiple times?

i. The code iterates over trials multiple times in one session: once to compute environment/reward summaries, once inside `compute_interneuron_mask`, and once more to build the final per-trial neural/input/output arrays. It also reads both fluorescence/neuropil data and the deconvolved data for the same session.

ii. ```python
for trial_idx, (start, end, next_start) in enumerate(zip(starts, ends, next_starts)):
    ...

is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
```

iii. The trajectory reflects this repeated work indirectly: it describes separate passes for corridor extraction/sanity checking, interneuron-screen dF/F reconstruction, and final event slicing during the same conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes several sanity-check quantities that are not used by the saved decoder dataset: `behavior_env_by_trial`, `env_mismatch`, `parsed_zone_matches`, `parsed_zone_checks`, `zone_counter`, `env_counter`, and the session-summary reporting fields built from them.

ii. ```python
behavior_env_by_trial = []
parsed_zone_matches = 0
parsed_zone_checks = 0
...
env_mismatch = int(np.sum(env_by_trial != np.asarray(behavior_env_by_trial, dtype=np.int64)))
...
zone_counter = Counter()
env_counter = Counter()
```

iii. The trajectory says these calculations were included as "scene sanity" and manuscript-level checks. They help validate the conversion but are discarded by downstream decoder analyses once the pickle has been written.
