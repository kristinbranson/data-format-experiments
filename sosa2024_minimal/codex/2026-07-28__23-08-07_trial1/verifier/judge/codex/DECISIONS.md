# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under the data directory, sorts them by parsed subject/session, and converts each file with `build_session()`. It reads the NWB files directly with `h5py` rather than `pynwb`.

ii.
```python
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
```

iii. The trajectory shows the agent inspected the NWB layout directly and then chose `h5py` after confirming it was available. `CONVERSION_NOTES.md` says the conversion uses the public NWB export as the source data.

## 1-b. How are the data split into subjects (mice)?

i. Sessions are grouped by the subject ID stored in each NWB file. The subject list is built from the converted session records, and `subject_idx` maps each session back to that subject.

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"][()])
...
subjects = []
subject_to_idx = {}
for record in session_records:
    subject = record["subject"]
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subjects)
        subjects.append(subject)
```

iii. The notes treat mice as the manuscript cohort and report manuscript-level checks by subject count, indicating the AI relied on NWB subject metadata rather than only directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order is determined by parsing the subject and session number from the file path.

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

iii. The trajectory repeatedly refers to “every NWB session” and the final dataset summary counts 152 sessions, so the AI explicitly treated one file as one session.

## 1-d. How are the data split into trials?

i. Trials start where `trial_start > 0`. Instead of using `teleport` to end trials, the AI takes frames until the last frame before leaving the 0 to 450 cm corridor, using `position` within the interval up to the next trial start.

ii.
```python
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
```python
starts, ends = find_trial_segments(position, trial_start)
```

iii. `CONVERSION_NOTES.md` justifies this as matching the paper’s 450 cm track, explicitly stating that frames are kept from `trial_start` until the last frame still on the corridor and that teleport-zone frames are excluded.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s explicit minimum-length trial filter. It only drops degenerate trial segments implicitly when no valid corridor frames are found, and raises an error if a session ends up with fewer than two trials.

ii.
```python
for start, next_start in zip(starts, next_starts):
    trial_pos = position[start:next_start]
    valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
    if len(valid) == 0:
        continue
```
```python
ntrials = len(starts)
if ntrials < 2:
    raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```

iii. No explicit short-trial justification appears in the notes or trajectory. The notes frame the main trial-quality choice as excluding teleport-zone frames rather than dropping short trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices come from NWB `processing/ophys/Deconvolved` event traces after cell filtering. The filtering step also consults `Fluorescence` and `Neuropil` to compute the interneuron mask.

ii.
```python
events = f["processing/ophys/Deconvolved"]
...
plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
```
```python
fluorescence = f["processing/ophys/Fluorescence"]
neuropil = f["processing/ophys/Neuropil"]
```

iii. `CONVERSION_NOTES.md` says the decoder neural input uses NWB deconvolved traces, while fluorescence and neuropil are only reconstructed “for the putative-interneuron screen.”

## 2-b. How is the `neural` data processed?

i. The AI keeps curated ROIs, reconstructs per-trial dF/F only to identify putative interneurons, excludes those interneurons, then uses the remaining deconvolved event traces as the neural signal. It concatenates curated cells across planes.

ii.
```python
plane_masks = get_curated_plane_masks(f)
is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
keep_mask = ~is_int
events = load_curated_events(f, plane_masks, keep_mask, common_length)
```
```python
neural = events[start:end].T.astype(np.float32)
neural_trials.append(neural)
```

iii. The notes say this mirrors manuscript/code logic: use exported deconvolved events for decoding, but reconstruct dF/F with neuropil subtraction, maximin baseline, and smoothing for the interneuron screen.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First the AI keeps only manually curated ROIs where `iscell[:, 0] == 1`. It then removes putative interneurons defined by correlation between reconstructed dF/F and speed greater than `0.5`.

ii.
```python
iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
...
masks[plane_name] = mask
```
```python
corr = np.divide(
    numerator,
    denom,
    out=np.zeros_like(numerator, dtype=np.float32),
    where=denom > 0,
)
all_is_int.append(corr > 0.5)
```

iii. `CONVERSION_NOTES.md` explicitly justifies both filters, and trajectory step 42 states the plan to use `iscell[:,0] == 1` plus the paper/code interneuron rule `corr(dff, speed) > 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start. For each trial, the AI slices the event matrix from the `trial_start` frame to the trial end chosen by `find_trial_segments()`.

ii.
```python
starts, ends = find_trial_segments(position, trial_start)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    neural = events[start:end].T.astype(np.float32)
```

iii. The notes explicitly say “Trials are aligned to `trial_start`,” and the metadata names the alignment event as “trial start / entry onto the 0 cm corridor position.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame resolution and does not rebin. It records the bin size as the median frame interval from `position/timestamps`, about 64.5 ms.

ii.
```python
dt_seconds = float(np.median(np.diff(position_ts)))
...
"time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. `CONVERSION_NOTES.md` says the median frame interval is `0.0644836 s` (`64.4836 ms`) and that this matches the approximately `15.5 Hz` per-plane sampling in the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives this input from `processing/behavior/BehavioralTimeSeries/position/timestamps`, not from `trial number` timestamps.

ii.
```python
position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
...
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. No separate justification is given in the notes beyond treating the behavior streams as frame-aligned and using trial-relative timestamps after clipping all streams to a common length.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp in that trial from every timestamp in the same slice.

ii.
```python
time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
...
inputs = np.vstack([
    time_trial,
```

iii. This is implicit in the code and consistent with the notes’ description of `time_from_trial_start_s`. No extra justification beyond trial-start alignment is given.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector uses the exact same `start:end` frame indices as the neural slice. All streams are truncated to the common minimum length before trial extraction.

ii.
```python
common_length = min(dense_lengths)
...
position_ts = position_ts[:common_length]
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
    neural = events[start:end].T.astype(np.float32)
```

iii. The notes describe the aligned behavior timestamps as matching the imaging frame rate. The agent’s process justification is therefore that simple shared frame indexing is sufficient after clipping streams to the same length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The final environment input is derived from the session `identifier` string, specifically the parsed `scene` name, not directly from the framewise `environment` behavior series. The behavior `environment` series is used only as a sanity check.

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
```

iii. `CONVERSION_NOTES.md` says the AI “reproduced the same schedule logic as `reward_relative.behavior.get_reward_zones`” and uses `Env1 -> 0`, `Env2 -> 1`, while also reporting a sanity check that parsed schedule and behavior stream matched on every trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI regex-parses the scene name, assigns a fixed environment for constant sessions, and for switch sessions splits trials at trial 30. It then broadcasts the per-trial environment label across all frames in the trial.

ii.
```python
fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
...
split = min(change_trial, ntrials)
```
```python
np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
```

iii. The notes justify this by saying it mirrors the manuscript/code reward-zone schedule logic and explicitly handles day-8 environment-switch sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the order of the extracted trial loop, i.e. the within-session trial index after segmentation.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` states that `trial_number` is `0`-indexed within session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. There is no processing beyond taking the loop index and repeating it across all timepoints in the trial.

ii.
```python
np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. The notes explicitly say the variable is `0`-indexed within session; no further transformation is described.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from NWB `Reward/timestamps`, combined with per-trial start and end timestamps from `position/timestamps`.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. The notes say “previous_trial_outcome uses the previous imaged trial in the same session,” and reward outcome is derived from sparse reward timestamps within each trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary `reward_outcome` for every trial by checking whether any reward timestamp falls inside the trial window. It then assigns each trial the previous trial’s outcome, with the first trial hard-coded to `0`.

ii.
```python
reward_trial = reward_outcome[trial_idx]
prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
...
np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` explicitly says the first trial is set to `0` and that previous-trial outcome is defined from the previous imaged trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from trial position plus reward-zone bounds inferred from the parsed `scene` schedule. The framewise `reward_zone` behavior stream is only used for a sanity check, not for the actual decoder target.

ii.
```python
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
...
zone_start, zone_end = zone_bounds[trial_idx]
...
discretize_distance_to_zone(pos_trial, zone_start, zone_end),
```
```python
zone_vals = reward_zone[start:end] > 0
if np.any(zone_vals):
    zone_pos = np.clip(position[start:end][zone_vals], 0.0, 450.0)
    ...
    parsed_zone_matches += int(inferred == zone_labels[trial_idx])
```

iii. The notes justify this by saying the conversion follows paper/code reward-zone schedules from scene names and then checks that those parsed schedules match observed reward-zone occupancy.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the active reward zone: negative before the zone, zero inside it, positive after it.

ii.
```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
```

iii. `CONVERSION_NOTES.md` gives exactly this interpretation and says it is distance to the nearest point in the active 50 cm reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is manually thresholded into seven categories matching the requested bins.

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

iii. The notes list the same distance interpretation and state that the decoder outputs are stored as categorical arrays.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance categories are computed from the same `start:end` trial slices used for the neural data, so alignment is by shared frame indices.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    pos_trial = np.clip(position[start:end], 0.0, 450.0)
    ...
    neural = events[start:end].T.astype(np.float32)
    outputs = np.vstack([
        discretize_distance_to_zone(pos_trial, zone_start, zone_end),
```

iii. The notes describe all decoder inputs and outputs as aligned framewise to the trial-aligned imaging stream.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the framewise `position` behavior series.

ii.
```python
position = np.asarray(beh["position/data"], dtype=np.float32)
...
pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The notes refer to a 450 cm virtual linear track and use clipped corridor positions for trial outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips positions to the corridor range `[0, 450]` and then bins them.

ii.
```python
pos_trial = np.clip(position[start:end], 0.0, 450.0)
...
discretize_position(pos_trial),
```
```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
```

iii. `CONVERSION_NOTES.md` justifies the corridor clipping through the trial-window decision: teleport-zone frames are excluded and the corridor is treated as 0 to 450 cm.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI divides the 0 to 450 cm corridor into five equal 90 cm bins by computing `floor(position / 90)`, with values above 4 clipped back to 4.

ii.
```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. The notes explicitly say “Absolute position uses five equal 90 cm corridor bins.”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position bins are computed from the same per-trial `start:end` frame slices used for the neural matrix.

ii.
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    pos_trial = np.clip(position[start:end], 0.0, 450.0)
    neural = events[start:end].T.astype(np.float32)
```

iii. No extra alignment step is documented; the AI’s stated logic is shared frame indexing after trial alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the framewise NWB `lick` behavior series.

ii.
```python
lick = np.asarray(beh["lick/data"], dtype=np.float32)
...
lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says licks are binarized from the framewise cumulative lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes licks with `lick > 0`.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
...
outputs = np.vstack([
    ...
    lick_trial,
```

iii. The notes explicitly justify this as converting the cumulative lick count stream into a binary decoder output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick vector is sliced with the same `start:end` trial indices as the neural data.

ii.
```python
lick_trial = (lick[start:end] > 0).astype(np.int64)
...
neural = events[start:end].T.astype(np.float32)
```

iii. The AI does not describe any extra alignment step; shared frame indexing is the stated approach throughout the notes.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The final reward-zone label is derived from the parsed `scene` string in the NWB `identifier`, not from the `reward_zone` behavior stream used in the human reference solution. The observed `reward_zone` occupancy is only used for agreement checks.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
scene = identifier.split("/")[-1]
...
env_by_trial, zone_labels, zone_idx, zone_bounds = scene_schedule(scene, ntrials)
```
```python
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` says the conversion follows reward-zone schedules from scene names and reports perfect agreement between the parsed schedule and observed reward-zone occupancy when occupancy was present.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI regex-parses scene names into fixed sessions, within-environment switches, and environment-switch sessions. It uses a fixed change point at trial 30 and maps labels `A/B/C` to indices `0/1/2`, then broadcasts the per-trial label across all frames.

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
CHANGE_TRIAL = 30
...
split = min(change_trial, ntrials)
...
zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
```
```python
np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. The notes justify this as mirroring `reward_relative.behavior.get_reward_zones` and explicitly mention handling day-8 environment-switch sessions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from NWB `Reward/timestamps` and the per-trial time bounds from `position/timestamps`.

ii.
```python
reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
...
reward_outcome.append(
    int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
)
```

iii. `CONVERSION_NOTES.md` explicitly says reward outcome is derived from sparse NWB reward timestamps within each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI marks reward outcome as `1` if any reward timestamp falls within that trial’s timestamp interval and `0` otherwise. It then repeats that binary label across all frames in the trial.

ii.
```python
reward_trial = reward_outcome[trial_idx]
...
np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```

iii. The notes describe reward outcome as a per-trial categorical output built from sparse reward timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips all dense streams to the shortest common length across behavior and ophys arrays, skips trial segments with no valid corridor samples, raises if fewer than two trials remain, and raises if no neurons survive filtering. It does not implement the reference solution’s explicit short-trial filter or reward-timestamp alignment assertion.

ii.
```python
common_length = min(dense_lengths)
...
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

iii. The trajectory and notes emphasize defensive sanity checks and stream alignment. The AI’s documented rationale is mainly to keep frame-aligned corridor data and validate manuscript-level counts.

## 13-a. What are the most time-consuming steps of the code?

i. The likely slowest parts are reading large NWB arrays for every session, reconstructing per-trial dF/F for every curated cell in `compute_interneuron_mask()`, building the full 9.3 GB dataset, and writing the large pickle outputs.

ii.
```python
for plane_name in get_plane_names(fluorescence):
    ...
    for start, end in zip(starts, ends):
        trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
        ...
        dff[:, start:end] = trial_dff
```
```python
full_data = build_full_dataset(args.data_dir)
...
save_pickle(args.full_output, full_data)
```

iii. The trajectory explicitly treats the full-session scan, full conversion, and large output files as the expensive steps, and the code structure shows the interneuron screen is the heaviest per-session computation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are the per-plane and per-trial loops in `compute_interneuron_mask()`, the per-trial loop that builds `neural/input/output`, and the loop that infers reward and sanity-check statistics trial by trial.

ii.
```python
for plane_name in get_plane_names(fluorescence):
    ...
    for start, end in zip(starts, ends):
        ...
```
```python
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    neural_trials.append(neural)
    input_trials.append(inputs)
    output_trials.append(outputs)
```

iii. No explicit vectorization discussion appears in the notes. This assessment is inferred directly from the implementation.

## 13-c. What processing does the code repeat multiple times?

i. The code makes repeated passes over the same session frames: once to derive reward/environment sanity stats, once to reconstruct dF/F for interneuron detection, and again to build the final trial tensors. It also rereads all sessions during downstream verification and decoder runs.

ii.
```python
for trial_idx, (start, end, next_start) in enumerate(zip(starts, ends, next_starts)):
    ...
    reward_outcome.append(...)
```
```python
is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
...
for trial_idx, (start, end) in enumerate(zip(starts, ends)):
    ...
    neural_trials.append(neural)
```

iii. The notes emphasize both sanity checking and manuscript-matching curation, so these repeated passes appear deliberate rather than accidental.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several diagnostics that are not used as decoder inputs or outputs: `behavior_env_by_trial`, `parsed_zone_matches`, `parsed_zone_checks`, `zone_counter`, `env_counter`, and the summary metadata. It also reconstructs full dF/F traces only to throw them away after producing the interneuron mask.

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

iii. The trajectory shows the AI prioritized self-validation against the paper, so this extra work was mainly for sanity checks and filtering rather than for the final decoder tensors.
