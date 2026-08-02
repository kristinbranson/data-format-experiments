# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `.nwb` file under `data/sub-*/*.nwb` with `glob`, opens each file directly with `h5py`, and reads the needed datasets from the NWB/HDF5 tree inside `process_session()`. It does not use `pynwb` or a separate subject/session discovery helper.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))

for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, show_processing=show, session_idx=file_idx)

with h5py.File(nwb_path, 'r') as f:
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][()]
    speed = behav['speed/data'][()]
    lick = behav['lick/data'][()]
    ...
    deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documents the `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb` layout and says there are 11 subjects and 152 sessions. The trajectory shows it inspected the NWB tree directly and then chose `h5py` as the loader.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB file paths and filenames. The output `subjects` list is built in encounter order from basenames like `sub-m11`, and each session gets a `subject_idx` pointing into that list.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11

subject_list = []
...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
...
'subjects': subject_list,
'subject_idx': np.array(all_subject_idx),
```

iii. The notes describe the dataset as organized by `sub-*` directories, and the trajectory shows the agent listing those directories early in exploration. There is no separate written justification for keeping the `sub-` prefix; that comes directly from the filename parsing in code.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session metadata are read from the file, but the session boundary itself is simply file-level.

ii.
```python
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, show_processing=show, session_idx=file_idx)

with h5py.File(nwb_path, 'r') as f:
    sess_id = f['general/session_id'][()].decode()
```

iii. `CONVERSION_NOTES.md` Step 2 says the NWB files are organized as one file per session. The trajectory also records the agent enumerating files per subject and treating them as session units.

## 1-d. How are the data split into trials?

i. Trials are split by pairing indices where `trial_start_signal > 0` with indices where `teleport_signal > 0`. For each trial, the timepoints are the half-open interval `[start, stop)`.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
n_trials = len(trial_starts)

for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    ...
    trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes explicitly list “Trial boundaries: `trial_start` to `teleport` signals” as a key decision. In Step 10 they say this matches the reference code’s `trial_start_inds` and `teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply the reference solution’s minimum-length trial filter. It only skips degenerate trials where `stop <= start`, and at the session level it skips sessions with fewer than 2 surviving trials.

ii.
```python
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]

    if stop <= start:
        continue
    ...

if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
```

iii. There is no explicit written justification for omitting the minimum-length trial filter. The notes only mention the decoder requirement that a session must have at least two trials, and the code reflects that session-level check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from the NWB `processing/ophys/Deconvolved/*/data` arrays, with `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` used for cell filtering.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]

planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. In Step 1 of `CONVERSION_NOTES.md`, the agent identifies neural data as `sess.timeseries['events']` / deconvolved calcium events and maps those to NWB `Deconvolved` arrays. The trajectory also records direct NWB inspection of those groups.

## 2-b. How is the `neural` data processed?

i. The agent concatenates all deconvolved planes across ROI columns, crops neural timepoints to match behavior if needed, casts trial arrays to `float32`, and then slices trials out of the session matrix. It also computes speed correlations for a possible interneuron screen, although in the produced dataset no cells were actually removed by that step.

ii.
```python
deconv_data = np.concatenate(deconv_planes, axis=1)

if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    deconv_data = deconv_data[:min_len, :]

trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. Step 5 of the notes says neural data are deconvolved events with multi-plane sessions concatenated. The trajectory adds the justification for the extra correlation step: computing full dF/F was “complex,” so the agent treated deconvolved events as an approximation for the paper’s interneuron exclusion metric.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first keeps only ROIs with `iscell[:, 0] == 1`. It then computes each cell’s correlation with running speed during trial timepoints and would exclude cells with correlation `> 0.5` as putative interneurons, although the run metadata indicate that zero cells were excluded this way in the final conversion.

ii.
```python
cell_mask = iscell[:, 0] == 1

valid_mask = trial_number >= 0
if np.sum(valid_mask) > 100:
    speed_corr = np.zeros(deconv_data.shape[1])
    for c in range(deconv_data.shape[1]):
        if cell_mask[c]:
            valid_neural = deconv_data[valid_mask, c]
            valid_speed = speed_valid[valid_mask]
            ...
            speed_corr[c] = r if not np.isnan(r) else 0
    interneuron_mask = speed_corr > 0.5
    cell_mask = cell_mask & ~interneuron_mask
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says the agent chose “Suite2p iscell + interneuron exclusion,” using the paper’s `r > 0.5` threshold instead of the code default. The trajectory says this was meant to match the paper’s curation more closely, while approximating dF/F with deconvolved events.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start implicitly by cutting each trial from `trial_start` to `teleport` and defining time 0 at that same trial start in the input stream.

ii.
```python
start = trial_starts[i]
stop = teleports[i]

trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes repeatedly state that temporal alignment is to trial start. Step 10 says this matches the reference code’s use of trial boundaries and that no extra event realignment was needed beyond slicing trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native behavior/imaging frame resolution, using `dt = median(diff(timestamps))`, which is about 64.48 ms. No temporal rebinning or resampling is applied.

ii.
```python
dt = np.median(np.diff(timestamps))
...
dt_ms = session_infos[0]['dt'] * 1000
...
'time_bin_size': dt_ms,
```

iii. `CONVERSION_NOTES.md` Step 2 reports a sampling rate of about 15.5 Hz, and Step 10 states the agent used raw imaging frames with no additional temporal binning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamp array `processing/behavior/BehavioralTimeSeries/position/timestamps`, together with the trial start index for each trial.

ii.
```python
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes describe this input as “timestamps - trial_start_time.” There is no explicit justification for preferring `position/timestamps` over another behavior stream’s timestamps; the agent appears to have assumed the streams are aligned.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The only processing is subtracting the first timestamp of the trial from all timestamps in that trial and casting to `float32`.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Step 5 of the notes describes this as a direct mapping from timestamps to “time from trial start.” No additional smoothing, interpolation, or binning is documented.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same `[start:stop]` trial slice as the neural matrix, after a session-level crop to the minimum neural/behavior length if those lengths disagree.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    timestamps = timestamps[:min_len]
    deconv_data = deconv_data[:min_len, :]

trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes say some sessions had a one-sample mismatch and that the agent chose to truncate to the minimum length. Step 10 then claims the trial-start-relative times matched the converted neural trials in sanity checks.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In the code, environment type is not derived from the loaded `environment` time series. Instead, it is derived from the NWB `identifier` string by parsing the scene name and converting `Env1/Env2` to `0/1`.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
environment = behav['environment/data'][()]
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The notes are inconsistent here: Step 5’s variable-mapping table says “environment signal,” but the actual code uses scene parsing. The trajectory around the reward-zone/environment exploration shows the agent deciding that the scene name plus `change_trial=30` was enough to reconstruct trial type.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses the scene string, infers any environment switch, assumes switches happen at trial 30, converts `Env1` to 0 and `Env2` to 1, and then broadcasts the per-trial value across all timepoints in the trial.

ii.
```python
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    env_map = {'Env1': 0, 'Env2': 1}
    env_vals = []
    for i in range(n_trials):
        if is_env_switch and i >= change_trial:
            env_vals.append(env_map.get(env_after, 0))
        else:
            env_vals.append(env_map.get(env_before, 0))

input_arr = np.array([
    time_from_start,
    np.full(n_tp, env_per_trial[i], dtype=np.float32),
    ...
], dtype=np.float32)
```

iii. The trajectory explicitly says the agent checked that `change_trial=30` matched example sessions and then used scene parsing as a simpler alternative to relying on the NWB reward-zone/environment streams.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the NWB `trial number` signal. It is derived from the trial loop index after trial boundaries are found from `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    input_arr = np.array([
        time_from_start,
        np.full(n_tp, env_per_trial[i], dtype=np.float32),
        np.full(n_tp, i, dtype=np.float32),
        ...
    ], dtype=np.float32)
```

iii. The notes say trial number is the per-session trial index. No separate rationale is given in the notes, but this choice is consistent with how the agent handled the reference trial boundaries everywhere else.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial index `i` is broadcast as a constant vector across the full trial.

ii.
```python
np.full(n_tp, i, dtype=np.float32)
```

iii. There is no explicit extra justification beyond using a within-session trial counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward/timestamps` event times, compared against each trial’s start and end times from the behavior timestamp array and trial boundaries.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
```

iii. Step 5 of the notes maps previous trial outcome to “reward events.” The agent’s rationale in the trajectory was that reward delivery is naturally an event stream and can be reduced to per-trial rewarded/omitted status from timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes `is_rewarded[i]` for every trial, then shifts that vector by one trial so that each trial receives the previous trial’s reward outcome. Trial 0 is assigned 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]

np.full(n_tp, prev_outcome[i], dtype=np.float32)
```

iii. The notes describe this input as “0=omitted, 1=rewarded,” and the code uses the obvious one-trial lag to implement that definition.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from per-trial position samples plus reward-zone coordinates inferred from the session identifier/scene, not from the loaded `reward_zone` behavior signal.

ii.
```python
position = behav['position/data'][()]
reward_zone_signal = behav['reward_zone/data'][()]
...
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say `scene name -> reward_zone_location`, and the trajectory shows the agent checking example sessions, confirming the trial-30 switch pattern, and then using scene parsing instead of reconstructing zones from the `reward_zone` signal.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the code computes signed distance to the current trial’s reward-zone interval: negative before the zone, 0 inside it, positive after it.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before_mask = position < rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    after_mask = position > rz_end
    dist[before_mask] = position[before_mask] - rz_start
    dist[in_mask] = 0.0
    dist[after_mask] = position[after_mask] - rz_end
    return dist
```

iii. The notes cite the paper’s reward-zone coordinates and describe this output as “position - reward zone,” which is consistent with the signed-distance computation used in code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 categories using hard-coded threshold comparisons that implement the instructed bins.

ii.
```python
bins = np.zeros(len(dist), dtype=int)
bins[dist < -50] = 0
bins[(dist >= -50) & (dist < -10)] = 1
bins[(dist >= -10) & (dist < 0)] = 2
bins[dist == 0] = 3
bins[(dist > 0) & (dist <= 10)] = 4
bins[(dist > 10) & (dist <= 50)] = 5
bins[dist > 50] = 6
```

iii. The notes and README both list the same 7 instructed distance bins. No alternate rationale is given.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial sample indices as the neural and timestamp arrays.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say the decoder variables are aligned at trial start and then sliced on common trial indices, so no separate interpolation step was introduced.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the `position` behavior time series.

ii.
```python
position = behav['position/data'][()]
...
trial_pos = position[start:stop].astype(np.float32)
```

iii. The notes map position directly to the absolute-position decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Before binning, the code clips the trial position values to the fixed corridor range `[0, 450]` cm.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. The README and notes describe the corridor as 450 cm long, and the code reflects the agent’s assumption that positions should be constrained to that nominal range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The agent uses 5 equal-width bins over `0-450` cm, implemented as 90 cm bins: `0-90`, `90-180`, `180-270`, `270-360`, and `360-450`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The README explicitly documents those 90 cm bins, so this was an intentional choice based on the track length rather than the observed `-50` to `450` position range used by the reference solution.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial `[start:stop]` slice as the neural data and all other time-varying variables.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
pos_disc = discretize_position(trial_pos)
```

iii. The notes say the trial-aligned neural and behavioral streams are kept on the same frame grid after any session-level truncation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the `lick` behavior time series.

ii.
```python
lick = behav['lick/data'][()]
...
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. The notes identify licks as coming from the lick sensor time series in the NWB behavior group.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent applies an error-correction heuristic per trial: if more than 35% of samples have lick values `> 2`, the whole trial’s lick trace is set to `NaN`; afterward, values `> 1` are clipped to 1 and `NaN` is replaced by 0, and the trial output is binarized as `trial_lick > 0`.

ii.
```python
lick_corrected = lick.copy()
for i in range(n_trials):
    trial_licks = lick_corrected[start:stop]
    if len(trial_licks) > 0:
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > 0.35:
            lick_corrected[start:stop] = np.nan

lick_binary = lick_corrected.copy()
lick_binary[lick_binary > 1] = 1
lick_binary[np.isnan(lick_binary)] = 0
...
lick_disc = (trial_lick > 0).astype(int)
```

iii. Step 1 of the notes lists lick-sensor-error correction from the reference analysis code, and Step 5 says the agent followed the code’s `0.35` threshold instead of the paper’s textual `>30%` description.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks are aligned by slicing the already corrected/binarized lick array on the same trial indices used for neural data.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_lick = lick_binary[start:stop].astype(np.float32)
lick_disc = (trial_lick > 0).astype(int)
```

iii. The notes treat licks as another framewise behavior stream on the same trial grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. In the final code, reward-zone location is derived from the NWB `identifier` scene string, parsed into labels `A/B/C` with a possible switch at trial 30. The loaded `reward_zone` behavior signal is not used to generate the final label.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
reward_zone_signal = behav['reward_zone/data'][()]
...
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The notes’ mapping table says “scene name” for reward-zone location, and the trajectory shows the agent deciding that scene parsing was simpler than reconstructing labels from the noisy `reward_zone` signal once the trial-30 switch rule had been confirmed.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses the session scene name, determines the pre-switch and post-switch reward-zone labels, applies a hard-coded `change_trial = 30`, and converts `A/B/C` to `0/1/2`. That scalar is then broadcast across the trial.

ii.
```python
CHANGE_TRIAL = 30

def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
        rz_labels.append(label)

rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The trajectory explicitly mentions checking an example switch session and concluding that `change_trial=30` “matches what we see.” The notes then carry that decision into the variable mapping.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward/timestamps` event times together with each trial’s behavioral start and end time.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
```

iii. The notes describe reward outcome as coming from reward events and define it as binary rewarded versus omitted per trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward event timestamp falls between the trial’s first and last sampled timestamps. The result is converted to `0/1` and broadcast across that trial’s timepoints in the output array.

ii.
```python
for i in range(n_trials):
    trial_start_time = timestamps[start]
    trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
    reward_in_trial = np.any(
        (reward_timestamps >= trial_start_time) &
        (reward_timestamps <= trial_end_time)
    )
    is_rewarded[i] = int(reward_in_trial)

output_arr = np.array([
    ...,
    np.full(n_tp, reward_out, dtype=int),
], dtype=int)
```

iii. The notes justify this as the natural binary per-trial reduction of the reward-event stream, with omission corresponding to 0 and delivered reward to 1.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several issues defensively: it truncates neural and behavior streams to the same minimum length, truncates trial counts if the numbers of `trial_start` and `teleport` events differ, skips trials with `stop <= start`, clips negative speeds to 0, clips positions into `[0, 450]`, and converts lick-error `NaN`s to 0 after correction. It does not attempt a reference-style reconstruction for missing reward-zone data because it does not use the reward-zone signal in the final conversion.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...

if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]

if stop <= start:
    continue

trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
trial_speed = np.clip(trial_speed, 0, None)
lick_binary[np.isnan(lick_binary)] = 0
```

iii. The notes document the neural/behavior length mismatch and the lick-sensor correction explicitly. The clipping behavior is not separately justified in writing, but it follows from the agent’s assumption of fixed valid physical ranges for track position and speed.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts of this implementation are opening each NWB file and reading the session arrays, looping over all cells to compute speed correlations, looping over all trials multiple times, and finally serializing the very large pickle file plus summary statistics.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
for c in range(deconv_data.shape[1]):
    ...
for i in range(n_trials):
    ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes report about 1.3 seconds per session and about 200 seconds total for full conversion, which implies I/O and session-level loops dominate runtime.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-cell correlation loop for interneuron screening, the per-trial reward/lick prepasses, and some of the end-of-script summary loops over every trial and every output dimension.

ii.
```python
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        ...

for i in range(n_trials):
    ...
    reward_in_trial = np.any(...)

for out_idx, out_name in enumerate(data['output_names']):
    for sess_outputs in all_output:
        for trial_output in sess_outputs:
            all_vals.extend(trial_output[out_idx].tolist())
```

iii. The agent does not discuss vectorization in the notes, so this is inferred from the implemented code structure itself.

## 13-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the same trial boundaries in one session: once to compute reward outcome, once to apply lick correction, and once again to build the final trial tensors. It also iterates over the full converted dataset after saving to compute printed output distributions.

ii.
```python
for i in range(n_trials):
    ...  # determine reward

for i in range(n_trials):
    ...  # lick correction

for i in range(n_trials):
    ...  # build neural/input/output arrays

for out_idx, out_name in enumerate(data['output_names']):
    for sess_outputs in all_output:
        for trial_output in sess_outputs:
            ...
```

iii. There is no explicit justification in the notes beyond practicality. The structure suggests the agent prioritized simple, readable passes over aggressive reuse of intermediate per-trial computations.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several raw arrays that are not used in the final mapping (`reward_zone_signal`, the `environment` time series, and most of `trial_number` except for a validity mask), computes helper constants like `REWARD_ZONE_CENTERS` and `SAMPLING_RATE` that are never consumed, and prints a full distribution summary after the dataset is already saved. The downstream decoder only uses the saved pickle.

ii.
```python
REWARD_ZONE_CENTERS = {k: (v[0] + v[1]) / 2 for k, v in REWARD_ZONE_DICT.items()}
SAMPLING_RATE = 15.5

reward_zone_signal = behav['reward_zone/data'][()]
environment = behav['environment/data'][()]
trial_number = behav['trial number/data'][()]
...
env_per_trial = get_env_per_trial(scene_info, n_trials)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
for out_idx, out_name in enumerate(data['output_names']):
    ...
```

iii. No explicit rationale is given. These look like leftovers from exploration, sanity checking, or alternative implementations that were not fully removed from the final script.
