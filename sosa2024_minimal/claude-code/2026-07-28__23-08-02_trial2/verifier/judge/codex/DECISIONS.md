# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over every subject directory under `/app/data` whose name starts with `sub-`, then iterates over every `.nwb` file inside each subject directory. Each file is loaded with `h5py`, and the script reads behavioral arrays, reward event arrays, ROI metadata, and neural arrays from the NWB HDF5 structure. Trials are not loaded separately from disk; they are derived later from the session-level arrays.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
with h5py.File(filepath, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
    ophys = f['processing/ophys']
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
```

iii. In the trajectory, the AI first explored the directory structure and NWB layout, then stated its design as “load NWB files from 11 mice” and process each session file. Its justification was that each subject directory corresponded to a mouse and each `.nwb` file corresponded to a session.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by top-level subdirectories named `sub-m*`. The output `subjects` list contains the mouse IDs without the `sub-` prefix, such as `m11`, and each session gets a `subject_idx` pointing into that list.

ii.
```python
for subj_dir_name in subjects:
    subj_path = os.path.join(data_dir, subj_dir_name)
    subj_id = subj_dir_name  # e.g., 'sub-m11'
    ...
    mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'

    if mouse_name not in subject_names:
        subject_names.append(mouse_name)
    subj_idx = subject_names.index(mouse_name)
```

iii. In the trajectory, the AI explicitly adopted the mapping “sub-mN -> GCAMPN” for paper metadata and used the directory names as the subject split. That choice followed its directory exploration steps.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session/day number is parsed from the filename component like `ses-03`.

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
ses_part = sess_file.split('_')[1]  # 'ses-03'
exp_day = int(ses_part.split('-')[1])
...
nwb_data = load_nwb_session(filepath)
result = process_session(nwb_data, scene, exp_day)
```

iii. In the trajectory, the AI said “session number = exp_day” and justified this from inspecting file names and matching them to the paper’s session metadata.

## 1-d. How are the data split into trials?

i. Trials are split using the indices where `trial_start > 0` as trial starts and the indices where `teleport > 0` as trial ends. The AI pairs starts and ends by order and slices each trial as `[start:end)`.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
teleport_idx = np.where(nwb_data['teleport'] > 0)[0]

n_trials = min(len(tstart_idx), len(teleport_idx))
tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. In the trajectory, the AI listed “Trial segmentation: Use trial_start and teleport events” as a key design decision. Earlier inspection steps focused on how `trial_start`, `teleport`, and `reward_zone` behaved inside NWB files, which is the basis it used for this segmentation rule.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s `< 50` timepoint threshold. It only drops trials if `end <= start` or if the trial has fewer than 2 timepoints. It also skips whole sessions if fewer than 2 valid trials remain.

ii.
```python
if end <= start:
    continue

n_timepoints = end - start
if n_timepoints < 2:
    continue
...
if len(trial_neural) < 2:
    print(f"  Skipping session: only {len(trial_neural)} valid trials")
    return None
```

iii. There is no explicit trajectory justification for the 2-sample threshold. The trajectory only mentions fixing a neural/behavior length mismatch and then validating the converted format; it does not discuss adopting a stricter trial-quality filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` output is taken from the NWB `Deconvolved` traces after filtering cells. The AI also loads `Fluorescence`, but only uses it to identify putative interneurons; it does not derive the final neural signal from raw fluorescence plus neuropil.

ii.
```python
deconv_keys = list(ophys['Deconvolved'].keys())
fluor_keys = list(ophys['Fluorescence'].keys())
...
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
...
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]
...
neural = deconvolved[:, neuron_mask]  # (timepoints, n_neurons)
```

iii. In the trajectory, the AI stated its design as “Neural data: Use deconvolved activity, filter by iscell, exclude interneurons.” Its notes and conversion write-up justify this as using the stored deconvolved calcium activity from the NWB files.

## 2-b. How is the `neural` data processed?

i. The AI pools planes by concatenation, applies `iscell` filtering, removes putative interneurons, and then uses the stored `Deconvolved` activity directly as the neural signal. It does not recompute dF/F, does not use neuropil subtraction, and does not rerun OASIS deconvolution.

ii.
```python
deconv_list = []
fluor_list = []
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])

deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
...
cell_mask = nwb_data['iscell'][:, 0] == 1
...
neural = deconvolved[:, neuron_mask]
```

iii. The trajectory justification is explicit: the AI chose to use the stored deconvolved activity and to pool planes because the paper pooled planes for analysis. It did not justify recomputing the signal from fluorescence, because it chose not to do that.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neural filters: keep only ROIs with `iscell[:, 0] == 1`, then drop cells whose fluorescence is strongly correlated with running speed (`r > 0.5`) over samples where position is positive. The interneuron screen is run on fluorescence, not on recomputed dF/F.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
...
valid_mask = nwb_data['position'] > 0  # exclude teleport period (pos = -500)

is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
...
neuron_mask = ~is_interneuron
```

```python
for c in range(n_neurons):
    neural_ts = neural_data[valid_mask, c]
    speed_ts = speed_data[valid_mask]
    ...
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
    if r > threshold:
        is_interneuron[c] = True
```

iii. In the trajectory, the AI justified this as matching the Methods’ interneuron exclusion rule of speed-correlation `> 0.5`, but it operationalized that rule using fluorescence as a proxy rather than the paper’s recomputed dF/F.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start simply by splitting the session-level neural array at the `trial_start` indices and taking per-trial slices from each start to the corresponding teleport index. There is no extra shift or interpolation.

ii.
```python
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. In the trajectory, the AI listed “Temporal alignment: Align to trial start” as one of its key design decisions. No further justification was given beyond following the task instruction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sample spacing of the stored arrays and does not perform temporal rebinning. In metadata it reports a fixed time bin size of `1000.0 / 15.5078125` ms, about 64.5 ms.

ii.
```python
'metadata': {
    ...
    'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
    ...
}
```

iii. In the trajectory, the AI summarized this as “Time bin: ~64.5 ms (1/15.5 Hz).” The trajectory does not mention any resampling step, and the code contains none.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI does not derive this variable from behavioral timestamps. Instead, it derives it from the trial length in samples and the imaging rate read from the NWB file.

ii.
```python
# Frame time
frame_time = 1.0 / nwb_data['imaging_rate']
...
n_timepoints = end - start
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. In the trajectory, the AI stated the dataset had a time bin of about 64.5 ms and then implemented time-from-start from sample index times frame duration. It did not justify using `timestamps` for this variable.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI constructs a uniformly spaced vector `0, frame_time, 2*frame_time, ...` using `np.arange(n_timepoints) * frame_time`. It does not subtract the first timestamp from the actual behavioral timestamps.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
...
input_arr[0, :] = time_from_start
```

iii. There is no detailed trajectory justification beyond the AI’s general assumption that the data had a fixed frame rate and could be represented on a regular grid.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI aligns time-from-start to neural data by giving both arrays the same per-trial sample count and using the same `[start:end)` trial slices. The temporal values themselves come from frame count, not from the session timestamps.

ii.
```python
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
...
n_timepoints = end - start
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The trajectory justification is implicit: the AI’s alignment strategy was to split every modality on the same trial boundaries and keep sample counts matched.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does not derive environment type from the NWB `environment` time series. Instead, it derives it from session metadata encoded in `SESSIONS_INFO` and parsed from the scene name.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. In the trajectory, the AI said it needed to “map subject IDs to the sessions_dict to determine reward zones” and then extended that same metadata-driven logic to environment type. Its conversion notes justify this by interpreting scene names like `Env1_*` and `Env2_*`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses the scene string. If a scene mentions only `Env1` or only `Env2`, the entire session gets that environment label. If the scene contains an environment switch, the AI assigns the first 30 trials to the pre-switch environment and the remaining trials to the post-switch environment.

ii.
```python
def parse_scene_environment(scene):
    if 'Env1' in scene and 'Env2' not in scene:
        return 0, None
    elif 'Env2' in scene and 'Env1' not in scene:
        return 1, None
    elif 'Env1' in scene and 'Env2' in scene:
        parts = scene.split('_to_')
        env_before = 0 if 'Env1' in parts[0] else 1
        env_after = 0 if 'Env1' in parts[1] else 1
        return env_before, env_after
```

```python
input_arr[1, :] = env_per_trial[i]
```

iii. The trajectory justification was that day-8 sessions are environment-switch sessions and that the switch occurs after 30 trials. The AI relied on the paper metadata rather than on the continuous NWB environment variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the within-session trial loop index `i`, after trials have been segmented from `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    input_arr[2, :] = float(i)
```

iii. In the trajectory, the AI treated trial numbering as the sequential trial index within a session after segmentation. No extra justification was given.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing is applied beyond assigning the trial loop index and repeating it across all timepoints in the trial.

ii.
```python
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
...
input_arr[2, :] = float(i)
```

iii. There is no separate trajectory justification beyond using the segmented-trial loop as the source of trial order.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward event timestamps in `Reward/timestamps` and behavioral timestamps used to define the time extent of each trial.

ii.
```python
timestamps = nwb_data['timestamps']
reward_ts = nwb_data['reward_ts']

for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1

prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. In the trajectory, the AI explicitly investigated whether reward data were event-based and concluded that trial outcome should be determined by whether any reward timestamp falls within a trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary `reward_per_trial` array, then shifts it by one trial so that each trial receives the previous trial’s outcome. The first trial is set to 0.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
...
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_arr[3, :] = float(prev_outcome[i])
```

iii. The trajectory justification was that reward is defined per trial and the first trial has no previous outcome, so it defaults to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives distance-to-reward-zone from the position trace plus reward-zone identity inferred from session scene metadata. It does not use the raw `reward_zone` time series to infer reward-zone location.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
pos = nwb_data['position'][start:end]
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. In the trajectory, the AI justified this by deciding to use session metadata plus a fixed switch at trial 30 to determine reward-zone location, after exploring the NWB `reward_zone` field and concluding that metadata would be simpler and more reliable.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each sample, the AI computes the signed distance to the current trial’s reward-zone edges: negative before the zone, zero inside the zone, positive after the zone. It then discretizes the result.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end

    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end

    return dist
```

```python
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
dist_disc = discretize_distance_to_reward(dist)
```

iii. The trajectory justification is the same metadata-based reward-zone assignment described above. Once a zone is chosen, the code comments and implementation show that the AI intended to encode signed distance relative to that zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses hard-coded comparison masks to assign the 7 requested categories: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
def discretize_distance_to_reward(distance):
    out = np.zeros_like(distance, dtype=int)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. In the trajectory, the AI said it was following the decoder specification for output binning. No more specific justification was given.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing position and neural data with the same per-trial `[start:end)` indices and computing distance sample-by-sample within that shared trial window.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
output_arr[0, :] = dist_disc
```

iii. The trajectory justification is implicit in the AI’s general alignment decision: all time-varying variables were split on the same trial boundaries.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the NWB `position` behavioral time series.

ii.
```python
position = bts['position/data'][:]
...
pos = nwb_data['position'][start:end]
```

iii. The AI’s trajectory exploration repeatedly inspected `position` ranges and used them as the basis for downstream outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips positions to `[0, 450]` cm, then discretizes them into 5 equal-width 90 cm bins using `floor(position / 90)` with clipping to `[0, 4]`.

ii.
```python
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The trajectory justification was that the track length is 450 cm and the decoder instructions requested five equal bins across the corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI thresholds position with 90 cm bins: `[0, 90)`, `[90, 180)`, `[180, 270)`, `[270, 360)`, and `[360, 450]` after clipping the raw position into the track range.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The trajectory justification was simply adherence to the decoder’s requested equal-width position bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking the `position[start:end]` slice for the same trial window as the neural slice.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
output_arr[1, :] = pos_disc
```

iii. The AI did not give a separate justification beyond using common trial boundaries for all streams.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the NWB `lick` behavioral time series, after a session-level error-correction pass.

ii.
```python
lick = bts['lick/data'][:]
...
licks_corrected, error_trials = correct_lick_sensor_errors(
    nwb_data['lick'], tstart_idx, teleport_idx,
    threshold=LICK_ERROR_THRESHOLD
)
...
lck = licks_corrected[start:end]
```

iii. In the trajectory and conversion notes, the AI justified this by invoking the paper’s lick-sensor error criterion and treating the continuous lick stream as the source for a binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first flags trials where more than 30% of samples have lick values greater than 2, sets those trials to `NaN`, then replaces `NaN` with 0 and binarizes all remaining lick values with `> 0`.

ii.
```python
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    licks = np.copy(lick_data)
    error_trials = []

    for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
        trial_licks = licks[start:end]
        ...
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > threshold:
            licks[start:end] = np.nan
            error_trials.append(i)
```

```python
lck = licks_corrected[start:end]
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
...
output_arr[3, :] = lck_binary
```

iii. In its write-up, the AI justified this as applying the paper’s lick-sensor correction rule before producing the binary lick output. The trajectory does not contain a deeper defense of replacing corrected trials with zeros.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the corrected lick array with the same `[start:end)` trial indices as the neural data and then binarizing within that window.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
lck = licks_corrected[start:end]
...
output_arr[3, :] = lck_binary
```

iii. The AI’s overall alignment strategy was to use the same trial boundaries for all trial-varying streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is not derived from a raw time series in the NWB file. It is derived from session scene metadata in `SESSIONS_INFO`, parsed into per-trial zone labels.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The trajectory shows the AI explored the noisy `reward_zone` field but then decided that scene metadata plus the known switch rule would be the clearest way to determine reward-zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene name into reward-zone labels. For fixed sessions it assigns one zone to all trials. For switch sessions it parses a before/after zone pair and switches from the first to the second at trial 30.

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    rz_labels = np.empty(n_trials, dtype='U1')
    ...
    else:
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after
```

```python
output_arr[4, :] = rz_loc
```

iii. The trajectory justification was that the paper/session metadata specify the zone identity and that reward-zone switches occur after 30 trials, so the AI encoded that rule directly.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward event timestamps in `Reward/timestamps` relative to each trial’s start and end time from the behavioral timestamps.

ii.
```python
timestamps = nwb_data['timestamps']
reward_ts = nwb_data['reward_ts']

for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. In the trajectory, the AI explicitly investigated the reward event stream and decided that outcome should be “rewarded if any reward timestamp falls within the trial boundaries.”

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI checks whether any reward event timestamp lies within each trial’s `[start_t, end_t]` interval. The result is a single binary value for the trial, repeated across all timepoints in the trial output matrix.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
...
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
...
output_arr[5, :] = reward_per_trial[i]
```

iii. The trajectory justification was that reward events are event-based rather than frame-based, so trial outcome should be determined by trial inclusion of those events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several minor issues defensively: it truncates neural and behavioral arrays to the minimum common length if they disagree; it skips malformed or extremely short trials; it skips sessions with fewer than two valid trials or no surviving neurons; and it replaces lick-error trials with zeros after marking them as `NaN`.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    position = position[:min_len]
    ...
    deconvolved = deconvolved[:min_len]
    fluorescence = fluorescence[:min_len]
```

```python
if end <= start:
    continue
...
if n_timepoints < 2:
    continue
...
if n_neurons < 1:
    print(f"  Skipping session: no valid neurons")
    return None
...
lck = np.nan_to_num(lck, nan=0.0)
```

iii. The explicit trajectory justification concerns the neural/behavior mismatch in `m18 ses-01`: the AI noticed a one-sample mismatch, then edited the script to truncate all arrays to the minimum length. The other defensive checks are present in the code and comments but are not separately justified in the trajectory.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are session loading from NWB/HDF5, concatenating large neural arrays across planes, the per-neuron correlation loop for interneuron detection, and the per-trial loop that constructs neural/input/output arrays for every trial. Unlike the reference solution, this code does not have a separate survey pass or dF/F plus OASIS recomputation pass.

ii.
```python
with h5py.File(filepath, 'r') as f:
    ...
    deconvolved = np.concatenate(deconv_list, axis=1)
    fluorescence = np.concatenate(fluor_list, axis=1)
```

```python
for c in range(n_neurons):
    ...
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

```python
for i in range(n_trials):
    ...
    trial_neural.append(trial_n)
    trial_input.append(input_arr)
    trial_output.append(output_arr)
```

iii. The trajectory shows the AI optimized for a single-pass conversion script: it explored the data once, wrote the converter, then ran sample and full conversions. It did not state a performance rationale beyond that simpler pipeline.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: the per-neuron correlation loop in `identify_interneurons`, the per-trial reward detection loop, the per-trial lick-error check, and parts of the main per-trial assembly loop. The subject/session loops are structural and less obviously vectorizable.

ii.
```python
for c in range(n_neurons):
    neural_ts = neural_data[valid_mask, c]
    speed_ts = speed_data[valid_mask]
    ...
```

```python
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

```python
for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
    trial_licks = licks[start:end]
    ...
```

iii. The trajectory does not discuss vectorization. This assessment follows directly from the implementation structure.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats several trial-wise passes over the same session: one pass for lick-error detection, one for reward-per-trial detection, and one for the main trial extraction/packing loop. It also repeatedly parses scene metadata per session. It does not, however, repeat full-file loading in a separate survey step the way the reference solution does.

ii.
```python
licks_corrected, error_trials = correct_lick_sensor_errors(
    nwb_data['lick'], tstart_idx, teleport_idx,
    threshold=LICK_ERROR_THRESHOLD
)
...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
for i in range(n_trials):
    start = tstart_idx[i]
    end = teleport_idx[i]
    ...
```

iii. The trajectory suggests this was a deliberate simplification: the AI wanted a direct conversion pipeline driven by session metadata and one processing pass per session, not a separate exploratory survey pipeline.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs a few unnecessary steps that are discarded later: it creates placeholder variables `input_arr` and `input_per_trial` before overwriting or ignoring them; it loads some NWB fields that are never used downstream (`trial_num`, `reward_zone_ts`, `autoreward`, `plane_idx`, `subject_id`, `session_id`); and it records extensive `session_info` metadata and runs summary checks that are not used by the decoder.

ii.
```python
input_arr = np.array([
    time_from_start[0] if False else 0,  # placeholder
])
...
input_time_varying = time_from_start  # (n_timepoints,)
input_per_trial = np.array([
    float(env_per_trial[i]),
    float(i),  # trial number
    float(prev_outcome[i]),
], dtype=np.float32)
...
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
```

```python
return {
    'position': position,
    'speed': speed,
    'lick': lick,
    'trial_start': trial_start,
    'teleport': teleport,
    'trial_num': trial_num,
    'environment': environment,
    'reward_zone_ts': reward_zone,
    'timestamps': timestamps,
    'autoreward': autoreward,
    ...
    'plane_idx': plane_idx,
    ...
    'subject_id': subject_id,
    'session_id': session_id,
}
```

iii. There is no explicit trajectory justification for these extra steps. They appear to be convenience code and reporting artifacts rather than required processing for the final decoder dataset.
