# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data` for `sub-*` subject directories, scans each subject directory for `.nwb` files, and treats each NWB file as one session. For each session it opens the NWB file with `h5py`, loads behavioral time series from `processing/behavior/BehavioralTimeSeries`, loads ROI metadata from `processing/ophys/ImageSegmentation/PlaneSegmentation`, and loads neural data by concatenating all planes found under `processing/ophys/Deconvolved` and `processing/ophys/Fluorescence`.

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
...
    deconv_keys = list(ophys['Deconvolved'].keys())
    fluor_keys = list(ophys['Fluorescence'].keys())
...
    deconvolved = np.concatenate(deconv_list, axis=1)
    fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. In `CONVERSION_NOTES.md`, the agent says the source is NWB data from the DANDI archive and that multi-plane animals are pooled. The trajectory shows it inspected the NWB tree and then chose to load these groups directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level folder names `sub-m3`, `sub-m4`, and so on. The script keeps only subjects listed in `SUBJECT_MAP`, converts each to a shorter mouse ID like `m11`, and records a per-session `subject_idx`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
subj_id = subj_dir_name
if subj_id not in SUBJECT_MAP:
    print(f"Warning: Unknown subject {subj_id}, skipping")
    continue
...
mouse_name = subj_id.replace('sub-', '')
if mouse_name not in subject_names:
    subject_names.append(mouse_name)
subj_idx = subject_names.index(mouse_name)
```

iii. `CONVERSION_NOTES.md` documents the same NWB-subject-to-GCAMP mapping and says the dataset contains 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Session identity is taken from the filename, and the experimental day is parsed from the `ses-XX` component. That day is then used to look up the session scene in the hard-coded `SESSIONS_INFO` table.

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
for sess_file in session_files:
    filepath = os.path.join(subj_path, sess_file)
    ses_part = sess_file.split('_')[1]
    exp_day = int(ses_part.split('-')[1])
...
    scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. The trajectory shows the agent read `sessions_dict.py` from the reference code and then manually copied the scene/day mapping into `SESSIONS_INFO`.

## 1-d. How are the data split into trials?

i. Trials are split by taking indices where `trial_start > 0` and `teleport > 0`, pairing those arrays in order, truncating to the shorter count, and slicing each trial from `trial_start` up to `teleport`. The teleport period is excluded.

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

iii. `CONVERSION_NOTES.md` says trials are defined by `trial_start` and `teleport` and span the on-track period only. The trajectory also shows the agent inspected these NWB fields directly.

## 1-e. How are trials filtered based on quality controls?

i. The script does not remove many trials. It truncates the trial boundary arrays to equal length, skips malformed trials where `end <= start` or `n_timepoints < 2`, skips sessions with fewer than two trials, and flags lick-error trials but keeps them in the dataset after later converting their lick values to zero.

ii.
```python
n_trials = min(len(tstart_idx), len(teleport_idx))
...
if n_trials < 2:
    print(f"  Skipping session: only {n_trials} trials")
    return None
...
if end <= start:
    continue
...
if n_timepoints < 2:
    continue
...
licks_corrected, error_trials = correct_lick_sensor_errors(...)
```

iii. In `CONVERSION_NOTES.md`, the agent says lick-error trials are flagged by the paper criterion and then set to 0 after correction. No stronger trial-removal rule is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural output is derived from `processing/ophys/Deconvolved/*/data`, with `processing/ophys/Fluorescence/*/data` used only for the interneuron filter and `iscell` used for manual Suite2P cell curation.

ii.
```python
iscell = seg['iscell'][:]
...
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
...
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. `CONVERSION_NOTES.md` says the neural signal is the NWB deconvolved activity and that fluorescence is used as the dF/F proxy for interneuron exclusion.

## 2-b. How is the `neural` data processed?

i. The script pools imaging planes, filters ROIs by `iscell`, estimates interneurons from fluorescence-speed correlation, applies that mask to the deconvolved traces, then slices each trial and stores the result as `float32` matrices with shape `(n_neurons, n_timepoints)`.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]
...
is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
...
neural = deconvolved[:, neuron_mask]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The notes say this matches the paper’s use of OASIS-deconvolved activity and pooled multi-plane ROIs. The trajectory shows the agent relied on the NWB `Deconvolved` group rather than re-running the dF/F pipeline from the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neural QC filters are applied: `iscell[:, 0] == 1` and exclusion of neurons whose fluorescence trace has Pearson correlation `> 0.5` with running speed at on-track samples (`position > 0`).

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
...
valid_mask = nwb_data['position'] > 0
is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
...
neuron_mask = ~is_interneuron
```

iii. `CONVERSION_NOTES.md` explicitly justifies both filters from the paper and methods text, while also admitting the code uses fluorescence as a proxy for dF/F.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start-of-trial event by taking the slice between each `trial_start` index and the corresponding `teleport` index. The first frame of each stored neural matrix is therefore the first imaging sample of the trial.

ii.
```python
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The notes state that the temporal alignment event is `trial_start`, with `off_start = 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one imaging frame, taken as `1 / imaging_rate`, approximately `64.48 ms`. No temporal rebinning is applied; the script keeps frame-level samples.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. `CONVERSION_NOTES.md` states the same imaging-rate-based bin size and says alignment is at the native frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the trial length implied by `trial_start` and `teleport` plus the session imaging rate. The code does not use the raw `timestamps` vector for this variable.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
n_timepoints = end - start
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The agent’s notes justify this from the fixed imaging rate and state that the bin size is one frame.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code creates a linearly increasing vector starting at zero and stepping by one frame duration, then copies that vector into row 0 of the input matrix.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
...
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. The only explicit justification is the note that the conversion is aligned to trial start and uses the native imaging-frame cadence.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is built with exactly `end - start` samples for each trial, so it has the same number of time points as the neural slice for that trial.

ii.
```python
n_timepoints = end - start
trial_n = neural[start:end, :].T.astype(np.float32)
...
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. The notes repeatedly describe all decoder variables as frame-aligned to the start-of-trial neural data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In the script, it is not derived from a raw NWB behavior variable. It is derived from the session scene string stored in the hard-coded `SESSIONS_INFO` mapping and parsed by `parse_scene_environment`.

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

iii. `CONVERSION_NOTES.md` says environment type is determined from scene names, including a trial-30 switch on day 8.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code parses the scene into ENV1 or ENV2, fills a constant per-trial environment label, and on environment-switch sessions changes the label after trial 30.

ii.
```python
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
...
input_arr[1, :] = env_per_trial[i]
```

iii. The notes justify this from the paper’s schedule: day-8 environment switches after 30 trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although `trial number/data` is loaded from the NWB file, the actual input variable is derived from the loop index `i`, which is the ordinal position of the trial in the paired `trial_start` and `teleport` arrays.

ii.
```python
trial_num = bts['trial number/data'][:]
...
for i in range(n_trials):
    ...
    input_arr[2, :] = float(i)
```

iii. `CONVERSION_NOTES.md` says trial number is “0-indexed trial within session,” which matches the use of `i` rather than the loaded raw field.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code repeats the scalar trial index across all time points of the trial and stores it as the third input row.

ii.
```python
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
...
input_arr[2, :] = float(i)
```

iii. The agent’s notes say the variable is constant within trial and 0-indexed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` together with the per-trial start and end times defined by `position/timestamps`, `trial_start`, and `teleport`.

ii.
```python
timestamps = nwb_data['timestamps']
reward_ts = nwb_data['reward_ts']
...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. `CONVERSION_NOTES.md` states that reward outcome is determined by whether any reward timestamp falls within trial boundaries and that the first previous-outcome value is set to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first computes a binary `reward_per_trial` vector, then shifts it by one trial to create `prev_outcome`, leaving the first trial at zero, and repeats that value across all time points of the current trial.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
...
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_arr[3, :] = float(prev_outcome[i])
```

iii. The notes document exactly this shift rule.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-trial `position/data` plus reward-zone coordinates inferred from the session scene via `SESSIONS_INFO` and `parse_scene_reward_zones`.

ii.
```python
pos = nwb_data['position'][start:end]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The notes say reward-zone locations come from `sessions_dict.py` and use the paper’s zone coordinates A: 80-130, B: 200-250, C: 320-370 cm.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code computes a signed distance to the nearest point in the active reward zone: negative before the zone, zero inside the zone, and positive after the zone.

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
```

iii. The notes justify this by the decoder specification and the paper’s fixed reward-zone coordinates.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into seven bins using the task’s exact thresholds: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. The thresholds match the agent’s `CONVERSION_NOTES.md` table for this decoder output.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial position slice `start:end` used for the neural data and then stored as a time-varying row with the same number of frame samples.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
output_arr[0, :] = dist_disc
```

iii. The notes say all decoder outputs are aligned to `trial_start` at frame resolution.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `position/data`.

ii.
```python
pos = nwb_data['position'][start:end]
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

iii. The trajectory shows the agent inspected the position range in the NWB file and then used that raw stream for this output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The position trace is clipped to `[0, 450]` cm and then discretized by flooring `position / 90`.

ii.
```python
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
...
bin_size = TRACK_LENGTH / n_bins
out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. `CONVERSION_NOTES.md` describes the same five equal 90 cm bins across the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The same bin labels are recorded in `output_values` and in `CONVERSION_NOTES.md`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same frame-by-frame per-trial slice as the neural data and is written as a time-varying output row with identical length.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
output_arr[1, :] = pos_disc
```

iii. The notes describe this as frame-aligned to trial start with no rebinning.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `lick/data`.

ii.
```python
lick = bts['lick/data'][:]
...
lck = licks_corrected[start:end]
```

iii. `CONVERSION_NOTES.md` says the capacitive lick signal is loaded from the NWB behavioral time series.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script first applies a trial-level bad-lick detector, marking trials where more than 30% of frame samples have cumulative lick count `> 2`. It sets those samples to `NaN`, then later converts `NaN` to `0` and binarizes all remaining values with `> 0 -> 1`.

ii.
```python
frac_high = np.sum(trial_licks > 2) / len(trial_licks)
if frac_high > threshold:
    licks[start:end] = np.nan
...
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. The notes say this rule comes from the paper’s lick-sensor QC, and they explicitly state the bad-lick trials are set to 0 after correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick values are sliced on the same `start:end` frame interval as the neural data, then stored as a time-varying output row of equal length.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
lck = licks_corrected[start:end]
...
output_arr[3, :] = lck_binary
```

iii. The notes describe lick as a time-varying decoder output aligned to trial start.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is not derived from a raw NWB time series. It is derived from the session scene metadata copied into `SESSIONS_INFO` and parsed into zone labels A, B, or C.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The notes explicitly say reward-zone locations are determined from `sessions_dict.py`, not from the NWB `reward_zone/data` values.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into a per-trial zone label. Constant-location sessions assign one label to all trials; switch sessions assign the pre-switch zone for the first 30 trials and the post-switch zone thereafter; labels are then mapped to integers `A=0`, `B=1`, `C=2`.

ii.
```python
if scene.endswith('_LocationA') or scene.endswith('LocationA'):
    rz_labels[:] = 'A'
...
else:
    zone_before, zone_after = parse_switch_zones(scene)
    ct = min(change_trial, n_trials)
    rz_labels[:ct] = zone_before
    rz_labels[ct:] = zone_after
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. `CONVERSION_NOTES.md` justifies this from the paper’s 30-trial switch schedule and the reference `sessions_dict.py`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward/timestamps` event series together with the per-trial boundaries defined by `trial_start`, `teleport`, and `position/timestamps`.

ii.
```python
reward_ts = nwb_data['reward_ts']
timestamps = nwb_data['timestamps']
...
start_t = timestamps[tstart_idx[i]]
end_t = timestamps[teleport_idx[i]]
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
```

iii. The notes explicitly justify reward outcome this way.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward event timestamp falls within the trial’s start and end times; if so, the per-trial outcome is `1`, otherwise `0`. That scalar is then repeated across the trial’s time axis in the output matrix.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
...
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
...
output_arr[5, :] = reward_per_trial[i]
```

iii. `CONVERSION_NOTES.md` says a trial is rewarded if any reward timestamp falls within its boundaries.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles mismatched behavior and neural lengths by truncating all streams to the shorter length. It skips sessions with too few trials or no valid neurons, skips malformed trials, and handles bad lick trials by setting those samples to `NaN` and then converting them to `0` before storing outputs.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    position = position[:min_len]
    ...
    deconvolved = deconvolved[:min_len]
    fluorescence = fluorescence[:min_len]
...
if n_trials < 2:
    return None
...
if n_neurons < 1:
    return None
...
lck = np.nan_to_num(lck, nan=0.0)
```

iii. `CONVERSION_NOTES.md` specifically mentions a one-sample mismatch in `m18 ses-01` being truncated to the minimum length and describes the bad-lick handling rule.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading large NWB arrays, concatenating multi-plane fluorescence and deconvolved traces, looping over every neuron to compute speed correlations for interneuron detection, and looping over every trial to assemble per-trial neural/input/output matrices.

ii.
```python
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
...
for c in range(n_neurons):
    ...
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
...
for i in range(n_trials):
    ...
    trial_neural.append(trial_n)
```

iii. The agent did not write a separate complexity analysis, but these are the obvious hotspots in the implemented code.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron loop in `identify_interneurons`, the per-trial lick-error loop, the per-trial reward-outcome loop, and some of the per-trial input/output filling could all be vectorized.

ii.
```python
for c in range(n_neurons):
    neural_ts = neural_data[valid_mask, c]
    ...
for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
    trial_licks = licks[start:end]
...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    ...
for i in range(n_trials):
    ...
    input_arr[1, :] = env_per_trial[i]
```

iii. No explicit justification is given in the notes; this follows directly from the code structure.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly scans reward timestamps once per trial, repeatedly slices arrays trial-by-trial for each output variable, and repeatedly fills constant per-trial quantities across all time points rather than storing them once.

ii.
```python
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
input_arr[1, :] = env_per_trial[i]
input_arr[2, :] = float(i)
input_arr[3, :] = float(prev_outcome[i])
...
output_arr[4, :] = rz_loc
output_arr[5, :] = reward_per_trial[i]
```

iii. This is not discussed in `CONVERSION_NOTES.md`; it is evident from the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads several raw variables that are never used in the converted outputs, including `trial_num`, `environment`, `reward_zone`, `autoreward`, `plane_idx`, `subject_id`, and `session_id`. It also creates unused temporary input arrays (`input_arr` placeholder, `input_time_varying`, `input_per_trial`) before overwriting them.

ii.
```python
trial_num = bts['trial number/data'][:]
environment = bts['environment/data'][:]
reward_zone = bts['reward_zone/data'][:]
autoreward = bts['autoreward/data'][:]
...
plane_idx = seg['planeIdx'][:]
...
subject_id = f['general/subject/subject_id'][()]
session_id = f['general/session_id'][()]
...
input_arr = np.array([
    time_from_start[0] if False else 0,
])
...
input_time_varying = time_from_start
input_per_trial = np.array([
    float(env_per_trial[i]),
    float(i),
    float(prev_outcome[i]),
], dtype=np.float32)
```

iii. The notes do not justify these extra loads or temporary arrays. They appear to be artifacts of the implementation rather than intentional downstream requirements.
