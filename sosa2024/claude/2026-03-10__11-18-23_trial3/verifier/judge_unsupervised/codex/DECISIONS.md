# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates all `sub-*` directories under `data/`, collects every `.nwb` file, and processes each file as one session. Within each session it opens the NWB file with `h5py`, loads behavioral time series from `processing/behavior/BehavioralTimeSeries`, loads ophys data from `processing/ophys`, and later splits the continuous arrays into trials.

ii. ```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    ...
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj.replace('sub-', ''),
                'filepath': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file,
            })

with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    position = behav['position']['data'][:]
    speed = behav['speed']['data'][:]
    ...
    ophys = f['processing']['ophys']
    seg = ophys['ImageSegmentation']['PlaneSegmentation']
```

iii. In `CONVERSION_NOTES.md` the agent says each NWB file contains one session for one mouse, with behavioral variables already aligned to imaging frames and with precomputed deconvolved activity in the NWB file.

## 1-b. How are the data split into subjects?

i. Subjects are split by directory name. The code treats each `sub-*` directory as one mouse and strips the `sub-` prefix to get subject IDs such as `m11`.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```

iii. The notes say the data are organized as `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`, and Step 0 records 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The final dataset stores one list entry per processed file in `data['neural']`, `data['input']`, and `data['output']`.

ii. ```python
for i, file_info in enumerate(nwb_files):
    neural_trials, input_trials, output_trials, session_info = process_session(
        file_info['filepath'], show_processing=show_processing, session_idx=i)
    ...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. The notes explicitly state “Each NWB file = 1 session in the output data structure.”

## 1-d. How are the data split into trials?

i. Trials are defined from the `trial_start` signal to the `teleport` signal in the behavioral time series. The code finds trial start indices, finds teleport indices, pairs them, drops invalid pairs where teleport does not follow the start, and slices all arrays from `start:end`.

ii. ```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
...
valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
...
start = trial_starts[t]
end = teleports[t]
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
```

iii. In the notes and trajectory the agent repeatedly states that trial boundaries should follow the reference code’s `trial_start_inds` to `teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. The code does very little whole-trial filtering. It skips trials shorter than 5 time points, does not drop lick-error trials but sets their lick samples to `NaN`, and keeps trials otherwise.

ii. ```python
for t in range(n_trials):
    ...
    if len(trial_lick) > 0:
        frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
        if frac_bad > LICK_ERROR_FRACTION:
            lick_binary[start:end] = np.nan
            lick_error_trials.append(t)
...
if n_timepoints < 5:
    continue  # Skip very short trials
```

iii. The notes mention lick sensor error correction and very short-trial skipping, but they do not describe any broader trial rejection step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are derived from the NWB `processing/ophys/Deconvolved/.../data` arrays, filtered by `iscell`. For multi-plane sessions the agent concatenates all planes first. It also loads fluorescence and neuropil data, but only for interneuron detection.

ii. ```python
iscell = seg['iscell'][:, 0].astype(bool)
planes = sorted(ophys['Deconvolved'].keys())
...
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. The notes say the critical discovery was that the NWB files already contain precomputed deconvolved events matching `sess.timeseries['events']`, so dF/F does not need to be recomputed for the decoder input itself.

## 2-b. How is the `neural` data processed?

i. The neural signal used in the output dataset is the deconvolved activity directly from the NWB file. The code concatenates planes if needed, filters neurons, slices into trials, transposes to `(neurons, time)`, and replaces any `NaN` values with zero. It computes dF/F only for a separate interneuron-identification step.

ii. ```python
neural_all = deconv_data[:, final_neuron_mask]  # (n_samples, n_neurons)
...
trial_neural = neural_all[start:end, :].T.copy()  # (n_neurons, n_timepoints)
trial_neural[np.isnan(trial_neural)] = 0
...
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
```

iii. The notes say the decoder should use deconvolved events, not raw fluorescence or dF/F, and that dF/F is only needed to reproduce the putative interneuron screen from the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only `iscell` ROIs and additionally removes putative interneurons defined as neurons with Pearson correlation greater than 0.5 between dF/F and running speed. The dF/F used for this check is computed from fluorescence and neuropil with the paper’s maximin baseline procedure.

ii. ```python
iscell = seg['iscell'][:, 0].astype(bool)
...
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
```

iii. The notes cite the methods text and reference code for “Pearson r > 0.5 with speed” as the interneuron criterion, and Step 5 lists this as a key decision.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned to trial start by slicing each trial from its start index to its teleport index. The first column of each trial matrix corresponds to the first imaging frame after `trial_start`.

ii. ```python
start = trial_starts[t]
end = teleports[t]
trial_neural = neural_all[start:end, :].T.copy()
...
'temporal_alignment_event': 'start of each trial (first imaging frame on track after teleport)',
'off_start': 0.0,
```

iii. The notes say “Temporal alignment: Align to trial start (first frame of each trial).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code uses one native imaging frame per time bin, with `FRAME_PERIOD = 1 / 15.5078125` s. No temporal rebinning is applied.

ii. ```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
...
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
```

iii. Step 5 in the notes says “Time bin = 1 imaging frame: ~64.5 ms. This matches the native temporal resolution.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the trial segmentation and the assumed constant imaging frame period, not from the NWB timestamps directly. Once a trial’s length is known, the code creates `0, 1, 2, ...` frame indices and multiplies by `FRAME_PERIOD`.

ii. ```python
FRAME_PERIOD = 1.0 / IMAGING_RATE
...
n_timepoints = end - start
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The notes say this variable is “computed from frame timestamps relative to trial start,” but the final code implements it from frame count times the global frame period.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. No interpolation or rebinning is applied. The code simply creates a per-trial ramp in seconds from zero at trial start and broadcasts it as row 0 of the input matrix.

ii. ```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
trial_input[0, :] = time_from_start
```

iii. The agent’s planning notes describe it as a simple time-varying input aligned to the first frame of each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is generated using the same `n_timepoints = end - start` that defines the neural slice, so every trial has one time value per neural frame and both begin at the same trial-start index.

ii. ```python
n_timepoints = end - start
trial_neural = neural_all[start:end, :].T.copy()
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input[0, :] = time_from_start
```

iii. The notes repeatedly state that all streams are aligned to trial start at the imaging-frame resolution.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the NWB behavioral variable `environment`.

ii. ```python
environment = behav['environment']['data'][:]
...
env_vals = environment[start:end]
```

iii. Step 5 of the notes maps NWB `environment` directly to the decoder input “Environment type.”

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the code takes the median of the nonnegative `environment` values within the trial and then broadcasts that single value across all time bins in the input matrix.

ii. ```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
...
trial_input[1, :] = env_val
```

iii. The notes describe environment as a per-trial binary variable and assume the within-trial value is constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although the code loads the NWB `trial number` time series, the final `trial_number` input is actually derived from the local trial loop index `t` after trial splitting.

ii. ```python
trial_number = behav['trial number']['data'][:]
...
trial_num = float(t)
...
trial_input[2, :] = trial_num
```

iii. The notes say the source should be NWB `trial number`, but the final implementation uses the enumerated trial index instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code turns each trial’s zero-based loop index into a float and broadcasts it across all time bins in that trial.

ii. ```python
trial_num = float(t)
...
trial_input[2, :] = trial_num
```

iii. The agent’s notes planned to use the raw `trial number` variable, but the shipped code uses the simpler derived index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the behavioral reward event stream `Reward`, specifically from whether the previous trial contained any reward event timestamp inside its trial boundaries.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
...
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
...
trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
trial_rewarded[t] = trial_rewards
...
prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. The notes say “Previous trial outcome: Derived from whether preceding trial had reward events.”

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are mapped to frame indices, each trial is labeled rewarded if any reward frame falls inside it, then the previous trial’s rewarded/not-rewarded flag is copied into the current trial. The first trial is set to 0.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    ...
    trial_rewarded[t] = trial_rewards

prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. The notes describe the first trial as having no previous outcome and defaulting to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the per-frame `position` time series plus an inferred reward-zone start and end for that trial. The reward-zone identity itself is inferred from `reward_zone` occupancy and position.

ii. ```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The notes say reward-zone location had to be inferred because the NWB file lacked the original scene-string metadata used in the paper code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each time bin the code computes signed distance to the nearest point in the current reward zone: negative before the zone, zero inside the zone, positive after the zone.

ii. ```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end
```

iii. Step 5 in the notes states this exact signed-distance convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distances are discretized into the 7 bins requested in the task, with a dedicated exact-zero bin for in-zone samples.

ii. ```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The bin definitions are copied into the notes and into the script constants.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `trial_pos = position[start:end]`, so it has exactly one sample per neural frame in the same trial-aligned slice.

ii. ```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
trial_output[0, :] = dist_bins
```

iii. The notes say all time-varying inputs and outputs are kept at the native imaging-frame resolution after slicing by trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the NWB `position` time series.

ii. ```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
```

iii. The mapping table in the notes says absolute position comes from NWB `position`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code converts the continuous position in centimeters into bin IDs by dividing by 90 cm, flooring, clipping to the range 0 to 4, and storing the result as a time-varying categorical output.

ii. ```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The notes say the agent intentionally chose “5 equal bins of 90 cm each” to match the task specification.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm.

ii. ```python
"""
0: 0-90 cm
1: 90-180 cm
2: 180-270 cm
3: 270-360 cm
4: 360-450 cm
"""
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The notes and code comments both describe these equal-width bins explicitly.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced over the exact same `start:end` frame interval as the neural trial, so the position-bin series is frame-aligned to the neural matrix.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
trial_output[1, :] = pos_bins
```

iii. The notes describe position as a time-varying output at the imaging frame rate.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the NWB behavioral variable `lick`, which the notes identify as cumulative lick count per frame.

ii. ```python
lick = behav['lick']['data'][:]
...
trial_lick = lick_binary[start:end]
```

iii. The notes say the NWB `lick` stream is cumulative and must be binarized after quality control.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code clips the cumulative lick signal into `0/1`, marks samples from lick-error trials as `NaN`, and finally converts `NaN` to `0` when writing the categorical output.

ii. ```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)
...
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes cite the reference code’s rule `licks[licks > 1] = 1` and the `>0.35` bad-trial threshold.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced on the same per-trial `start:end` interval as the neural data, so it has one sample per neural time bin.

ii. ```python
trial_lick = lick_binary[start:end]
...
trial_output[3, :] = lick_vals
```

iii. The notes describe lick as a time-varying binary output on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `position` and `reward_zone` occupancy. The code looks at positions where `reward_zone > 0`, infers whether those positions are closest to zone A, B, or C, and stores the corresponding zone label/index.

ii. ```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
...
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. The notes say the NWB file lacks the original scene string, so the reward zone had to be inferred from occupancy positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code computes the mean position of in-zone samples within a trial, assigns the nearest of the three canonical zones, forward-fills missing trials using the last known zone, back-fills leading missing trials using the first observed zone, and falls back to zone B if still unresolved.

ii. ```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    ...
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    ...
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        ...

for t in range(n_trials):
    zone = identify_reward_zone(...)
    if zone is not None:
        last_known_zone = zone
    trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. The notes frame this as a workaround for missing scene metadata and say the agent would “infer from position where reward_zone > 0.”

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the NWB `Reward` event timestamps, mapped onto the behavioral/imaging frame axis.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
...
trial_rewarded[t] = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
```

iii. The notes say reward outcome should come from reward events and be stored per trial as 0/1.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices and a trial is labeled rewarded if any reward index falls within that trial’s `start:end` interval. The per-trial value is then broadcast across time bins in `trial_output[5, :]`.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    ...
    trial_rewarded[t] = trial_rewards
...
reward_out = int(trial_rewarded[t])
trial_output[5, :] = reward_out
```

iii. The mapping table in the notes describes this output as binary `0=no, 1=rewarded`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases by truncating behavior and neural arrays to a common minimum length, skipping invalid or very short trials, forward/back-filling missing reward-zone labels, falling back to zone B if still unresolved, replacing `NaN` neural values with zero, and replacing `NaN` lick values with zero in the final output.

ii. ```python
if n_behav_samples != n_neural_samples:
    position = position[:n_samples]
    ...
    deconv_data = deconv_data[:n_samples, :]

valid = teleports > trial_starts
...
if trial_rz_label[0] is None:
    ...
trial_rz_start[t], trial_rz_end[t] = 200, 250
...
trial_neural[np.isnan(trial_neural)] = 0
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes explicitly mention the one-frame behavior/neural mismatch fix for multi-plane animals and describe the reward-zone inference as needing fallbacks because some trials lack clean in-zone occupancy.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large NWB arrays, computing dF/F over all neurons and frames, and computing the interneuron screen. The notes report full conversion taking about 879 seconds for 152 sessions and specifically call out the interneuron step as a target for optimization.

ii. ```python
t_load = time.time() - t0
...
t1 = time.time()
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
t_dff = time.time() - t1
...
t1 = time.time()
is_interneuron = identify_interneurons(dff, speed, iscell)
t_int = time.time() - t1
```

iii. In Step 6 of the notes the agent says “Vectorized interneuron detection ... 4.3s -> 0.4s per session” and “Full conversion: ~879s.”

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent itself identified the interneuron loop as a vectorization target and rewrote it accordingly. Other remaining per-trial loops still handle reward detection, reward-zone inference, environment summarization, lick-error checks, and trial construction one trial at a time.

ii. ```python
for t in range(n_trials):
    ...
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
...
for t in range(n_trials):
    zone = identify_reward_zone(...)
...
for t in range(n_trials):
    env_vals = environment[start:end]
...
for t in range(n_trials):
    ...
    neural_trials.append(...)
```

iii. The trajectory shows the agent explicitly replacing an earlier per-neuron `pearsonr` loop with a vectorized implementation because it was a bottleneck.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly loops over trials for different summaries instead of computing them in one pass: reward outcome, reward-zone identity, lick QC, environment, previous-outcome propagation, and final tensor construction are all separate passes over the same trial boundaries. It also loads fluorescence/neuropil for every session even though those arrays are not part of the final dataset.

ii. ```python
for t in range(n_trials):
    ... trial_rewarded[t] ...
for t in range(n_trials):
    ... trial_rz_label.append(...)
for t in range(n_trials):
    ... lick_error_trials.append(t)
for t in range(n_trials):
    ... trial_env[t] ...
for t in range(n_trials):
    ... build per-trial data ...
```

iii. The notes’ performance section focuses on speeding up the interneuron calculation, but the final code still keeps several separate per-trial passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is computing dF/F from fluorescence and neuropil only to detect interneurons, then discarding the dF/F entirely because the final dataset uses deconvolved events. The code also loads `planeIdx`, `scanning`, and `reward_data` without using them in the final tensors.

ii. ```python
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
...
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. The notes explicitly say the NWB files already contain the deconvolved activity needed for the decoder, so the extra dF/F computation exists only for the interneuron-quality-control step.
