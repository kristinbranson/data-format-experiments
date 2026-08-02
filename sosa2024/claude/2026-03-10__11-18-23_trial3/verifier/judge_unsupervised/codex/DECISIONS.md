# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all `.nwb` files by enumerating `data/sub-*` directories, collecting every session file, then opening each file with `h5py` and reading behavioral and ophys arrays directly from the NWB hierarchy. Sample mode hard-codes two sessions.

ii. ```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    ...
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj.replace('sub-', ''),
                'filepath': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file,
            })
```
```python
with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    position = behav['position']['data'][:]
    ...
    ophys = f['processing']['ophys']
    seg = ophys['ImageSegmentation']['PlaneSegmentation']
    ...
```

iii. `CONVERSION_NOTES.md` Step 2 says each NWB file is one session and describes behavioral plus ophys streams at imaging-frame resolution. Step 6 says the script uses NWB paths directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by `sub-*` directory name during file discovery, then recorded again from `general/subject/subject_id` when processing each session. The final dataset stores unique subject IDs in `subjects` and per-session indices in `subject_idx`.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```
```python
subject_id = f['general']['subject']['subject_id'][()].decode() ...
...
if subj not in subjects_list:
    subjects_list.append(subj)
subject_idx_list.append(subjects_list.index(subj))
```

iii. Step 2 of the notes states there are 11 subject folders and Step 10 reports 11 subjects in the converted output.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `build_dataset()` iterates over the discovered NWB files and appends one session entry each to `neural`, `input`, `output`, and `subject_idx`.

ii. ```python
for i, file_info in enumerate(nwb_files):
    ...
    neural_trials, input_trials, output_trials, session_info = process_session(
        file_info['filepath'], show_processing=show_processing, session_idx=i)
    ...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. The notes explicitly say “Each NWB file = 1 session” in Step 5 and report 152 total sessions in Steps 2, 9, and 10.

## 1-d. How are the data split into trials?

i. Trials are defined by the frame indices where `trial_start > 0` and `teleport > 0`. Starts and teleports are paired by order, truncated to equal count, filtered to keep only `teleport > trial_start`, then each trial slice is `start:end`.

ii. ```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
...
n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]
valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
```
```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    n_timepoints = end - start
    ...
    trial_neural = neural_all[start:end, :].T.copy()
```

iii. Step 1 and Step 5 of the notes say the reference uses `trial_start_inds` to `teleport_inds`, and the agent decided to mirror that with 0-based NWB indices.

## 1-e. How are trials filtered based on quality controls?

i. The code barely filters trials. It skips trials shorter than 5 frames and later skips whole sessions with fewer than 2 valid trials. Lick-sensor-error trials are not removed; only lick values for those trials are set to `NaN` and later converted to zeros.

ii. ```python
if n_timepoints < 5:
    continue  # Skip very short trials
```
```python
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
    lick_error_trials.append(t)
```
```python
if len(neural_trials) < 2:
    print(f"  WARNING: Skipping session with <2 valid trials")
    continue
```

iii. The notes describe lick-error handling as a trial-quality rule, but the implementation keeps those trials and only invalidates the lick channel. The trajectory never mentions any additional trial exclusion beyond this.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The kept neural signal is derived from NWB `processing/ophys/Deconvolved/*/data`, after filtering ROIs with `iscell` and excluding putative interneurons identified from dF/F computed from `Fluorescence` and `Neuropil`.

ii. ```python
iscell = seg['iscell'][:, 0].astype(bool)
...
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
```
```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. Step 1 of the notes says the decoder should use deconvolved events, not raw dF/F, and that dF/F is only needed for interneuron detection.

## 2-b. How is the `neural` data processed?

i. The code uses the precomputed deconvolved events directly, concatenating planes when needed. It computes dF/F only for interneuron detection. Final trial neural matrices are transposed to `(neurons, time)` and any `NaN`s are replaced with zero.

ii. ```python
if len(planes) == 1:
    deconv_data = ophys['Deconvolved']['plane0']['data'][:]
else:
    ...
    deconv_data = np.concatenate(deconv_parts, axis=1)
```
```python
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. Step 5 says “Use Deconvolved events directly” and Step 10 claims spot checks matched NWB deconvolved data exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The implemented neural QC is `iscell` plus exclusion of putative interneurons with dF/F-speed correlation `> 0.5`. The code does not mask or remove low-speed (`<2 cm/s`) neural samples, even though the notes say it should.

ii. ```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_interneuron = identify_interneurons(dff, speed, iscell)
...
final_neuron_mask[iscell_indices[non_interneuron]] = True
neural_all = deconv_data[:, final_neuron_mask]
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` say the reference excludes activity when speed `< 2 cm/s`, and Step 5 says “Speed threshold applied as masking.” That justification is not reflected in the code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by slicing each trial from `trial_start` to `teleport`, with the first column of each trial corresponding to the first trial-start frame.

ii. ```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    ...
    trial_neural = neural_all[start:end, :].T.copy()
```

iii. The instructions require trial-start alignment, and the notes repeat “Align to trial start” in Step 5 and in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code keeps the native frame resolution and does no temporal rebinning. It uses a constant frame period of `1 / 15.5078125` s for all sessions and stores `1000 / IMAGING_RATE` ms in metadata.

ii. ```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE
...
'time_bin_size': 1000.0 / IMAGING_RATE,
```

iii. Step 5 of the notes says “Time bin = 1 imaging frame” and “no rebinning.” The notes also mention two-plane animals, so using a fixed constant rather than session timestamps was a simplification.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. In the code, this input is derived from trial length and a fixed constant `FRAME_PERIOD`, not from raw timestamp arrays. The agent’s notes say it should be derived from frame timestamps relative to trial start.

ii. ```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```
```python
behav_timestamps = behav['position']['timestamps'][:]
```

iii. Step 5 says “Computed from frame timestamps relative to trial start,” but the implementation never uses `behav_timestamps` for this input.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The code creates a uniformly spaced ramp starting at 0 with one step per imaging frame. It assumes constant sampling and ignores per-session imaging-rate metadata and actual timestamps.

ii. ```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input[0, :] = time_from_start
```

iii. The notes justify this as native-frame trial-start timing, but their stated plan was to use timestamps, not a hard-coded frame period.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by giving each trial exactly the same number of time bins as the neural slice for that trial, with bin 0 corresponding to the same `trial_start` frame.

ii. ```python
n_timepoints = end - start
trial_neural = neural_all[start:end, :].T.copy()
...
trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
trial_input[0, :] = time_from_start
```

iii. The notes say the decoder data are organized per trial at frame resolution with trial-start alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii. ```python
environment = behav['environment']['data'][:]
...
env_vals = environment[start:end]
```

iii. Step 5 explicitly maps NWB `environment` to the decoder input and Step 10 says environment values were verified against the source.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the code takes all nonnegative `environment` samples within the trial, uses the median as the per-trial environment label, defaults to 0 if none are valid, and broadcasts that value across time.

ii. ```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
else:
    trial_env[t] = 0
...
trial_input[1, :] = env_val
```

iii. The notes describe environment as a per-trial binary contextual variable and report environment transitions looked correct.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The code does not derive this input from the raw `trial number` array it loads. Instead it uses the loop index `t` after pairing and filtering trial boundaries. The notes claimed it should come from NWB `trial number`.

ii. ```python
trial_number = behav['trial number']['data'][:]
...
trial_num = float(t)
trial_input[2, :] = trial_num
```

iii. Step 5 maps source `trial number` to the input, and Step 10 says trial number matched the NWB source. That is the stated justification, but not the implemented behavior.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code uses zero-based within-session trial order after start/teleport pairing, converts it to float, and broadcasts it across all time bins in the trial.

ii. ```python
trial_num = float(t)
...
trial_input[2, :] = trial_num
```

iii. No separate processing justification appears beyond the notes’ claim that this variable was sourced from NWB trial numbers.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward events in the behavioral NWB group: `Reward/timestamps`, mapped into per-trial `trial_rewarded`, then shifted by one trial.

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

iii. Step 5 says previous-trial outcome is “derived from whether preceding trial had reward events.”

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are converted to frame indices with `searchsorted`, each trial is marked rewarded if any reward falls inside its frame interval, then `prev_trial_outcome[1:]` is set to the prior trial’s reward flag. Trial 0 is left at 0.

ii. ```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
...
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. The notes describe the same shift operation and say the first trial uses 0 as “unknown/omitted.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-frame `position` plus inferred per-trial reward-zone bounds, where zone identity is inferred from `reward_zone` signal and position.

ii. ```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The notes say the NWB lacks explicit scene labels, so reward-zone location must be inferred from positions where `reward_zone > 0`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code computes signed distance to the nearest edge of the current trial’s reward zone: negative before the zone, zero inside, positive after the zone.

ii. ```python
before = position < rz_start
inside = (position >= rz_start) & (position <= rz_end)
after = position > rz_end
distance[before] = position[before] - rz_start
distance[inside] = 0.0
distance[after] = position[after] - rz_end
```

iii. Step 5 justifies this as the reward-relative spatial variable requested by the decoder spec.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into seven bins matching the task instructions: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii. ```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The notes and the task instructions use the same 7-bin scheme.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. For each trial, the position segment `position[start:end]` is converted to distance bins and written into `trial_output[0, :]`, so it shares the same frame bins as the trial neural matrix.

ii. ```python
trial_pos = position[start:end]
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
dist_bins = discretize_distance(dist_to_rz)
...
trial_output[0, :] = dist_bins
```

iii. The notes say all time-varying outputs are frame-aligned within each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii. ```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
```

iii. Step 5 maps NWB `position` directly to this decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Each per-frame position value is divided by 90 cm, floored, clipped to `[0,4]`, and stored as a 5-bin categorical position.

ii. ```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. This matches the instruction to use 5 equal-sized bins across a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are five equal 90 cm bins: `0-90`, `90-180`, `180-270`, `270-360`, `360-450`.

ii. ```python
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)
...
['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm']
```

iii. The notes repeat the same bin edges in the variable mapping table.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The code slices position over the same `start:end` trial interval used for neural data and writes one categorical value per frame.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
trial_output[1, :] = pos_bins
```

iii. The notes state all trial signals share the native frame bins.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii. ```python
lick = behav['lick']['data'][:]
...
trial_lick = lick_binary[start:end]
```

iii. The notes identify NWB `lick` as cumulative lick count and discuss binarization plus trialwise error correction.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The implemented code clips cumulative lick values to `[0,1]`, marks whole trials as invalid if more than 35% of their samples have `lick > 2`, then converts `NaN` to 0. This means after the first lick in a trial, many later frames remain 1. The notes instead say lick binarization should be `diff(cumulative_lick) > 0`.

ii. ```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)
...
frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. Step 6 of the notes says “Lick binarization: `diff(cumulative_lick) > 0` per frame,” which is a different decision from the final implementation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick channel is sliced over the same `start:end` frame interval as neural data and written into `trial_output[3, :]`.

ii. ```python
trial_lick = lick_binary[start:end]
...
trial_output[3, :] = lick_vals
```

iii. The notes say lick is handled as a framewise time-varying output aligned to trial-start-framed neural data.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` plus `position`: the agent infers which of the three fixed zone locations was active by looking at positions where `reward_zone > 0`.

ii. ```python
reward_zone_signal = behav['reward_zone']['data'][:]
position = behav['position']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
```

iii. The notes justify this because the NWB files apparently lack the scene string used in the original codebase.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the mean position of in-zone samples is compared to the centers of the predefined A/B/C ranges, the nearest label is chosen, missing labels are forward-filled from the previous known zone and optionally backward-filled at session start, then labels are mapped to `0/1/2`.

ii. ```python
in_rz = rz_trial > 0
rz_pos = pos_trial[in_rz]
mean_rz_pos = np.mean(rz_pos)
...
for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
    zone_center = (zone_start + zone_end) / 2
```
```python
last_known_zone = None
for t in range(n_trials):
    zone = identify_reward_zone(...)
    if zone is not None:
        last_known_zone = zone
    trial_rz_label.append(zone if zone is not None else last_known_zone)
...
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. Step 5 says reward-zone identity must be inferred from the reward-zone occupancy signal and mapped to A/B/C by location.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, mapped onto the imaging-frame timeline with `position/timestamps`, and reduced to a per-trial rewarded/not-rewarded flag.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
...
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
...
trial_rewarded[t] = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
```

iii. The notes explicitly describe reward outcome as event-derived rather than from a dedicated framewise state variable.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are snapped to frame indices, each trial is marked 1 if any reward event occurs inside its interval, and the scalar label is broadcast across all time bins in that trial.

ii. ```python
reward_out = int(trial_rewarded[t])
...
trial_output[5, :] = reward_out
```

iii. Step 5 and Step 10 both describe reward outcome as a per-trial binary label derived from reward events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses a set of ad hoc repairs: it truncates behavior and neural streams to a common minimum length; defaults environment to 0 if all values are invalid; forward-fills and backward-fills missing reward-zone labels and otherwise falls back to zone B; replaces `NaN` neural values with 0; and converts invalid lick trials from `NaN` back to 0 in the final output.

ii. ```python
if n_behav_samples != n_neural_samples:
    position = position[:n_samples]
    ...
    deconv_data = deconv_data[:n_samples, :]
```
```python
trial_rz_label.append(zone if zone is not None else last_known_zone)
...
trial_rz_start[t], trial_rz_end[t] = 200, 250
```
```python
trial_neural[np.isnan(trial_neural)] = 0
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes mention shape-mismatch truncation and reward-zone inference as explicit design decisions. The rest are implementation-level defaults rather than documented reference-derived behavior.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are dF/F computation for all ROIs and interneuron detection. The trajectory says the agent optimized interneuron detection because it was a major bottleneck, and after optimization dF/F remained dominant.

ii. ```python
t1 = time.time()
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
t_dff = time.time() - t1
...
t1 = time.time()
is_interneuron = identify_interneurons(dff, speed, iscell)
t_int = time.time() - t1
```

iii. Trajectory steps 73 and 80 explicitly say the per-neuron Pearson correlation and then dF/F computation were the main bottlenecks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python loops that could be vectorized or reduced: per-trial reward assignment, reward-zone inference and fills, environment summarization, lick-error scanning, and the main per-trial packing loop. The agent only vectorized interneuron detection.

ii. ```python
for t in range(n_trials):
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```
```python
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

iii. The trajectory documents one specific optimization of `identify_interneurons()` and leaves the rest unchanged.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly iterates over the same trial boundaries for reward detection, reward-zone inference, lick QC, environment extraction, previous-outcome creation, and final tensor packing. It also recomputes reward-zone distance both for outputs and again in diagnostic plotting.

ii. ```python
for t in range(n_trials):
    ...
```
```python
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
...
dist = compute_distance_to_reward_zone(position[start:end], trial_rz_start[t], trial_rz_end[t])
```

iii. This repeated trialwise structure is visible in `process_session()` and was not called out as a problem in the notes, beyond runtime discussion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded work is full-session dF/F computation, which is only used to flag interneurons and then thrown away. The script also loads unused variables (`reward_data`, `trial_number`, `scanning`, `plane_idx`), computes per-session timing only for logs, and can generate diagnostic plots not used by downstream decoding.

ii. ```python
reward_data = behav['Reward']['data'][:]
trial_number = behav['trial number']['data'][:]
scanning = behav['scanning']['data'][:]
...
plane_idx = seg['planeIdx'][:]
```
```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
...
if show_processing and session_idx < 2:
    plot_processing(...)
```

iii. The trajectory’s runtime discussion centers on dF/F and interneuron computation even though the final decoder uses deconvolved events, confirming this preprocessing is largely overhead relative to the saved dataset.
