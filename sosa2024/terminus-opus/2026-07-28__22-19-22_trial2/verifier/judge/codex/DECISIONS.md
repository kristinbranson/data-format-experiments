# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/sub-*/*.nwb`, sorts the file list, and processes each NWB file as one session. It loads NWB contents with `h5py`, not `pynwb`, and reads behavior and ophys arrays directly from HDF5 paths.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
...
with h5py.File(nwb_path, 'r') as f:
    behav = f['processing/behavior/BehavioralTimeSeries']
    ...
    planes = sorted(f['processing/ophys/Deconvolved'].keys())
    deconv_planes = []
    for plane in planes:
        deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
```

iii. In `CONVERSION_NOTES.md` the AI says the dataset is NWB files organized by subject and session and that the conversion script handles “full” processing over all 152 sessions. No additional justification for choosing `h5py` over `pynwb` was documented.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB file basename prefix, e.g. `sub-m11`, and unique subject names are accumulated in encounter order while iterating through sorted files.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
```

iii. The notes say the data are organized as `data/sub-{mouse}/...` and list 11 subjects. The trajectory shows the AI inspecting that directory structure early and then using the file prefix as the subject id.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session.

ii.
```python
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
```

iii. `CONVERSION_NOTES.md` states “Each NWB session = one session in the output.”

## 1-d. How are the data split into trials?

i. Trials start at indices where `trial_start_signal > 0` and end at indices where `teleport_signal > 0`. Each trial uses the slice `[start:stop]`.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
...
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    ...
    trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes say trial boundaries are “trial_start to teleport signals.” The trajectory repeatedly mentions adopting the paper/code convention of `trial_start` to `teleport`.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a substantive quality-control trial filter such as a minimum length threshold. It only skips trials where `stop <= start`, and later skips whole sessions with fewer than 2 surviving trials.

ii.
```python
if stop <= start:
    continue
...
if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
```

iii. No explicit trial QC justification is given in the notes beyond handling edge cases and ensuring sessions have at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is taken directly from the NWB `processing/ophys/Deconvolved/<plane>/data` arrays, then restricted by `iscell`.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
...
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The notes map “Deconvolved events” directly to `neural`, and the trajectory explicitly says “The NWB files already contain deconvolved events, so I don't need to compute dF/F.”

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, crops neural and behavior streams to a shared minimum length if needed, applies cell filtering, and then slices the stored deconvolved signal into trials. It does not recompute dF/F or redo deconvolution from fluorescence and neuropil.

ii.
```python
deconv_data = np.concatenate(deconv_planes, axis=1)
...
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
    deconv_data = deconv_data[:min_len, :]
...
neural_data = deconv_data[:, cell_indices]
...
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes say the script “Handles multi-plane sessions” and “Handles neural/behavioral length mismatches.” The trajectory says recomputing the paper’s dF/F pipeline was skipped because the NWB already stored deconvolved events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only `iscell` ROIs, then excludes putative interneurons by correlating each cell’s deconvolved activity with clipped running speed over timepoints where `trial_number >= 0`; cells with `r > 0.5` are removed.

ii.
```python
cell_mask = iscell[:, 0] == 1
...
speed_valid = speed.copy()
speed_valid[speed_valid < 0] = 0
valid_mask = trial_number >= 0
...
r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
...
interneuron_mask = speed_corr > 0.5
cell_mask = cell_mask & ~interneuron_mask
```

iii. `CONVERSION_NOTES.md` says the code uses “suite2p iscell + interneuron exclusion (speed corr > 0.5).” The trajectory acknowledges this is an approximation because the paper defined the interneuron filter on dF/F, not stored deconvolved events.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by splitting each session into `[trial_start, teleport)` slices; no extra offset is applied.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
...
'temporal_alignment_event': 'trial_start',
'off_start': 0.0,
'off_end': None,
```

iii. The notes state “Align to trial start,” and the trajectory says no extra realignment is needed beyond trial slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame resolution and sets the time bin size from the median difference of behavior timestamps, about 64.48 ms. No additional rebinning is applied.

ii.
```python
dt = np.median(np.diff(timestamps))
...
dt_ms = session_infos[0]['dt'] * 1000
...
'time_bin_size': dt_ms,
```

iii. The notes say “Using raw imaging frames (~64.48 ms), no additional temporal binning,” and list the sampling rate as about 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position/timestamps` array in the behavior group.

ii.
```python
timestamps = behav['position/timestamps'][()]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes say this input is “timestamps - trial_start_time.” No separate justification for choosing `position/timestamps` over another behavior time series’ timestamps is documented.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Within each trial, the first timestamp is subtracted from the trial’s timestamp vector.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes describe this variable as “float seconds” and the code implements the straightforward subtraction.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start:stop]` trial slice is used for both timestamps and neural activity, after any global truncation to a shared minimum length.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    timestamps = timestamps[:min_len]
    deconv_data = deconv_data[:min_len, :]
...
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says neural/behavior mismatches are handled by truncation and that temporal alignment is `trial_start` to `teleport`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the NWB `identifier` string, specifically the scene name embedded at the end of the identifier path, not from the `environment` time series.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
...
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
```

iii. The trajectory says the AI concluded that the “actual environment” could be determined from the scene name and that environment switches happened at trial 30 in matching sessions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed into pre-switch and post-switch environments, mapped as `Env1 -> 0` and `Env2 -> 1`, switched at `CHANGE_TRIAL = 30` for environment-switch sessions, and broadcast across all timepoints in each trial.

ii.
```python
CHANGE_TRIAL = 30
...
env_map = {'Env1': 0, 'Env2': 1}
...
if is_env_switch and i >= change_trial:
    env_vals.append(env_map.get(env_after, 0))
else:
    env_vals.append(env_map.get(env_before, 0))
...
np.full(n_tp, env_per_trial[i], dtype=np.float32)
```

iii. The notes say “Environment type: 0=Env1, 1=Env2,” and the trajectory reports checking sessions where the environment switch at trial 30 matched the stored data.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the raw `trial number` time series. It is the per-session loop index after splitting trials using `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. The notes say “trial index” and list trial boundaries as `trial_start` to `teleport`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The integer trial index `i` is broadcast to all timepoints in the trial.

ii.
```python
np.full(n_tp, i, dtype=np.float32)
```

iii. No extra justification beyond using trial index as a per-trial covariate is documented.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, plus trial start and stop times from the behavior timestamps and the trial boundary signals.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
trial_start_time = timestamps[start]
trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
```

iii. The notes map “reward events” to previous and current reward outcomes. The trajectory says reward omission is determined from reward events within trial windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a per-trial `is_rewarded` array, then shifts it by one trial so that each trial gets the previous trial’s outcome; the first trial is assigned 0.

ii.
```python
is_rewarded = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    ...
    is_rewarded[i] = int(reward_in_trial)
...
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
```

iii. `CONVERSION_NOTES.md` explicitly says “previous trial reward: 0=omitted, 1=rewarded.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` and from per-trial reward-zone coordinates inferred from the scene identifier, not from the raw `reward_zone` signal.

ii.
```python
position = behav['position/data'][()]
...
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
...
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The trajectory says the AI decided the `reward_zone` field encoded zone occupancy/proximity rather than A/B/C identity and therefore inferred the actual zone from the session scene name and trial number.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside it, positive after it.

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

iii. The notes map “position - reward zone” to the distance output. No separate justification beyond the task definition is given.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into seven bins using explicit comparisons at `-50`, `-10`, `0`, `10`, and `50` cm, with an exact zero class.

ii.
```python
def discretize_distance(dist):
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins
```

iii. The notes say the variable is “7 bins,” matching the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the same trial slice `[start:stop]`, so distance is aligned frame-by-frame to the neural data.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes describe all time-varying inputs and outputs as being aligned to trial start with the native frame rate.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `position/data`.

ii.
```python
position = behav['position/data'][()]
...
trial_pos = position[start:stop].astype(np.float32)
```

iii. The notes map “position” directly to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The position trace is sliced per trial, converted to `float32`, clipped into `[0, 450]` cm, and then discretized.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
...
pos_disc = discretize_position(trial_pos)
```

iii. The trajectory and notes emphasize the 450 cm track length. No further discussion of the clipping choice is documented.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is discretized into 5 equal-width bins across the 0 to 450 cm track using `np.digitize`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The notes explicitly say “5 equal bins (90 cm each).”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same per-trial `[start:stop]` slice is used for both position and neural data.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop].astype(np.float32)
```

iii. Alignment follows the common session timebase and trial slicing described in the notes.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `lick/data`.

ii.
```python
lick = behav['lick/data'][()]
...
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. The notes map “lick (binary)” to the lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies a lick-sensor error heuristic per trial: if more than 35% of samples exceed 2, it marks the trial’s lick values as `NaN`. It then caps lick values above 1 to 1, converts `NaN` to 0, and outputs a binary `trial_lick > 0`.

ii.
```python
lick_corrected = lick.copy()
for i in range(n_trials):
    ...
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > 0.35:
        lick_corrected[start:stop] = np.nan
...
lick_binary = lick_corrected.copy()
lick_binary[lick_binary > 1] = 1
lick_binary[np.isnan(lick_binary)] = 0
...
lick_disc = (trial_lick > 0).astype(int)
```

iii. `CONVERSION_NOTES.md` says this follows the reference code’s lick sensor error correction and treats erroneous licks as 0 for the decoder.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned to neural activity by taking the same `[start:stop]` trial slice.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. The notes treat lick as a time-varying variable on the same imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene name, not from the `reward_zone` time series itself.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
```

iii. The trajectory explicitly says the AI concluded that the actual zone location “needs to be inferred from the session scene name.”

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into pre-switch and post-switch zone labels. The script assigns the first label for trials `< 30` and the second label for trials `>= 30` in switch sessions, then maps `A/B/C` to `0/1/2`.

ii.
```python
def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    ...
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
...
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The notes say “scene name” maps to reward zone location, and the trajectory reports validating that `change_trial = 30` matched the observed switch sessions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, combined with trial time windows.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
...
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
```

iii. The notes map “reward events” to reward outcome and discuss the ~15% omission rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls between the trial’s start and end times. That binary value is then broadcast across all timepoints in the trial.

ii.
```python
is_rewarded = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    ...
    is_rewarded[i] = int(reward_in_trial)
...
np.full(n_tp, reward_out, dtype=int)
```

iii. `CONVERSION_NOTES.md` says reward outcome is per-trial and binary, with random omission on about 15% of trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles neural/behavior length mismatches by truncating to the shorter stream; mismatched counts of trial starts and teleports by truncating both lists to the smaller count; negative speeds by clipping to 0; lick-sensor-error trials by turning them into zeros after temporary `NaN`; and zero/negative-length trials by skipping them.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]
...
speed_valid[speed_valid < 0] = 0
...
lick_binary[np.isnan(lick_binary)] = 0
...
if stop <= start:
    continue
```

iii. The notes explicitly list multi-plane handling, neural/behavior mismatch truncation, lick sensor errors, and unusual trial lengths as handled edge cases.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s expensive steps are loading large NWB arrays, concatenating/processing full-session deconvolved data, the per-cell speed-correlation loop for interneuron exclusion, the per-trial construction loops, and writing the large pickle.

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

iii. The notes report about 1.3 s per session and a 9+ GB output file, implying I/O and whole-dataset serialization are major costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell correlation loop for interneuron exclusion, the per-trial reward detection loop, the per-trial lick-correction loop, and parts of the final per-trial assembly loop could have been vectorized or consolidated.

ii.
```python
for c in range(deconv_data.shape[1]):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    ...
```

iii. No explicit vectorization discussion appears in the notes, but these loops are the obvious CPU-side repeated operations in `process_session`.

## 13-c. What processing does the code repeat multiple times?

i. It makes multiple passes over the same trials within each session: once to determine reward outcome, once to correct lick traces, once to build trial tensors, and again later to compute summary distributions for printing. It also reuses the same parsed session metadata in several places.

ii.
```python
for i in range(n_trials):
    ...  # reward detection
for i in range(n_trials):
    ...  # lick correction
for i in range(n_trials):
    ...  # build neural/input/output trials
...
for sess_outputs in all_output:
    for trial_output in sess_outputs:
        all_vals.extend(trial_output[out_idx].tolist())
```

iii. The notes do not call this out, but the repeated passes are visible in the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `reward_zone_signal`, `environment`, and `trial_number` arrays but does not use them directly for the corresponding final decoded variables; it also computes extensive plotting and summary/reporting machinery that is only for sanity checks, not for the saved dataset.

ii.
```python
reward_zone_signal = behav['reward_zone/data'][()]
environment = behav['environment/data'][()]
trial_number = behav['trial number/data'][()]
...
if show_processing:
    plot_processing(...)
...
for out_idx, out_name in enumerate(data['output_names']):
    all_vals = []
```

iii. The trajectory shows the AI chose scene parsing instead of using the loaded `reward_zone` and `environment` signals directly, so those arrays are partly exploratory overhead rather than required conversion inputs.
