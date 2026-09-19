# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories named `sub-*`, then scans each subject directory for `.nwb` files. Each `.nwb` file is processed as one session. It reads the files directly with `h5py` and then pulls behavioral and ophys arrays from the NWB/HDF5 hierarchy.

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
    ...
    ophys = f['processing']['ophys']
```

iii. `CONVERSION_NOTES.md` says the dataset is organized as `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb` and that each file contains one session. The notes justify this by the observed directory structure and session counts, but they describe the loader as NWB-aware preprocessing while the final script actually uses raw `h5py`.

## 1-b. How are the data split into subjects?

i. Subjects are defined by directory names under `data` that start with `sub-`. The stored subject id for the output dataset comes from each session’s NWB metadata, but the file discovery step uses the directory layout.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```
```python
subject_id = f['general']['subject']['subject_id'][()].decode() if isinstance(
    f['general']['subject']['subject_id'][()], bytes) else str(f['general']['subject']['subject_id'][()])
```

iii. The notes justify this by listing 11 `sub-*` directories and matching them to the 11 switch-condition mice described in the paper.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The output dataset appends one session entry per file processed.

ii. ```python
for nwb_file in nwb_files:
    all_files.append({
        'subject': subj.replace('sub-', ''),
        'filepath': os.path.join(subj_dir, nwb_file),
        'filename': nwb_file,
    })
```
```python
for i, file_info in enumerate(nwb_files):
    ...
    neural_trials, input_trials, output_trials, session_info = process_session(
        file_info['filepath'], show_processing=show_processing, session_idx=i)
```

iii. The notes explicitly say “Each file contains one session for one mouse,” which is the justification the AI gives for this split.

## 1-d. How are the data split into trials?

i. Trials are defined from frames where `trial_start > 0` to frames where `teleport > 0`. The code pairs those arrays by truncating them to equal length and then dropping any pair where the teleport index is not after the start index.

ii. ```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]

valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
```

iii. The notes justify trial boundaries as “trial_start to teleport indices” and say teleport periods should be excluded. There is no script-side justification for using every positive `teleport` sample instead of teleport onset edges.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial-length quality control in the final script is to skip trials shorter than 5 frames. Sessions with fewer than 2 remaining trials are then skipped entirely.

ii. ```python
if n_timepoints < 5:
    continue  # Skip very short trials
```
```python
if len(neural_trials) < 2:
    print(f"  WARNING: Skipping session with <2 valid trials")
    continue
```

iii. There is no explicit justification in the code or trajectory for the 5-frame cutoff. The notes discuss short-trial filtering in general but do not explain why the final script uses `5`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved `neural` matrices come from the NWB `Deconvolved` arrays after neuron selection. The script also loads `Fluorescence` and `Neuropil`, but only to compute dF/F for interneuron detection, not for the saved neural output.

ii. ```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
```
```python
neural_all = deconv_data[:, final_neuron_mask]
```

iii. `CONVERSION_NOTES.md` explicitly claims the NWB files already contain the needed deconvolved activity and says “we use that directly.” The same notes also acknowledge that dF/F still has to be computed for the interneuron filter.

## 2-b. How is the `neural` data processed?

i. The saved neural signal is not recomputed from fluorescence. Instead, the script loads the precomputed deconvolved events, filters neurons, splits them into trials, transposes to `(neurons, time)`, and replaces any NaNs with zero. dF/F is computed separately only for the interneuron exclusion step.

ii. ```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
neural_all = deconv_data[:, final_neuron_mask]
```
```python
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The notes justify this by claiming NWB `Deconvolved` “matches the reference pipeline” and by using dF/F only “for interneuron check.” That justification conflicts with the human reference pipeline, which recomputes the events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two steps: keep only ROIs with `iscell == 1`, then exclude putative interneurons whose dF/F is correlated with running speed above 0.5.

ii. ```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_interneuron = identify_interneurons(dff, speed, iscell)
...
iscell_indices = np.where(iscell)[0]
non_interneuron = ~is_interneuron
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
```
```python
is_interneuron = r > INTERNEURON_CORR_THRESHOLD
```

iii. The notes cite the paper’s neuron curation rules: manual `iscell` curation plus interneuron exclusion at Pearson `r > 0.5` with speed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned simply by slicing each trial from the same `trial_start`/`teleport` indices used for the behavioral variables. No additional temporal shifting is applied.

ii. ```python
start = trial_starts[t]
end = teleports[t]
...
trial_neural = neural_all[start:end, :].T.copy()
```

iii. The notes justify this by stating that the decoder should be aligned to the start of each trial and that the NWB arrays are already frame-aligned across modalities.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use one imaging frame per time bin with a hard-coded frame period of `1 / 15.5078125` s, stored as `1000 / 15.5078125` ms in metadata. No temporal rebinning is applied.

ii. ```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
```
```python
'time_bin_size': 1000.0 / IMAGING_RATE,
```

iii. The notes justify this as the native per-plane temporal resolution of the recordings, including the two-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The final script derives this input from trial length and a fixed frame period, not from NWB timestamps. The raw ingredients are the trial boundaries plus the hard-coded imaging rate.

ii. ```python
n_timepoints = end - start
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The notes claim this should match the imaging-frame timing and describe the time bin as one imaging frame. The notes do not acknowledge that the final script stopped using behavior timestamps here.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code constructs an evenly spaced vector starting at 0 and increasing by the constant frame period at each sample.

ii. ```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
...
trial_input[0, :] = time_from_start
```

iii. The implicit justification is that sampling is regular at the imaging frame rate. No additional justification is given in the final script.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector has exactly one element per sliced neural frame in the same trial.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
...
trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
trial_input[0, :] = time_from_start
```

iii. The notes justify this using the claim that behavior and neural streams are already frame-aligned and, when necessary, are truncated to a shared length before trial slicing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii. ```python
environment = behav['environment']['data'][:]
```

iii. The notes identify the NWB `environment` variable as `0=ENV1, 1=ENV2`, which is the stated justification.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the code takes all nonnegative environment values in that trial, uses their median as the trial label, and broadcasts that scalar across all timepoints in the trial.

ii. ```python
env_vals = environment[start:end]
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
...
trial_input[1, :] = env_val
```

iii. The notes justify this by saying environment is a per-trial binary context and is effectively constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over the trial slices, which themselves come from `trial_start` and `teleport`.

ii. ```python
for t in range(n_trials):
    ...
    trial_num = float(t)
```

iii. The notes justify this as the within-session trial index after trial segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code converts the trial index to a scalar and broadcasts it across all timepoints in the trial.

ii. ```python
trial_num = float(t)
...
trial_input[2, :] = trial_num
```

iii. No additional justification is given beyond treating trial number as a per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` event timestamps, aligned to behavioral frame indices using the behavioral timestamps.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
...
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
```

iii. The notes justify this by saying reward is stored as an event stream with its own timestamps and must be mapped onto the frame-based trial structure.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first marks each trial as rewarded or not by checking whether any mapped reward event falls inside its `[start, end)` frame range. It then sets `prev_trial_outcome[t]` to the previous trial’s reward flag and broadcasts that across the current trial.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```
```python
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
...
trial_input[3, :] = prev_outcome
```

iii. The notes explicitly justify the first trial being set to `0` because there is no previous trial, and they interpret the variable as binary rewarded vs omitted.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` time series together with a per-trial reward-zone label inferred from the `reward_zone` time series. The script infers the reward-zone label from the mean position of frames where `reward_zone > 0`, then carries the last known label forward if a trial has no reward-zone occupancy.

ii. ```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
```
```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]
    ...
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
```
```python
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
if zone is not None:
    last_known_zone = zone
trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. The notes justify this by saying the NWB files lack the original scene metadata, so reward-zone identity must be inferred from where `reward_zone > 0` occurs in position space.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial’s reward-zone boundaries are chosen, the code computes signed distance to the nearest edge: negative before the zone, zero inside it, positive after it.

ii. ```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end
    return distance
```

iii. The notes justify this as the decoder-task definition of distance relative to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code manually thresholds the continuous distance into the 7 instructed bins.

ii. ```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The justification in the notes is that these are exactly the task-specified decoder bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing position from the same trial frame interval as the neural data and computing the distance on that slice.

ii. ```python
trial_pos = position[start:end]
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
trial_output[0, :] = dist_bins
```

iii. The notes justify this by treating the behavioral and neural arrays as already aligned frame-by-frame.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii. ```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
```

iii. The notes justify this as the VR corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code slices position per trial and discretizes it into five 90 cm bins using `floor(position / 90)` clipped to the range `[0, 4]`.

ii. ```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```
```python
pos_bins = discretize_position(trial_pos)
```

iii. The notes justify this from the 450 cm track length and the decoder requirement of 5 equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The five categories are the clipped 90 cm bins produced by `discretize_position`.

ii. ```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The notes justify this as equal-size bins across the 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by using the same `[start:end)` frame slice as the neural trial.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
trial_output[1, :] = pos_bins
```

iii. The notes again justify this with the shared frame indexing of the arrays.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii. ```python
lick = behav['lick']['data'][:]
...
trial_lick = lick_binary[start:end]
```

iii. The notes identify the NWB lick signal as the source and describe it as a cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script clips the lick signal to `0/1`, marks trials with too many `lick > 2` samples as lick-error trials by setting those frames to `NaN`, and then converts `NaN` back to `0` in the final output array.

ii. ```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)
...
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
```
```python
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
trial_output[3, :] = lick_vals
```

iii. The notes justify this as following the reference lick-error correction heuristic and then producing the binary decoder output required by the task.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the per-frame lick vector on the same trial interval as the neural data.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_lick = lick_binary[start:end]
trial_output[3, :] = lick_vals
```

iii. The notes justify this using the same shared-frame alignment argument as for position and distance.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the `reward_zone` behavior signal plus the `position` signal, using the same per-trial inference described for distance to reward zone.

ii. ```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
```

iii. The notes justify this because the NWB files do not contain the original scene labels used in the reference code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The script classifies each trial into A/B/C by comparing the mean position of reward-zone occupancy to the zone centers, then uses forward fill and one backward fill pass for missing trials. It then encodes `A/B/C` as `0/1/2`.

ii. ```python
for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
    zone_center = (zone_start + zone_end) / 2
    dist = abs(mean_rz_pos - zone_center)
    if dist < best_dist:
        best_dist = dist
        best_zone = zone_name
```
```python
if zone is not None:
    last_known_zone = zone
trial_rz_label.append(zone if zone is not None else last_known_zone)
...
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. The notes justify this as a necessary inference step from observed reward-zone occupancy in the NWB representation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` event timestamps aligned to the behavior frame timestamps.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
```

iii. The notes justify this because reward delivery is event-based rather than stored as a per-frame binary vector.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The script marks each trial as rewarded if any mapped reward event falls inside that trial’s frame interval, then broadcasts that binary label across all timepoints of the trial output.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```
```python
reward_out = int(trial_rewarded[t])
trial_output[5, :] = reward_out
```

iii. The notes justify this as the requested binary trial outcome: rewarded vs omitted.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script applies several ad hoc repairs: truncate neural and behavioral arrays to the same minimum length; truncate unmatched `trial_start`/`teleport` arrays to the smaller count; drop start/end pairs where `teleport <= trial_start`; forward-fill or backfill missing reward-zone labels, with a final fallback to zone B; mark lick-error trials and then convert those NaNs to zeros; replace neural NaNs with zeros; skip very short trials; and skip sessions with fewer than 2 kept trials.

ii. ```python
n_samples = min(n_behav_samples, n_neural_samples)
if n_behav_samples != n_neural_samples:
    ...
    deconv_data = deconv_data[:n_samples, :]
```
```python
n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]
valid = teleports > trial_starts
```
```python
if trial_rz_label[0] is None:
    for t in range(n_trials):
        if trial_rz_label[t] is not None:
            for tt in range(t):
                trial_rz_label[tt] = trial_rz_label[t]
            break
...
trial_rz_start[t], trial_rz_end[t] = 200, 250
```
```python
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The notes explicitly justify the length truncation and lick-error handling. For missing reward zones, the notes say the source data require inference because scene labels are unavailable, but they do not justify the final forward-fill/backfill heuristic in detail.

## 13-a. What are the most time-consuming steps of the code?

i. The code itself measures file loading, dF/F computation, and interneuron detection per session, and the notes say dF/F became the main bottleneck after vectorizing interneuron detection. Full conversion is therefore dominated by NWB I/O plus dF/F/interneuron preprocessing.

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
```python
'load_time': t_load,
'dff_time': t_dff,
'interneuron_time': t_int,
'total_time': t_total,
```

iii. The trajectory explicitly says “The main dFF bottleneck is the per-trial filtering” and that vectorizing interneuron detection reduced that step from 4.3 s to 0.4 s on one session.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The script already vectorizes the interneuron correlation calculation, but it still has multiple serial loops over trials: reward detection, reward-zone inference, lick-error detection, environment extraction, previous-outcome construction, and trial packaging. Some of those loops could be merged or partially vectorized.

ii. ```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```
```python
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
    ...
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    ...
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    n_timepoints = end - start
    ...
```

iii. The trajectory shows the AI already recognized `identify_interneurons()` as a vectorization target and left the dF/F per-trial loop as the main remaining bottleneck.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats several passes over the same trial structure. It scans trials once to decide reward outcome, again to infer reward zone, again for lick-error detection, again for environment extraction, and again to build the final trial tensors. It also loads fluorescence/neuropil and computes dF/F even though the final saved neural data come from `Deconvolved`.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    ...
```
```python
for t in range(n_trials):
    zone = identify_reward_zone(...)
```
```python
for t in range(n_trials):
    ...
    if frac_bad > LICK_ERROR_FRACTION:
        ...
```
```python
for t in range(n_trials):
    ...
    env_vals = environment[start:end]
```
```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. The notes justify the dF/F pass as necessary for interneuron filtering, but they do not try to collapse the multiple later trial loops.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest discarded computation is the full-session dF/F calculation, which is used only to exclude interneurons and is not saved. The script also loads or computes several arrays that never affect the final dataset, including `reward_data`, `scanning`, `plane_idx`, the unused bin-edge constants, and the placeholder `rz_counts` in plotting.

ii. ```python
reward_data = behav['Reward']['data'][:]
...
scanning = behav['scanning']['data'][:]
...
plane_idx = seg['planeIdx'][:]
```
```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
...
neural_all = deconv_data[:, final_neuron_mask]
```
```python
SPEED_BINS = [0, 2, 10, 20, 40, np.inf]
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)
DIST_BINS = [-np.inf, -50, -10, 0, 0, 10, 50, np.inf]
```

iii. The notes explicitly say the NWB files already contain deconvolved activity, which makes the retained dF/F computation purely an internal QC/filtering step rather than part of the downstream saved representation.
