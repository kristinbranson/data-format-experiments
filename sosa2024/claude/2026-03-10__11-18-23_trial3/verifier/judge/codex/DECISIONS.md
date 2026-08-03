# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans `/app/data` for subdirectories named `sub-*`, then scans each subject directory for `.nwb` files. Each NWB file is treated as one session and is opened directly with `h5py`; session processing is then done by `process_session()`, and dataset assembly by `build_dataset()`.

ii.
```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

    all_files = []
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
    ...
```

```python
for i, file_info in enumerate(nwb_files):
    neural_trials, input_trials, output_trials, session_info = process_session(
        file_info['filepath'], show_processing=show_processing, session_idx=i)
```

iii. `CONVERSION_NOTES.md` says the dataset is organized as `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`, with 11 subjects and 152 sessions, and treats each NWB file as a session. The notes justify direct NWB access as sufficient because the files already contain the required behavioral and ophys arrays.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `sub-*` directory names, and the final dataset’s `subjects`/`subject_idx` are built from the per-session `subject_id` metadata returned by `process_session()`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```

```python
subject_id = f['general']['subject']['subject_id'][()].decode() if isinstance(
    f['general']['subject']['subject_id'][()], bytes) else str(f['general']['subject']['subject_id'][()])
```

```python
subj = session_info['subject']
if subj not in subjects_list:
    subjects_list.append(subj)
subject_idx_list.append(subjects_list.index(subj))
```

iii. The notes state the data are arranged by mouse subdirectory and list the 11 switch-task mice explicitly, so directory-level subject parsing was the AI’s main split criterion.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The full dataset is a list over files, with one session appended per processed file.

ii.
```python
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
for i, file_info in enumerate(nwb_files):
    ...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. `CONVERSION_NOTES.md` explicitly says “Each file contains one session for one mouse” and “Each NWB file = 1 session in the output data structure.”

## 1-d. How are the data split into trials?

i. Trials are split by taking every frame where `trial_start > 0` as a start index and every frame where `teleport > 0` as an end index, then pairing them in order and keeping only pairs where `teleport > start`.

ii.
```python
trial_start_signal = behav['trial_start']['data'][:]
teleport_signal = behav['teleport']['data'][:]
...
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]

valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
```

iii. The notes say trial boundaries should be `trial_start` to `teleport` and describe teleport periods as excluded. The implementation uses the raw positive `teleport` samples rather than a teleport onset detector.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with at least 5 frames. Sessions with fewer than 2 surviving trials are discarded from the final dataset.

ii.
```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    n_timepoints = end - start

    if n_timepoints < 5:
        continue  # Skip very short trials
```

```python
if len(neural_trials) < 2:
    print(f"  WARNING: Skipping session with <2 valid trials")
    continue
```

iii. The notes emphasize preserving sessions that can support decoder evaluation and mention edge-case handling for very short trials, but the concrete threshold in code is 5 frames rather than the 50-frame threshold used in the human reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` output is taken from `processing/ophys/Deconvolved/.../data`, but the neuron-selection mask is also derived from `ImageSegmentation/.../iscell` and from dF/F computed from `Fluorescence` and `Neuropil`.

ii.
```python
seg = ophys['ImageSegmentation']['PlaneSegmentation']
iscell = seg['iscell'][:, 0].astype(bool)
```

```python
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

iii. The notes say the NWB files already contain precomputed deconvolved events and also say dF/F should be recomputed only for interneuron filtering. The implementation follows that hybrid rationale.

## 2-b. How is the `neural` data processed?

i. The AI concatenates across planes for multi-plane recordings, computes dF/F from fluorescence plus neuropil subtraction, identifies interneurons by correlation with speed, removes those neurons, extracts deconvolved activity for the remaining neurons, slices it per trial, transposes to `(neurons, time)`, and replaces NaNs with zero.

ii.
```python
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][:])
    fluor_parts.append(ophys['Fluorescence'][plane]['data'][:])
    neuro_parts.append(ophys['Neuropil'][plane]['data'][:])
deconv_data = np.concatenate(deconv_parts, axis=1)
fluor_data = np.concatenate(fluor_parts, axis=1)
neuropil_data = np.concatenate(neuro_parts, axis=1)
```

```python
F_corr = F - NEUROPIL_COEF * Fneu
...
smoothed = gaussian_filter1d(trial_F, sigma=15, axis=0)
min_filtered = minimum_filter1d(smoothed, size=BASELINE_WINDOW, axis=0)
baseline = maximum_filter1d(min_filtered, size=BASELINE_WINDOW, axis=0)
trial_dff = (trial_F - baseline) / abs_baseline
```

```python
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. `CONVERSION_NOTES.md` frames this as matching paper-style preprocessing: use stored deconvolved events as neural data but recompute dF/F from fluorescence/neuropil to remove putative interneurons. The trajectory shows the AI then optimized dF/F and interneuron detection because they became the main runtime bottlenecks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `iscell`, then further filtered by an interneuron detector that marks a neuron as interneuron if its dF/F has Pearson correlation `> 0.5` with running speed.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
```

```python
r = cov_XY / (std_X * std_Y)
is_interneuron = r > INTERNEURON_CORR_THRESHOLD
```

```python
iscell_indices = np.where(iscell)[0]
non_interneuron = ~is_interneuron
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
```

iii. The notes justify this by citing the paper’s interneuron exclusion rule and describe computing dF/F only to support this filter. The trajectory also highlights matching the paper’s reported interneuron fraction as a sanity check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start implicitly by cutting each trial from `start:end`, where `start` comes from the `trial_start` signal. No additional temporal shifting is applied.

ii.
```python
start = trial_starts[t]
end = teleports[t]
...
trial_neural = neural_all[start:end, :].T.copy()
```

iii. The notes state “Temporal alignment: Align to trial start (first frame of each trial)” and the metadata later describes the alignment event as trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses one native imaging frame per time bin, with a fixed frame period of `1 / 15.5078125 s` (`~64.5 ms`). No temporal rebinning or resampling is applied.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
```

```python
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
```

iii. The notes say “Time bin = 1 imaging frame: ~64.5 ms. This matches the native temporal resolution,” including for two-plane recordings after pooling neurons across planes.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. In code, this input is derived from the trial length and a hard-coded frame period; it does not use the NWB behavioral timestamps even though those are loaded.

ii.
```python
behav_timestamps = behav['position']['timestamps'][:]
```

```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The notes had planned to use “frame timestamps relative to trial start,” but the implementation simplifies that to a fixed-rate clock using `FRAME_PERIOD`.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI constructs a regularly spaced vector `0, FRAME_PERIOD, 2*FRAME_PERIOD, ...` and writes that into the first input row.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
...
trial_input[0, :] = time_from_start
```

iii. The notes justify one-frame bins at a constant imaging rate, so the implementation uses sample index times rather than subtracting per-trial timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector is created with exactly `n_timepoints = end - start`, the same length as the per-trial neural slice.

ii.
```python
n_timepoints = end - start
trial_neural = neural_all[start:end, :].T.copy()
...
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
```

iii. The notes state the dataset should use one imaging frame as the common time base, so the AI aligned time and neural data by shared trial slicing and shared frame count.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
environment = behav['environment']['data'][:]
```

```python
env_vals = environment[start:end]
```

iii. The notes identify NWB `environment` as the source variable and describe it as `0=ENV1, 1=ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median of nonnegative `environment` values within each trial, then broadcasts that constant value across all time bins in the trial.

ii.
```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
else:
    trial_env[t] = 0
```

```python
env_val = float(trial_env[t])
trial_input[1, :] = env_val
```

iii. The notes say environment is effectively per-trial, and the code uses a robust per-trial summary instead of preserving the raw framewise series.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the raw NWB `trial number` series. It is derived from the loop index over trial segments found from `trial_start` and `teleport`.

ii.
```python
trial_number = behav['trial number']['data'][:]
```

```python
for t in range(n_trials):
    ...
    trial_num = float(t)
```

iii. The notes explicitly say “Trial number: 0-indexed trial within session,” and the AI relied on its own trial segmentation rather than the stored `trial number` array.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond assigning the 0-based loop index and broadcasting it across the trial.

ii.
```python
trial_num = float(t)
...
trial_input[2, :] = trial_num
```

iii. This follows the notes’ interpretation of trial number as a within-session sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event timestamps after mapping those timestamps onto frame indices using behavioral timestamps.

ii.
```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
...
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
```

```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```

iii. The notes say reward outcome and previous-trial outcome should come from reward events mapped to frame indices.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a per-trial `trial_rewarded` flag, then sets `prev_trial_outcome[t] = trial_rewarded[t-1]` and broadcasts that value across each trial. The first trial is set to `0`.

ii.
```python
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

```python
prev_outcome = float(prev_trial_outcome[t])
trial_input[3, :] = prev_outcome
```

iii. The notes describe this exactly: previous-trial outcome is binary and defined by whether the preceding trial had reward delivery.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` behavioral time series plus a per-trial reward-zone label inferred from the `reward_zone` signal. The inferred label comes from the mean position of samples where `reward_zone > 0`, with missing trials filled from neighboring detected zones.

ii.
```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
```

```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]
    in_rz = rz_trial > 0
    if not np.any(in_rz):
        return None
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    ...
```

```python
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
    if zone is not None:
        last_known_zone = zone
    trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. The notes justify reward-zone inference because the NWB files do not contain the original scene strings used by the reference repo. The notes say the AI decided to infer zones from position where `reward_zone > 0`; unlike the human reference, the implementation does not perform a separate survey-plus-Viterbi assignment.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial’s reward-zone start and end are set, the AI computes signed distance to the nearest edge: negative before the zone, zero inside, positive after.

ii.
```python
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

```python
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The notes describe the same signed-distance interpretation and list A/B/C reward-zone coordinates as fixed constants.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded manually into 7 bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
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

iii. The binning follows the decoder-task categories recorded in the notes.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI uses the same `start:end` trial slice for `position` as for `neural`, then computes the distance vector on that slice.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The notes describe all behavioral and neural outputs as being put on the same framewise trial grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
```

iii. The notes identify `position` as the NWB variable for absolute VR corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI discretizes per-frame position into five equal 90 cm bins spanning a 450 cm track, using `floor(position / 90)` clipped to `[0,4]`.

ii.
```python
TRACK_LENGTH = 450.0
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)
```

```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

```python
pos_bins = discretize_position(trial_pos)
```

iii. The notes explicitly justify five equal bins of 90 cm each from the 450 cm track length, matching the task instructions more literally than the human reference.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm, implemented by the same `floor(position / 90)` rule.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

```python
'output_values': [
    ...
    ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
    ...
]
```

iii. The notes say the corridor is 450 cm long and therefore use equal 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same `start:end` trial slice as neural activity, so alignment is by shared frame index.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
```

iii. The notes consistently treat the per-frame behavior streams as already aligned to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick']['data'][:]
```

```python
trial_lick = lick_binary[start:end]
```

iii. The notes identify `lick` as an NWB behavioral stream and describe it as a cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI clips the raw lick signal to `[0,1]`, marks trials with too many samples where `lick > 2` as lick-error trials by setting them to `NaN`, and then converts NaNs to `0` when building the output.

ii.
```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)
```

```python
frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
```

```python
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes justify this from the paper’s lick-error rule and say the AI intended to correct trials with obvious lick-sensor problems. The notes also mention lick binarization, although the exact implementation in code is clipping rather than differentiating the cumulative count.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The AI slices `lick_binary[start:end]` using the same trial boundaries as neural activity.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()
trial_lick = lick_binary[start:end]
```

iii. The notes treat lick as a frame-aligned behavioral stream sampled on the same trial grid as the neural data.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the `reward_zone` behavioral stream together with `position`, using the same per-trial zone inference described in 7-a.

ii.
```python
reward_zone_signal = behav['reward_zone']['data'][:]
position = behav['position']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
```

```python
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. The notes say the NWB files lack the original scene labels, so the AI inferred reward-zone identity from where `reward_zone > 0` occurred in position space.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the AI maps the mean in-zone position to the nearest of A/B/C, forward-fills missing trials from the last detected zone, backward-fills leading missing trials from the first detected zone, and otherwise falls back to zone B.

ii.
```python
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

iii. The notes describe inferring reward-zone identity from the reward-zone occupancy signal, but the implemented fill strategy is simpler than the Viterbi-based smoothing used by the human reference.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward` event timestamps mapped into the frame index space.

ii.
```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
```

```python
trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
trial_rewarded[t] = trial_rewards
```

iii. The notes state reward outcome should come from reward-event timing, not from the `reward_zone` occupancy signal.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI marks each trial as rewarded if any mapped reward event falls between that trial’s `start` and `end`, then broadcasts that binary value across all time bins of the trial in `output[5]`.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```

```python
reward_out = int(trial_rewarded[t])
...
trial_output[5, :] = reward_out
```

iii. The notes justify reward outcome as a per-trial binary variable driven by reward delivery events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases heuristically: it truncates neural and behavioral streams to their shared minimum length, clips reward-event frame indices into range, skips trials with fewer than 5 frames, forward/backward-fills missing reward-zone labels and falls back to zone B if still missing, defaults missing environment labels to 0, and converts NaNs in neural or lick data to 0 before output.

ii.
```python
if n_behav_samples != n_neural_samples:
    ...
    position = position[:n_samples]
    ...
    deconv_data = deconv_data[:n_samples, :]
```

```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
```

```python
if n_timepoints < 5:
    continue
...
trial_neural[np.isnan(trial_neural)] = 0
...
trial_env[t] = 0
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes and trajectory show the AI actively adding defensive handling for mismatched lengths, missing reward-zone occupancy, and bad lick trials so that conversion and decoder training would still complete.

## 13-a. What are the most time-consuming steps of the code?

i. In the AI’s implementation, the most time-consuming steps are reading large NWB arrays, computing dF/F for every session, and running interneuron detection. The trajectory explicitly says dF/F and interneuron detection became the main bottlenecks.

ii.
```python
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

iii. The trajectory contains multiple runtime-focused messages such as “The main dFF bottleneck is the per-trial filtering” and “Interneuron detection went from 4.3s to 0.4s,” showing that these steps dominated the AI’s profiling effort.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized interneuron correlation, but several loops remain scalar or trial-by-trial: `compute_dff()` still iterates over trials, reward assignment loops over trials, reward-zone identification loops over trials, lick-error detection loops over trials, and environment extraction loops over trials.

ii.
```python
for i, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

```python
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
```

iii. The trajectory says the AI specifically vectorized interneuron detection but found dF/F harder to vectorize further because it is done per trial.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly scans trial boundaries for several separate passes: reward detection, reward-zone inference, lick-error detection, environment extraction, previous-trial outcome construction, and finally trial assembly. It also reads fluorescence, neuropil, and deconvolved data even though only deconvolved activity is kept in the output.

ii.
```python
for t in range(n_trials):
    ...
    trial_rewarded[t] = trial_rewards
```

```python
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
    ...
```

```python
for t in range(n_trials):
    ...
    if frac_bad > LICK_ERROR_FRACTION:
        lick_binary[start:end] = np.nan
```

```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    ...
    neural_trials.append(...)
```

iii. The notes and trajectory show the AI preferred multiple simpler per-trial passes over a single integrated pass, partly because it was validating each behavioral/neural output separately.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded computation is dF/F from fluorescence and neuropil: it is computed for every session only to support interneuron filtering, then the final saved neural data are still the original deconvolved events. The code also loads arrays that are not used downstream (`trial number`, `scanning`, `reward_data`, `plane_idx`) and keeps timing/debug metadata only for reporting.

ii.
```python
trial_number = behav['trial number']['data'][:]
scanning = behav['scanning']['data'][:]
reward_data = behav['Reward']['data'][:]
...
plane_idx = seg['planeIdx'][:]
```

```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. The notes explicitly say the NWB already contains precomputed deconvolved activity, so recomputing dF/F is auxiliary work that is not itself written to the output. The trajectory confirms this auxiliary processing became a large runtime cost.
