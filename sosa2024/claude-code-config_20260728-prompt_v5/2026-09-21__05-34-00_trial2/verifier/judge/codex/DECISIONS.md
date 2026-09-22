# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every NWB file under `/app/data/sub-*/*.nwb`, sorts the paths, and processes each file independently. It opens each NWB file directly with `h5py.File` rather than `pynwb`.

ii.
```python
def get_all_nwb_files():
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    return files

for fpath in all_files:
    result = process_session(fpath, ...)

with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as using all provided switch-task mice and all NWB sessions, and treated the NWB files as already containing the needed aligned arrays.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are effectively split by the `sub-*` directory structure and by the `general/subject/subject_id` field in each NWB file. The final `subjects` list is built in first-seen order from session metadata.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode() ...

unique_subjects = []
...
subj = result['subject']
if subj not in unique_subjects:
    unique_subjects.append(subj)
subj_idx = unique_subjects.index(subj)
```

iii. The notes state that the dataset contains 11 subject directories (`m3`, `m4`, `m7`, `m11`-`m15`, `m17`-`m19`) and that these correspond to the 11 mice in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session metadata are read from `general/session_id`, but the session boundary is fundamentally the file boundary.

ii.
```python
for fpath in all_files:
    result = process_session(fpath, ...)

session_id = f['general/session_id'][()].decode() ...
```

iii. The notes explicitly say “Each NWB session = one session in output.”

## 1-d. How are the data split into trials?

i. Trial starts are all indices where `trial_start_data > 0`. Trial ends are all indices where `teleport_data > 0`. If the counts differ, the AI trims both arrays to the shorter length. Each trial is then the half-open slice `ts:te`.

ii.
```python
trial_start_inds = np.where(trial_start_data > 0)[0]
teleport_inds = np.where(teleport_data > 0)[0]

if len(teleport_inds) != n_trials:
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
    teleport_inds = teleport_inds[:min_len]

trial_pos = position[ts:te]
trial_neural = deconv[ts:te, :]
```

iii. The notes justify trial segmentation at a high level by saying to use `trial_start` and `teleport` and include only data between those markers, but they do not justify using every positive `teleport` sample instead of teleport onsets.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not use the reference solution’s `<50 timepoint` rule. Instead, it skips trials when `te <= ts`, when a raw trial has fewer than 2 samples, or when fewer than 2 samples remain after masking out low-speed and NaN-neural frames. It also skips sessions with fewer than 2 trials initially or fewer than 2 valid trials after filtering.

ii.
```python
if n_trials < 2:
    return None

if te <= ts:
    continue

if n_trial_tp < 2:
    continue

valid_mask = speed_mask & neural_nan_mask
if valid_mask.sum() < 2:
    continue

if len(neural_trials) < 2:
    return None
```

iii. The notes frame this around decoder feasibility: timepoints with speed `<2 cm/s` are excluded and sessions need at least two valid trials. No explicit justification is given for replacing the reference’s minimum-trial-length rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB `processing/ophys/Deconvolved/plane*/data` arrays, then applies an `iscell` mask. It does not derive `neural` from raw `Fluorescence` and `Neuropil`.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0]
cell_mask = iscell == 1

deconv_group = f['processing/ophys/Deconvolved']
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_all = np.concatenate(plane_data, axis=1)
deconv = deconv_all[:, cell_mask]
```

iii. The notes justify this by claiming the NWB “Deconvolved” data are already the signal used by the reference decoder and that recomputing dF/F is unnecessary.

## 2-b. How is the `neural` data processed?

i. The AI concatenates deconvolved planes, filters to `iscell` ROIs, segments by trial, removes low-speed and NaN-neural frames, fills remaining neural NaNs with zero, and transposes each trial to `(n_neurons, n_timepoints)`. It does not recompute dF/F, baseline, smoothing, or OASIS deconvolution.

ii.
```python
trial_neural = deconv[ts:te, :]
speed_mask = trial_speed >= SPEED_THRESHOLD
neural_nan_mask = ~np.isnan(trial_neural[:, 0])
valid_mask = speed_mask & neural_nan_mask
trial_neural_valid = trial_neural[valid_mask, :]
trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
neural_matrix = trial_neural_valid.T.astype(np.float32)
```

iii. The notes say the NWB is “already preprocessed,” that the reference decoder uses deconvolved events, and that the reference applies a speed threshold at 2 cm/s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only cell-level quality filter is `iscell == 1`. At the timepoint level, the AI removes samples with speed `<2 cm/s` and samples where the first neuron is NaN. It does not apply the reference interneuron exclusion.

ii.
```python
cell_mask = iscell == 1
deconv = deconv_all[:, cell_mask]

speed_mask = trial_speed >= SPEED_THRESHOLD
neural_nan_mask = ~np.isnan(trial_neural[:, 0])
valid_mask = speed_mask & neural_nan_mask
```

iii. In the notes, the AI explicitly says it skipped interneuron exclusion because it thought the effect was small and because dF/F was not stored in the NWB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by using the same `trial_start`/`teleport` slices that define the trials. No additional temporal shifting is applied.

ii.
```python
trial_start_inds = np.where(trial_start_data > 0)[0]
...
trial_neural = deconv[ts:te, :]
trial_timestamps = timestamps[ts:te]
time_from_start = trial_times_valid - trial_timestamps[0]
```

iii. The notes state that the decoder alignment event is “start of trial (entry to virtual track).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps one sample per imaging frame and applies no temporal rebinning. In metadata it hard-codes an approximate `64.5` ms bin size and `15.5` Hz imaging rate instead of deriving them per session.

ii.
```python
frame_period = 1.0 / imaging_rate  # seconds per frame
...
'metadata': {
    'time_bin_size': 64.5,
    ...
    'imaging_rate_hz': 15.5,
}
```

iii. The notes say the data are already at imaging-frame resolution and describe the recordings as sampled at about 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps taken from `position/timestamps`.

ii.
```python
timestamps = bts['position/timestamps'][:]
...
trial_timestamps = timestamps[ts:te]
time_from_start = trial_times_valid - trial_timestamps[0]
```

iii. The notes say the behavior streams are already aligned and imply that any of the common behavior timestamp arrays would work.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI subtracts the first timestamp of the raw trial slice from the valid timestamps that remain after masking.

ii.
```python
trial_timestamps = timestamps[ts:te]
trial_times_valid = trial_timestamps[valid_mask]
time_from_start = trial_times_valid - trial_timestamps[0]
```

iii. The stated rationale is simply to represent elapsed time since trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamp vector and neural data are sliced with the same `ts:te` trial window and then filtered by the same `valid_mask`, so the surviving time values align one-to-one with the surviving neural frames.

ii.
```python
trial_neural = deconv[ts:te, :]
trial_timestamps = timestamps[ts:te]
valid_mask = speed_mask & neural_nan_mask
trial_neural_valid = trial_neural[valid_mask, :]
trial_times_valid = trial_timestamps[valid_mask]
```

iii. The notes say the NWB behavior and imaging data are already frame-aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. In the implemented code, environment type is derived from the scene string parsed from the NWB `identifier`, not from the loaded `environment/data` array.

ii.
```python
identifier = f['identifier'][()].decode() ...
scene = parse_scene(identifier)
...
env_per_trial = get_environment_from_scene(scene, n_trials)
```

iii. The notes argue that scene names encode the task condition structure, including the environment switch sessions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses the scene string. If the scene is an `Env1` or `Env2` session it assigns a constant 0 or 1; if it is a cross-environment switch scene it switches from one environment label to the other at trial 30; then it broadcasts the per-trial value across timepoints.

ii.
```python
def get_environment_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    if '_to_Env' in scene:
        ...
        env[:change_trial] = env1_num - 1
        env[change_trial:] = env2_num - 1
    elif scene.startswith('Env1'):
        env[:] = 0
    elif scene.startswith('Env2'):
        env[:] = 1

input_data[1, :] = env_type
```

iii. The notes say that all environment switches occur at trial 30 and that there are 11 such sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is not derived from the NWB `trial number` series. The AI uses the Python loop index `i` for each segmented trial.

ii.
```python
trial_number_data = bts['trial number/data'][:]
...
for i in range(n_trials):
    ...
    trial_num = float(i)
```

iii. The notes say trial number should be the within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is converted to float and broadcast across all timepoints in the trial.

ii.
```python
trial_num = float(i)
...
input_data[2, :] = trial_num
```

iii. The notes treat this as a simple per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` together with the behavior timestamp axis and the trial boundaries.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
trial_time_start = timestamps[ts]
trial_time_end = timestamps[te] if te < len(timestamps) else timestamps[-1]
rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                           (reward_timestamps <= trial_time_end))
```

iii. The notes define previous trial outcome as whether the preceding trial was rewarded or omitted.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes `reward_per_trial` as a binary outcome for each trial, then shifts that vector forward by one trial, setting the first trial’s previous outcome to 0, and broadcasts the resulting value across timepoints.

ii.
```python
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0

prev_outcome = np.zeros(n_trials, dtype=np.int64)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_data[3, :] = prev_out
```

iii. The notes say this matches the requested binary “omitted = 0, rewarded = 1” variable.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` and from reward-zone coordinates inferred from the scene string using `REWARD_ZONE_DICT` and a hard-coded switch at trial 30.

ii.
```python
position = bts['position/data'][:]
...
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
dist = compute_distance_to_reward_zone(trial_pos_valid, rz_start, rz_end)
```

iii. The notes justify this by saying the session identifier encodes reward-zone identity and switch structure.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each valid timepoint in a trial, the AI computes signed distance to the active reward zone: negative before the zone, zero inside, positive after the zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    distance[before] = position[before] - rz_start
    distance[after] = position[after] - rz_end
    distance[inside] = 0.0
    return distance
```

iii. The notes say this is the intended reward-relative spatial variable for the decoder.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI manually thresholds the continuous signed distance into seven bins matching the requested categories.

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

iii. The notes say these bins were chosen to match the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI computes the distance from `trial_pos_valid`, which has already been filtered by the same `valid_mask` used for the neural data, so it is aligned sample-by-sample to `trial_neural_valid`.

ii.
```python
valid_mask = speed_mask & neural_nan_mask
trial_pos_valid = trial_pos[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
dist = compute_distance_to_reward_zone(trial_pos_valid, rz_start, rz_end)
```

iii. The notes say behavior and neural streams are already aligned to imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the NWB `position/data` time series.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[ts:te]
pos_disc = discretize_position(trial_pos_valid)
```

iii. The notes identify this as the animal’s VR corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices position by trial, applies the same `valid_mask` used for neural data, and discretizes the surviving position samples. There is no additional smoothing or interpolation.

ii.
```python
trial_pos = position[ts:te]
trial_pos_valid = trial_pos[valid_mask]
pos_disc = discretize_position(trial_pos_valid)
```

iii. The notes say the aligned NWB position signal can be used as-is.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses five hard thresholds corresponding to 90 cm bins over the 450 cm track.

ii.
```python
out[position < 90] = 0
out[(position >= 90) & (position < 180)] = 1
out[(position >= 180) & (position < 270)] = 2
out[(position >= 270) & (position < 360)] = 3
out[position >= 360] = 4
```

iii. The notes say these are the requested equal-width absolute-position bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural data are aligned by the same trial slice and the same `valid_mask`.

ii.
```python
trial_pos_valid = trial_pos[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The notes state that the NWB behavior variables are already aligned to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the NWB `lick/data` time series.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick_binary[ts:te]
```

iii. The notes treat `lick` as the aligned framewise lick readout.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first applies a lick-sensor error heuristic: if more than 35% of samples in a trial have `lick > 2`, it sets the entire trial’s lick values to `NaN`. It then binarizes licks as `>0`, and finally converts `NaN` licks to 0.

ii.
```python
for i in range(n_trials):
    trial_licks = lick_corrected[ts:te]
    frac_error = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_error > LICK_ERROR_THRESHOLD:
        lick_corrected[ts:te] = np.nan

lick_binary[lick_corrected > 0] = 1
lick_binary[np.isnan(lick_corrected)] = 0
```

iii. The notes justify this from the paper code’s lick sensor correction threshold and say the threshold `0.35` was taken from code rather than the methods text.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick values are sliced by the same trial boundaries and then filtered by the same `valid_mask` as neural data.

ii.
```python
trial_lick = lick_binary[ts:te]
...
trial_lick_valid = trial_lick[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The notes say all behavior variables are already imaging-frame aligned in the NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the scene string parsed from the NWB `identifier`, not from the NWB `reward_zone` behavior series.

ii.
```python
identifier = f['identifier'][()].decode() ...
scene = parse_scene(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. The notes say the scene name encodes whether the session is fixed or switching and which reward-zone labels apply.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string into trialwise reward-zone labels, switches labels at trial 30 for switch scenes, and maps `A/B/C` to `0/1/2`. The per-trial label is then broadcast across all timepoints.

ii.
```python
rz_coords[:change_trial] = REWARD_ZONE_DICT[loc1]
rz_labels[:change_trial] = loc1
rz_coords[change_trial:] = REWARD_ZONE_DICT[loc2]
rz_labels[change_trial:] = loc2
...
rz_label = REWARD_ZONE_LABEL_MAP[rz_labels[i]]
output_data[4, :] = rz_label
```

iii. The notes say reward-zone switches occur at trial 30 and that class balance looked approximately one-third per zone.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, interpreted relative to the behavior timestamp axis and the trial boundaries.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                           (reward_timestamps <= trial_time_end))
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
```

iii. The notes describe this as the binary per-trial rewarded-versus-omitted outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI marks the trial as rewarded if any reward timestamp falls inside that trial’s time window, then broadcasts the binary value across the trial’s timepoints.

ii.
```python
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
...
reward_out = reward_per_trial[i]
output_data[5, :] = reward_out
```

iii. The notes say reward omission is a per-trial binary decoder target.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles data problems pragmatically and mostly silently: it trims mismatched numbers of trial starts and teleports to the shorter length, skips empty or too-short trials/sessions, skips sessions with no `iscell` neurons, converts NaN neural samples to 0 after masking, and converts lick-error NaNs to 0 after binarization. It does not implement the reference crop-for-length-mismatch or reward-alignment assertions.

ii.
```python
if len(teleport_inds) != n_trials:
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
    teleport_inds = teleport_inds[:min_len]

if n_neurons == 0:
    return None

trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
lick_binary[np.isnan(lick_corrected)] = 0
```

iii. The notes frame these as practical sanity and robustness choices, but do not document detailed validation for them.

## 13-a. What are the most time-consuming steps of the code?

i. In this code, the expensive steps are loading large NWB/HDF5 arrays, concatenating all deconvolved planes, looping over every trial to build masked neural/input/output arrays, and serializing the final pickle. If plotting is enabled, plot generation also adds cost.

ii.
```python
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_all = np.concatenate(plane_data, axis=1)
...
for i in range(n_trials):
    ...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes say the NWB already contains aligned, preprocessed signals, so the AI avoided the heavier survey and dF/F pipeline used in the reference solution.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several per-trial Python loops unvectorized: computing `reward_per_trial`, applying lick-sensor correction, and constructing per-trial neural/input/output arrays. Plotting also loops over trials repeatedly.

ii.
```python
for i in range(n_trials):
    ...
    rewards_in_trial = np.sum(...)

for i in range(n_trials):
    ...
    if frac_error > LICK_ERROR_THRESHOLD:
        ...

for i in range(n_trials):
    ...
    neural_trials.append(neural_matrix)
```

iii. No explicit justification is given beyond writing the code in a straightforward trial-by-trial style.

## 13-c. What processing does the code repeat multiple times?

i. The AI makes multiple passes over the same trial structure: one pass to compute `reward_per_trial`, one pass to correct licks, and a third pass to build the converted trial outputs. It also applies the same mask separately to each data stream and loads some arrays it never uses.

ii.
```python
for i in range(n_trials):
    ... reward_per_trial ...

for i in range(n_trials):
    ... lick_corrected ...

for i in range(n_trials):
    ... trial_pos_valid = trial_pos[valid_mask]
    ... trial_speed_valid = trial_speed[valid_mask]
    ... trial_lick_valid = trial_lick[valid_mask]
    ... trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The notes do not discuss this repetition directly; it is visible in the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `trial_number_data` and `environment_data` but never uses them, computes `n_timepoints`, `frame_period`, and `subjects_list` without using them, and imports `warnings` without using it. The plotting code also computes diagnostic figures that are not part of the converted dataset.

ii.
```python
import warnings
...
n_timepoints = len(position)
...
trial_number_data = bts['trial number/data'][:]
environment_data = bts['environment/data'][:]
...
frame_period = 1.0 / imaging_rate
...
subjects_list = []
```

iii. There is no explicit justification for these discarded computations; they appear to be leftovers from development and debugging.
