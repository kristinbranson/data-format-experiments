# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds every NWB file under `data/sub-*/*.nwb`, opens each file with `h5py.File`, and loads behavioral and ophys arrays from each file into memory before per-session processing.

ii. ```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
```

```python
with h5py.File(nwb_path, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
    ophys = f['processing/ophys']
```

iii. In `CONVERSION_NOTES.md`, Step 2 says the dataset is organized as one NWB file per session under `data/sub-{mouse}/...`, and Step 5 lists “Load NWB: Read all behavioral and neural data” as the first pipeline step.

## 1-b. How are the data split into subjects?

i. Subjects are split implicitly by NWB file and then identified by the NWB `general/subject/subject_id` field. After processing, unique subject IDs are sorted into `subjects`, and each session gets a `subject_idx`.

ii. ```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes repeatedly treat subject identity as NWB metadata (`subject_id`) and describe the dataset as 11 subject directories / mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. The main loop iterates file-by-file, and each `process_session(...)` return becomes one session in the output lists.

ii. ```python
for nwb_path in nwb_files:
    session_label = fname.replace('_behavior+ophys.nwb', '')
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. Step 2 of the notes says the files are organized as `sub-{mouse}_ses-{session}_behavior+ophys.nwb`, so the agent chose a one-file-one-session mapping.

## 1-d. How are the data split into trials?

i. Trials are split using `trial_start` and `teleport` behavioral flags. For each matched start/end pair, the script slices all arrays with `[s:e]`.

ii. ```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
...
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    trial_pos = position[s:e]
    trial_neural = neural_data[:, s:e]
```

iii. The notes say trial alignment is “between `trial_start` and `teleport`” and that the decoder should be aligned to trial start.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only dropped if the end index is not after the start index or if the segment has fewer than 2 time bins. The code does not implement the reference converter’s `min_ntimepoints=50` threshold.

ii. ```python
if e <= s or (e - s) < 2:
    ...
    continue
...
if valid_trial_count < 2:
    return None
```

iii. The agent’s notes emphasize session-level validity (“at least two trials”) and do not document a stricter per-trial minimum beyond malformed slices.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` output is derived from `processing/ophys/Deconvolved/plane*/data`, after concatenating planes and filtering ROIs by `iscell` and then by the script’s interneuron mask.

ii. ```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
...
cell_mask = iscell[:, 0].astype(bool)
deconv_cells = deconv_all[:, cell_mask].T
neural_data = deconv_cells[final_cell_mask]
```

iii. Step 5 in the notes states “Neural data = deconvolved events” and explicitly says the NWB files already contain precomputed deconvolved events.

## 2-b. How is the `neural` data processed?

i. The deconvolved planes are concatenated, optionally cropped to match behavior length, filtered by cell masks, and then sliced per trial. Separately, the script computes dF/F only to support interneuron detection, not as the stored neural signal.

ii. ```python
deconv_all = np.concatenate(deconv_list, axis=1)
...
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
...
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The notes say the decoder should use deconvolved events directly, but they also justify computing dF/F because the paper/code used dF/F-speed correlation for interneuron exclusion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first keeps only ROIs with `iscell[:, 0] == 1`, then removes ROIs whose dF/F is correlated with speed above `0.5`, calling those putative interneurons.

ii. ```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

iii. Step 5 in the notes says “Filter neurons: Apply iscell filter; exclude interneurons (speed corr > 0.5).”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by using each trial’s `trial_start` index as the zero-time point and then slicing until `teleport`.

ii. ```python
s = trial_start_inds[i]
e = teleport_inds[i]
trial_neural = neural_data[:, s:e]
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes explicitly say “Trial alignment = trial start” to match the decoder instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses one native imaging frame per time bin. For single-plane recordings the rate is `imaging_rate`; for multi-plane sessions it uses `imaging_rate / n_planes`. No further rebinning is applied.

ii. ```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
...
'time_bin_size': time_bin_ms,
```

iii. The notes say “Time bin = imaging frame: ~64.5 ms per frame at ~15.5 Hz. This matches the native sampling rate.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. In the implemented code it is derived from the trial length and the effective imaging rate, not from the raw NWB timestamp array.

ii. ```python
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes described this as “(t - trial_start) / imaging_rate,” which matches the implemented synthetic frame clock rather than using raw timestamps directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the script creates a `0, 1/effective_rate, 2/effective_rate, ...` vector of length `n_t` and stores it as the first input row.

ii. ```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The agent justified using the imaging frame as the native time bin and treating trial start as time zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is built with exactly the same trial slice length `n_t = e - s` as the neural matrix, so every neural frame gets one corresponding time value.

ii. ```python
trial_neural = neural_data[:, s:e]
n_t = e - s
input_arr = np.zeros((4, n_t), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. The notes say all streams are aligned at the imaging frame rate and sliced between the same trial boundaries.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. ```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
```

iii. The notes map “Environment (morph)” to the environment behavioral timeseries.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code takes the median of nonnegative environment samples within the trial, converts that to a scalar float, and then broadcasts it across the full trial as input row 1.

ii. ```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
input_arr[1, :] = env_type
```

iii. The notes describe environment as a per-trial binary variable, so the code collapses each trial to one environment label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. In the actual implementation it is not derived from the raw `trial number/data` values. It is derived from the loop index `i` after trial segmentation.

ii. ```python
trial_num = bts['trial number/data'][:]
...
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. The notes intended this to be “trial number within session, 0-indexed”; the code realizes that by using trial order rather than reading the stored trial-number series.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code uses the zero-based trial loop index, converts it to float, and broadcasts it across the whole trial.

ii. ```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. This matches the notes’ decision to use trial number “within session” rather than a global counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` by testing whether the previous trial contained any reward timestamp.

ii. ```python
reward_timestamps = bts['Reward/timestamps'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
prev_outcome = np.float32(prev_trial_rewarded)
```

iii. The notes say previous outcome should come “from reward detection in previous trial.”

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code tracks a `prev_trial_rewarded` state variable across the trial loop. For trial 0 it forces 0; afterward it sets the next trial’s input based on whether the current trial had any reward event.

ii. ```python
prev_trial_rewarded = 0
...
prev_outcome = np.float32(prev_trial_rewarded)
...
prev_trial_rewarded = int(was_rewarded)
```

iii. The notes explicitly say “For the first trial, assume no previous reward.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-trial `position` samples plus the reward-zone start/end coordinates assigned to that trial.

ii. ```python
trial_pos = position[s:e]
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Step 5 in the notes maps this output to position plus reward-zone coordinates.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each position sample, the code computes a signed distance to the nearest point in the reward zone: negative before the zone, zero inside, positive after the zone.

ii. ```python
before = position < rz_start
inside = (position >= rz_start) & (position <= rz_end)
after = position > rz_end
dist[before] = position[before] - rz_start
dist[inside] = 0.0
dist[after] = position[after] - rz_end
```

iii. The notes justify this as reward-relative position centered on the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is manually thresholded into 7 bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii. ```python
out[dist < -50] = 0
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. These thresholds are copied directly from the decoder-task instructions and from the notes’ Step 5 table.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It uses the same `[s:e]` trial slice as the neural data and is stored as a time-varying row in the per-trial output matrix.

ii. ```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
output_arr[0, :] = dist_to_rz_binned
```

iii. The notes say all outputs should remain frame-aligned to the neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. ```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The notes map absolute position directly from the position behavioral timeseries.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code bins continuous position into 5 equal-width bins over `[0, 450]` cm using `np.linspace`.

ii. ```python
POSITION_BIN_EDGES = np.linspace(0, TRACK_LENGTH, POSITION_BINS + 1)
...
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
binned = np.digitize(position, bin_edges) - 1
```

iii. The notes explicitly say “5 equal bins (90 cm each),” following the decoder-task instruction rather than the hidden reference converter’s `50/150/250/350` thresholds.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` assigns each sample to one of five bins and then clips out-of-range values into `[0, 4]`.

ii. ```python
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes define the intended bins as `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, `[360,450]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same trial indices `[s:e]` and written as a time-varying output row with one value per neural frame.

ii. ```python
trial_pos = position[s:e]
output_arr[1, :] = pos_binned
```

iii. The notes say behavioral and neural streams are already synchronized at the frame level.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. ```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The notes map lick directly from the NWB lick timeseries.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code copies the lick segment, zeros the whole trial if more than 35% of samples exceed 2, caps values above 1, and then converts the result to binary `0/1`.

ii. ```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The notes justify this from the paper/repo’s lick-sensor error rule and say to “apply the 35% threshold ... then cap at 1.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks are sliced with the same trial `[s:e]` indices as the neural data and stored as a time-varying binary row.

ii. ```python
trial_lick = lick[s:e].copy()
output_arr[3, :] = lick_binned
```

iii. The notes say all outputs are kept at the imaging-frame resolution.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string, not from the raw `reward_zone/data` values. The script parses the scene name and infers per-trial reward-zone labels from it.

ii. ```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
...
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
rz_loc = rz_label_to_idx(rz_labels[i])
```

iii. The notes say to use `behavior.get_reward_zones()`-style scene parsing with default switch trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code maps scene names such as `Env1_LocationB_to_A` to a trial-by-trial label sequence (`A/B/C`) using hard-coded coordinates and a default switch at trial 30, then converts labels to indices `A=0, B=1, C=2`.

ii. ```python
elif 'B_to' in scene and scene[-1] == 'A':
    rz_coords[:change_trial] = REWARD_ZONE_DICT['B']
    rz_labels[:change_trial] = 'B'
    rz_coords[change_trial:] = REWARD_ZONE_DICT['A']
    rz_labels[change_trial:] = 'A'
...
mapping = {'A': 0, 'B': 1, 'C': 2}
```

iii. The notes explicitly call out `get_reward_zones()` from the paper code and record “Change trial = 30.”

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` (with `Reward/data` loaded but not actually used).

ii. ```python
reward_timestamps = bts['Reward/timestamps'][:]
reward_data = bts['Reward/data'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The notes say reward outcome should come from reward detection within each trial window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code checks whether any reward timestamp falls between the trial’s first and last behavior timestamps and broadcasts that `0/1` value across the whole trial.

ii. ```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    return np.any((reward_timestamps >= trial_start_time) &
                  (reward_timestamps <= trial_end_time))
...
output_arr[5, :] = reward_outcome
```

iii. The notes describe this as “check if any reward timestamp falls within trial window.”

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script crops behavior and neural arrays to the shorter length if they differ, skips malformed/very short trial segments, defaults environment to 0 if a trial has no nonnegative environment samples, defaults unknown scenes to reward zone A, and uses `1e-10` guards in dF/F baseline division.

ii. ```python
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    position = position[:n_timepoints_total]
...
if e <= s or (e - s) < 2:
    continue
...
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
...
abs_baseline[abs_baseline < 1e-10] = 1e-10
```

iii. The notes mention length-cropping, handling odd lick/error cases, and using robust defaults where the NWB contents do not match ideal assumptions.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are per-session loading of large NWB matrices, per-trial dF/F computation for all `iscell` neurons, and the interneuron-detection pass across the full session.

ii. ```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
is_interneuron = detect_interneurons(dff_full, speed)
```

iii. The trajectory says “The bottleneck is interneuron detection ... Let me vectorize it,” which confirms where the agent saw the main runtime cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial dF/F loop is the clearest candidate. The agent already vectorized the Pearson-correlation step, but trial-by-trial dF/F assignment, reward detection, and per-trial array assembly still run in Python loops.

ii. ```python
for i in range(n_trials):
    ...
    dff_full[:, s:e] = compute_dff_trial(...)
...
for i in range(n_trials):
    ...
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. The trajectory explicitly notes an optimization pass for interneuron detection, implying the remaining trial loops were still serial.

## 13-c. What processing does the code repeat multiple times?

i. It loops over trials once to compute dF/F for interneuron detection and then again to build trial outputs. It also repeatedly slices the same trial boundaries across many behavioral variables.

ii. ```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(...)

for i in range(n_trials):
    trial_pos = position[s:e]
    trial_speed = speed[s:e]
    trial_lick = lick[s:e].copy()
    trial_neural = neural_data[:, s:e]
```

iii. This structure follows the agent’s two-stage plan in the notes: first neuron filtering, then per-trial extraction.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads raw fluorescence and neuropil traces and computes dF/F even though the stored `neural` output uses only deconvolved events. It also loads variables like `reward_data`, `trial_num`, `scanning`, and `rzone_cumul` without using them in the final dataset.

ii. ```python
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
...
dff_full[:, s:e] = compute_dff_trial(...)
...
reward_data = bts['Reward/data'][:]
trial_num = bts['trial number/data'][:]
scanning = bts['scanning/data'][:]
```

iii. The notes justify dF/F as a support computation for interneuron exclusion, but that work is discarded once the final deconvolved neural matrices are written.
