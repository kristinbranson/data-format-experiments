# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for subject subdirectories whose names start with `sub-`, then scans each subject directory for `.nwb` files. Each NWB file is treated as one session and opened directly with `h5py`; within each file it reads behavioral datasets from `processing/behavior/BehavioralTimeSeries` and neural datasets from `processing/ophys`.

ii.
```python
for sub_dir in sorted(os.listdir(data_dir)):
    if not sub_dir.startswith('sub-'):
        continue
    subject_id = sub_dir.replace('sub-m', '')
    sub_path = os.path.join(data_dir, sub_dir)
    for fname in sorted(os.listdir(sub_path)):
        if not fname.endswith('.nwb'):
            continue
        ses_num = int(fname.split('_ses-')[1].split('_')[0])
        all_files.append({
            'path': os.path.join(sub_path, fname),
            'subject_id': subject_id,
            'session_num': ses_num,
        })
...
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']
```

iii. In `CONVERSION_NOTES.md` the AI says it found 11 subject folders and 152 NWB files, and it explicitly chose HDF5-based loading because the NWB files exposed the needed arrays in stable paths.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by directory name. The code strips the `sub-m` prefix from each subject directory, stores that as `subject_id`, and later rebuilds session-to-subject indexing with `subject_map`.

ii.
```python
for sub_dir in sorted(os.listdir(data_dir)):
    if not sub_dir.startswith('sub-'):
        continue
    subject_id = sub_dir.replace('sub-m', '')
...
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)
...
subject_idx.append(subject_map[sub_name])
```

iii. The notes state that the dataset is organized as `data/sub-mX/...`, so directory boundaries were taken as mouse boundaries.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The session number is parsed from the filename fragment `_ses-XX_`.

ii.
```python
for fname in sorted(os.listdir(sub_path)):
    if not fname.endswith('.nwb'):
        continue
    ses_num = int(fname.split('_ses-')[1].split('_')[0])
    all_files.append({
        'path': os.path.join(sub_path, fname),
        'subject_id': subject_id,
        'session_num': ses_num,
    })
```

iii. The notes say the NWB naming scheme is `sub-mX_ses-YY_behavior+ophys.nwb`, so the AI used that session identifier directly.

## 1-d. How are the data split into trials?

i. Trials are defined from the `trial_start` and `teleport` behavioral signals. The AI finds all positive `trial_start` indices, finds all positive `teleport` indices, and for each trial start chooses the next teleport as that trial’s end. Each trial is then sliced as `[start:stop)`.

ii.
```python
trial_start_inds = np.where(trial_start_signal > 0)[0]
teleport_inds = np.where(teleport_signal > 0)[0]
...
for i in range(len(trial_start_inds)):
    start = trial_start_inds[i]
    future_teleports = teleport_inds[teleport_inds > start]
    if len(future_teleports) > 0:
        matched_starts.append(start)
        matched_teleports.append(future_teleports[0])
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. In the notes the AI says it matched the reference task structure by aligning trials from `trial_start` to `teleport`.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference trial QC. Instead it skips sessions with fewer than 2 trials, and within a retained session it skips only very short trials with fewer than 3 timepoints.

ii.
```python
if n_trials < 2:
    print(f"  Skipping: only {n_trials} trials")
    return None
...
trial_len = stop - start
if trial_len < 3:
    continue
```

iii. `CONVERSION_NOTES.md` says “Short trials (<3 timepoints): Skipped”; no stronger trial filtering justification is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB `Deconvolved` arrays, concatenated across imaging planes, plus the `iscell` mask for ROI filtering.

ii.
```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
```

iii. The notes explicitly say “Use deconvolved events from NWB (matches reference code sess.timeseries['events'])”.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: concatenate deconvolved planes, apply the `iscell` mask, then slice trials and cast to `float32`. The AI does not recompute dF/F or deconvolution from fluorescence and neuropil.

ii.
```python
deconv = np.concatenate(deconv_parts, axis=1)
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The notes justify this as using the NWB event signal because it “matches reference code sess.timeseries['events']”.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC is filtering ROIs by `iscell[:, 0] == 1`, plus dropping sessions with fewer than 5 surviving cells. There is no putative interneuron exclusion.

ii.
```python
cell_mask = iscell[:, 0] == 1
n_cells = cell_mask.sum()

if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None

neural_all = deconv[:, cell_mask]
```

iii. The notes say “Cell filtering: iscell=1 (Suite2P manual curation)” and do not mention the reference interneuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned by cutting trials from `trial_start` to `teleport`, so timepoint 0 of each trial corresponds to trial start.

ii.
```python
trial_start_inds = np.array(matched_starts)
teleport_inds = np.array(matched_teleports)
...
start = trial_start_inds[i]
stop = teleport_inds[i]
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
...
'temporal_alignment_event': 'trial_start',
```

iii. The notes repeatedly state that the conversion is “aligned to trial start”.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame period `1 / imaging_rate` as the time bin size and does no temporal rebinning.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
'time_bin_size': dt_ms,
```

iii. The notes say “Time bin: native imaging rate (~64.5 ms)” and justify leaving the data at native resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from trial length and the imaging rate, not from behavioral timestamps. The AI uses `trial_len` together with `dt = 1 / imaging_rate`.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
trial_len = stop - start
time_from_start = np.arange(trial_len) * dt
```

iii. The notes say this variable was computed from frame index times the native imaging timestep.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial the AI creates a uniformly spaced vector beginning at 0 seconds: `0, dt, 2*dt, ...`.

ii.
```python
time_from_start = np.arange(trial_len) * dt
...
trial_input[0, :] = time_from_start
```

iii. The notes justify this as using the native frame rate for trial-aligned continuous time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is aligned implicitly because it has exactly one sample per neural frame in the `[start:stop)` neural slice for that trial.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_len = stop - start
time_from_start = np.arange(trial_len) * dt
trial_input = np.zeros((4, trial_len), dtype=np.float32)
trial_input[0, :] = time_from_start
```

iii. The notes describe the inputs and neural activity as sharing the same per-trial frame count.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does not use the raw `environment` time series. It derives environment type from the hard-coded `scene` string looked up from the session table.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
...
def get_environment_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    env = np.zeros(n_trials, dtype=int)
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        before_env = 0 if 'Env1' in parts[0] else 1
        after_env = 0 if 'Env1' in parts[1] else 1
        env[:change_trial] = before_env
        env[change_trial:] = after_env
```

iii. The notes say the conversion uses “scene-based reward zone determination using reference code's sessions_dict”; the same scene table is also used for environment identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code parses `Env1` versus `Env2` from the scene name. For cross-environment switch sessions it assumes a fixed switch after trial 30; otherwise one environment label is broadcast across all trials in the session.

ii.
```python
if '_to_Env' in scene:
    parts = scene.split('_to_')
    before_env = 0 if 'Env1' in parts[0] else 1
    after_env = 0 if 'Env1' in parts[1] else 1
    env[:change_trial] = before_env
    env[change_trial:] = after_env
else:
    env[:] = 0 if 'Env1' in scene else 1
...
trial_input[1, :] = float(env_type)
```

iii. The notes justify this using the paper’s session metadata and the stated “switch after 30 trials”.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from a raw NWB variable. It is derived from the trial loop index `i` after trial segmentation.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = i
```

iii. The notes list trial number as “0-indexed” and generated during conversion.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is done beyond assigning the current trial index and broadcasting it across all timepoints in the trial.

ii.
```python
trial_input[2, :] = float(trial_number)
```

iii. The notes describe this as a per-trial contextual input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps (`Reward/timestamps`) and, additionally, from whether `reward_zone` was active during the trial. The AI first computes a per-trial `isreward` vector.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The notes say the agent consulted reward/omission handling in the reference code and wanted to distinguish rewarded from omitted trials.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. After computing `isreward` per trial, the AI shifts that vector by one trial: the first trial gets 0, and each later trial gets the previous trial’s reward outcome, broadcast across the trial.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
prev_out = prev_outcome[i]
...
trial_input[3, :] = float(prev_out)
```

iii. The notes explicitly say “prev_outcome[i] = reward_outcome[i-1]”.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The continuous distance is derived from `position` plus reward-zone coordinates inferred from session metadata (`scene` via the hard-coded session table), not from the raw `reward_zone` signal itself.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
trial_pos = position[start:stop]
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes justify this with the reference code’s `sessions_dict` and record that the AI corrected an initially wrong copied session table.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the reward-zone interval: negative before the zone, zero inside it, and positive after it.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    distance[before] = position[before] - rz_start
    in_zone = (position >= rz_start) & (position <= rz_end)
    distance[in_zone] = 0
    after = position > rz_end
    distance[after] = position[after] - rz_end
    return distance
```

iii. The notes state that distance-to-zone was one of the main decoded outputs and should reflect reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses a hand-written 7-bin thresholding scheme with exact comparisons for the required bins.

ii.
```python
bins[distances < -50] = 0
bins[(distances >= -50) & (distances < -10)] = 1
bins[(distances >= -10) & (distances < 0)] = 2
bins[distances == 0] = 3
bins[(distances > 0) & (distances <= 10)] = 4
bins[(distances > 10) & (distances <= 50)] = 5
bins[distances > 50] = 6
```

iii. The notes say the output bins were chosen to match the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing position-derived outputs from the same per-trial `[start:stop)` slices used for the neural data.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
trial_output[0, :] = dist_bins
```

iii. The notes say the conversion uses trial-aligned behavioral and neural time series with shared frame counts.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the raw `position` behavioral time series.

ii.
```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. The notes identify `position` as the source variable for corridor location.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position to the nominal track range `[0, 450]` and then discretizes it into five equal-width bins.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

iii. The notes say the track length is 450 cm and position should be converted to 5 equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are generated from `np.linspace(0, TRACK_LENGTH, 6)`, then `np.digitize` is used and clipped to indices `0..4`.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The notes justify this as five equal bins spanning the 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial `[start:stop)` slices as the neural data.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop]
trial_output[1, :] = pos_bins
```

iii. The notes describe behavior and neural activity as frame-aligned within each trial.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the raw `lick` behavioral time series.

ii.
```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. The notes identify `lick` as the source of the decoder’s lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies a lick-sensor error heuristic: if more than 35% of a trial’s samples exceed 2, the whole trial is set to 0. Otherwise values above 1 are clipped to 1, and the result is cast to integers.

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The notes say the AI copied the `>35%` lick-sensor error rule from the reference code, but adapted it to produce binary decoder labels.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced on the same `[start:stop)` trial interval as the neural data.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_lick = lick_raw[start:stop].copy()
trial_output[3, :] = trial_lick
```

iii. The notes describe lick as one of the time-varying trial outputs built on the same trial segmentation as neural activity.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from session metadata (`scene` from the hard-coded session table), not from the raw `reward_zone` NWB variable.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. The notes justify this with the reference session metadata and document that the session table had to be corrected.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine location A/B/C, and switch sessions are split at trial 30 into pre-switch and post-switch zone labels. Those labels are converted to indices 0, 1, 2.

ii.
```python
rz_coords[:change_trial] = REWARD_ZONE_DICT[before_zone]
rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
rz_coords[change_trial:] = REWARD_ZONE_DICT[after_zone]
rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
...
if rz_labels[i] == 'A':
    rz_label_idx[i] = 0
elif rz_labels[i] == 'B':
    rz_label_idx[i] = 1
elif rz_labels[i] == 'C':
    rz_label_idx[i] = 2
```

iii. The notes say the reward zone comes from scene-based session metadata and the known 30-trial switch rule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward event timestamps (`Reward/timestamps`) and the raw `reward_zone` signal, via the per-trial `isreward` calculation.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The notes justify this as distinguishing rewarded from omitted trials using the reference omission logic.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code marks the trial as rewarded if at least one reward event timestamp falls between the trial’s start and stop timestamps and the reward-zone signal was active in that trial. The resulting scalar is broadcast across the trial.

ii.
```python
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
    rzone_active = np.any(rzone_data[start:stop+1] > 0)
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
...
trial_output[5, :] = rew_out
```

iii. The notes report that reward outcome and previous-trial outcome were checked against this shifted `isreward` logic.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Error handling is lightweight. The AI skips sessions with too few cells or trials, skips trials shorter than 3 samples, zeroes lick traces on suspected lick-sensor-error trials, and catches exceptions at the session level so conversion can continue.

ii.
```python
if n_cells < 5:
    return None
...
if n_trials < 2:
    return None
...
if trial_len < 3:
    continue
...
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
...
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    continue
```

iii. The notes explicitly mention handling short trials and lick-sensor errors; they do not document special handling for timestamp mismatches or missing behavioral streams.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading whole behavioral and deconvolved arrays from every NWB file, concatenating plane-wise neural arrays, iterating over all trials to build per-trial tensors, and finally serializing the large pickle.

ii.
```python
f = h5py.File(nwb_path, 'r')
...
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
...
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes say the full run processed 152 sessions in about 90 seconds and produced a ~9.8 GB pickle, implying I/O and whole-dataset serialization dominate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-start/teleport matching loop, per-trial reward classification loop, reward-zone-label conversion loop, and much of the main per-trial conversion loop could in principle be vectorized or precomputed at the session level.

ii.
```python
for i in range(len(trial_start_inds)):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    if rz_labels[i] == 'A':
        rz_label_idx[i] = 0
...
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    ...
```

iii. The notes do not discuss vectorization directly, but the code structure makes these loops the obvious opportunities.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly parses scene-derived metadata, repeatedly allocates and fills per-trial input/output arrays, and repeatedly performs the same range tests for reward outcome and discretization trial by trial. Across sessions it also re-reads the same kinds of large arrays from disk.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
env_per_trial = get_environment_from_scene(scene, n_trials)
...
trial_input = np.zeros((4, trial_len), dtype=np.float32)
trial_output = np.zeros((n_outputs, trial_len), dtype=np.int64)
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
```

iii. This is mostly implicit in the code; the notes only mention that scene mapping and session metadata were reused throughout the conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several arrays it never uses (`trial number`, `environment`, `Reward/data`, `scanning`, `autoreward`) and also creates an initial 1-D `trial_input` array that is immediately overwritten by a 2-D version.

ii.
```python
trial_num = behav['trial number']['data'][()]
env_data = behav['environment']['data'][()]
reward_data = behav['Reward']['data'][()]
scanning = behav['scanning']['data'][()]
autoreward = behav['autoreward']['data'][()]
...
trial_input = np.array([time_from_start[0], float(env_type), float(trial_number), float(prev_out)], dtype=np.float32)
trial_input = np.zeros((4, trial_len), dtype=np.float32)
```

iii. The notes do not call this out, but the dead loads and overwritten allocation are directly visible in the script.
