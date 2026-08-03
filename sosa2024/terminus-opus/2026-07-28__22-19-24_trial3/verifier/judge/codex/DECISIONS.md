# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script discovers every `sub-*` directory under `data/`, collects every `.nwb` file in each subject directory as a session, and processes each session one by one. Inside each session it opens the NWB file with `h5py` and reads specific behavioral and ophys arrays directly from the HDF5 tree rather than using `pynwb`.

ii.
```python
all_files = []
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
```

```python
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']
```

iii. In trajectory step 53 the AI explicitly says it decided to "Load NWB files using h5py". `CONVERSION_NOTES.md` says all 152 NWB files across 11 subjects should be processed and that NWB contains the needed behavioral and deconvolved neural arrays.

## 1-b. How are the data split into subjects?

i. Subjects are identified from directory names of the form `sub-mX`, and the saved subject names are `mX`.

ii.
```python
for sub_dir in sorted(os.listdir(data_dir)):
    if not sub_dir.startswith('sub-'):
        continue
    subject_id = sub_dir.replace('sub-m', '')
```

```python
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)
```

iii. `CONVERSION_NOTES.md` Step 2 lists the 11 subject directories and states that the NWB files are organized as `data/sub-mX/...`.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session number is parsed from the filename segment `_ses-YY_`.

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

iii. `CONVERSION_NOTES.md` Step 2 says the dataset contains files named `data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb` and counts 152 sessions total.

## 1-d. How are the data split into trials?

i. Trials are defined from each positive `trial_start` sample to the first later sample where `teleport > 0`. The script first finds all indices with `trial_start_signal > 0`, then for each start chooses the next positive teleport index.

ii.
```python
trial_start_inds = np.where(trial_start_signal > 0)[0]
teleport_inds = np.where(teleport_signal > 0)[0]

matched_starts = []
matched_teleports = []

for i in range(len(trial_start_inds)):
    start = trial_start_inds[i]
    future_teleports = teleport_inds[teleport_inds > start]
    if len(future_teleports) > 0:
        matched_starts.append(start)
        matched_teleports.append(future_teleports[0])
```

iii. In trajectory steps 13, 15, and 53 the AI states that it would use `trial_start` and `teleport` to define trials because that matched the reference code better than the stored trial number.

## 1-e. How are trials filtered based on quality controls?

i. The script keeps only sessions with at least 2 detected trials and at least 5 cells. Within a session it skips only very short trials with fewer than 3 samples.

ii.
```python
if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None
...
if n_trials < 2:
    print(f"  Skipping: only {n_trials} trials")
    return None
...
trial_len = stop - start
if trial_len < 3:
    continue
```

iii. The notes emphasize "at least two trials within each session" as a decoder-format requirement. The trajectory does not show a stronger trial-quality rule in the final script; the stricter `<50`-sample filter used in the human reference was not adopted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the NWB `processing/ophys/Deconvolved/*/data` arrays, filtered by `iscell`.

ii.
```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
```

iii. `CONVERSION_NOTES.md` Step 5 says "Use deconvolved events from NWB" and trajectory steps 49-50 conclude that the NWB `Deconvolved` data already correspond to the processed `events` used by the reference code.

## 2-b. How is the `neural` data processed?

i. The AI concatenates all deconvolved planes along the neuron axis, filters to `iscell == 1`, slices each trial, transposes to `(neurons, time)`, and casts to `float32`. It does not recompute dF/F or deconvolution.

ii.
```python
deconv = np.concatenate(deconv_parts, axis=1)
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Trajectory steps 57-58 explain the multi-plane decision: concatenate plane data then apply the combined `iscell` mask because the methods say planes were pooled. The notes also state that deconvolved events in NWB already match the reference `events`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The final code filters neurons only with the Suite2p `iscell` flag and skips sessions with fewer than 5 retained cells. It does not implement the additional interneuron exclusion that the AI considered during development.

ii.
```python
cell_mask = iscell[:, 0] == 1
n_cells = cell_mask.sum()

if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None

neural_all = deconv[:, cell_mask]
```

iii. `CONVERSION_NOTES.md` says cell filtering is `iscell=1`. Trajectory steps 102 and 104 explicitly note that interneuron exclusion was not implemented in the final script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start implicitly by slicing each trial from the chosen `trial_start` index to the matched teleport index.

ii.
```python
start = trial_start_inds[i]
stop = teleport_inds[i]
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The notes say "Temporal alignment: Align to trial start" and trajectory step 53 lists "Align to trial start" as a key design decision.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script uses the native imaging frame rate from NWB, with `dt = 1 / imaging_rate`, and does not apply any additional temporal rebinning.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
'time_bin_size': dt_ms,
```

iii. `CONVERSION_NOTES.md` Step 5 says "Time bin: native imaging rate (~64.5 ms)" and "No speed masking". Trajectory step 24 reports the imaging rate as 15.5078125 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The final script derives this input from trial length and the imaging rate, not from behavioral timestamps. It uses the number of frames in the trial and `dt = 1 / imaging_rate`.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
trial_len = stop - start
time_from_start = np.arange(trial_len) * dt
```

iii. The notes say "Time from trial start | frame_index * dt". The AI’s own mapping plan in `CONVERSION_NOTES.md` therefore documents the frame-based interpretation rather than using the stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the script generates a synthetic time axis starting at 0 and incrementing by one imaging frame duration.

ii.
```python
time_from_start = np.arange(trial_len) * dt
...
trial_input[0, :] = time_from_start
```

iii. The notes describe this transformation as `frame_index * dt` and say the data are aligned to trial start, so no further offset is applied.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The generated time vector is aligned by construction: it has exactly `trial_len` samples for the same `start:stop` slice used for the neural matrix.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_len = stop - start
time_from_start = np.arange(trial_len) * dt
trial_input = np.zeros((4, trial_len), dtype=np.float32)
trial_input[0, :] = time_from_start
```

iii. The notes say temporal alignment is to trial start and use a single native time step. There is no extra interpolation or timestamp matching in the final code.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The final script does not derive environment from the NWB `environment` time series. Instead it derives it from the per-session `scene` metadata in the copied `sessions_dict`.

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
    else:
        env[:] = 0 if 'Env1' in scene else 1
```

iii. `CONVERSION_NOTES.md` Step 5 says "Environment (0/1) | From scene name". Trajectory steps 44-45 say the AI decided session metadata captured the environment switch pattern.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script parses whether the scene name contains `Env1` or `Env2`, and for cross-environment switch sessions it changes the value after trial 30; otherwise the environment is constant across the whole session.

ii.
```python
env_per_trial = get_environment_from_scene(scene, n_trials)
...
env_type = env_per_trial[i]
...
trial_input[1, :] = float(env_type)
```

iii. The notes say the environment is per-trial and scene-based. Trajectory step 44 explicitly mentions using sessions that switch from one environment to another at trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the order of the detected trial loop after splitting with `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    ...
    trial_number = i
```

iii. `CONVERSION_NOTES.md` lists trial boundaries as coming from `trial_start` and `teleport`, and the stored `trial number` stream is loaded but not used for this input.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond taking the zero-based loop index and broadcasting it across all timepoints in the trial.

ii.
```python
trial_number = i
...
trial_input[2, :] = float(trial_number)
```

iii. The notes describe trial number simply as "0-indexed" and per trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward-event timestamps (`Reward/timestamps`) together with the `reward_zone` stream to decide whether the previous trial counted as rewarded.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. Trajectory steps 51-52 say the AI adopted the omission-trial logic from the reference code: a trial is rewarded only if there is a reward event and the reward zone was active.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first computes a per-trial `isreward` array, then shifts it by one trial so each trial carries the outcome of the immediately preceding trial. The first trial is set to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
prev_out = prev_outcome[i]
trial_input[3, :] = float(prev_out)
```

iii. `CONVERSION_NOTES.md` Step 5 defines previous outcome as binary omission/reward per trial, and trajectory step 52 describes the shift-from-previous-trial logic.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the `position` time series and reward-zone coordinates inferred from the session `scene`, not from the NWB `reward_zone` stream itself.

ii.
```python
position = behav['position']['data'][()]
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. `CONVERSION_NOTES.md` says "Reward zone from sessions_dict scene mapping". Trajectory steps 78-81 show the AI trusted scene metadata strongly enough that a copied `sessions_dict` bug caused a large reward-zone error that it later fixed.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint the script computes signed distance to the reward-zone interval: negative before the zone, zero in the zone, positive after the zone.

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

iii. The notes say distance is "Signed distance, 7 bins" and the trajectory discusses matching the paper’s reward-relative formulation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is manually thresholded into 7 categories matching the requested bins.

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

iii. `CONVERSION_NOTES.md` Step 5 lists "Signed distance, 7 bins". The thresholds follow the task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same `start:stop` trial slice as the neural data and computing distance from the per-trial `position` segment.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes repeatedly state that all converted variables are aligned to trial start and share the same per-trial time axis.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `position` directly to `output[1]`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The script clips positions to `[0, 450]` cm and then discretizes them into 5 equal-width bins over that interval.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The notes say "5 equal bins" and use the 450 cm track length from the paper as the basis.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholding uses 5 equal bins spanning 0 to 450 cm, i.e. edges at 90, 180, 270, and 360 cm after clipping.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(positions, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The justification in the notes is simply that the track is 450 cm and the task asked for 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by taking the same per-trial slice as the neural data and assigning one discretized position value per timepoint.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop]
pos_bins = discretize_position(pos_clipped)
```

iii. The notes treat position as a time-varying output aligned to trial start with the same native frame step.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the raw `lick` behavioral time series.

ii.
```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. `CONVERSION_NOTES.md` maps `lick` directly to `output[3]`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script applies a lick-sensor quality rule per trial: if more than 35% of samples have values greater than 2, it zeroes the whole trial. Otherwise it clips values above 1 down to 1 and casts to integer.

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The notes explicitly mention the reference lick-sensor error heuristic (`>35%` samples with cumulative lick `>2`). Trajectory steps 15 and 102 mention that the AI chose the code value 0.35 rather than the 0.30 value described in the paper.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same per-trial `start:stop` slice as the neural data and then binarizing within that slice.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_lick = lick_raw[start:stop].copy()
```

iii. The notes say all behavioral outputs are organized as trial-aligned time series at the native frame resolution.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is not derived from the NWB `reward_zone` stream in the final code. It is derived from the copied session `scene` metadata, which is converted to per-trial zone labels A/B/C.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
if rz_labels[i] == 'A':
    rz_label_idx[i] = 0
elif rz_labels[i] == 'B':
    rz_label_idx[i] = 1
elif rz_labels[i] == 'C':
    rz_label_idx[i] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 says "Reward zone from sessions_dict scene mapping". Trajectory steps 78-81 document that the AI originally copied this metadata incorrectly and then corrected the copied dictionary.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The script parses the scene string, assigns a single reward-zone location for fixed sessions, and for switch sessions splits trials at `CHANGE_TRIAL = 30` into before/after locations.

ii.
```python
if '_to_' in scene or '_to_Env' in scene:
    parts = scene.split('_to_')
    before_loc = parts[0][-1]
    after_loc = parts[1][-1]
    before_zone = LOCATION_TO_ZONE[before_loc]
    after_zone = LOCATION_TO_ZONE[after_loc]
    rz_coords[:change_trial] = REWARD_ZONE_DICT[before_zone]
    rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
    rz_coords[change_trial:] = REWARD_ZONE_DICT[after_zone]
    rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
```

iii. The notes claim this matches the reference code’s scene logic. The trajectory shows the AI treated the scene string as authoritative, even though its first copied version was wrong for GCAMP11.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward-event timestamps and the `reward_zone` behavioral stream via the intermediate per-trial `isreward` array.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. Trajectory step 52 says the AI used reward events plus reward-zone activity to distinguish rewarded trials from omissions and lapses.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the output is 1 if there is at least one reward event during that trial and the reward-zone stream is active somewhere in the same trial; otherwise 0. The trial-level value is then broadcast across all timepoints.

ii.
```python
rew_out = isreward[i]
...
trial_output[5, :] = rew_out
```

iii. The notes define reward outcome as binary per trial. The trajectory says the extra `rzone_active` condition came from the omission-trial logic in the reference analysis code.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles a few cases by skipping data rather than repairing it: it skips sessions with too few cells or trials, skips trials shorter than 3 samples, catches per-session exceptions and continues, and zeroes lick-error trials. It does not implement explicit cropping for neural/behavior length mismatches or a fallback for missing reward-zone labels because those are inferred from scene metadata.

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
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    continue
```

```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
```

iii. The notes and late trajectory emphasize decoder compatibility and continuing past bad sessions. Trajectory steps 102 and 104 also acknowledge some omitted robustness features, such as the unimplemented interneuron exclusion.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are reading every NWB file from disk, loading and concatenating full deconvolved arrays, iterating over all trials in all sessions, and saving the very large pickle. The final summary pass that concatenates all outputs across all sessions is also costly.

ii.
```python
for i, file_info in enumerate(all_files):
    ...
    result = process_session(...)
```

```python
for out_idx, out_name in enumerate(data['output_names']):
    vals = []
    for sess in output_list:
        for trial in sess:
            vals.append(trial[out_idx, :])
    all_vals = np.concatenate(vals)
```

iii. `CONVERSION_NOTES.md` Step 7 estimates runtime and Step 9 notes the pickle is about 9.8 GB, implying I/O and serialization dominate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop inside `process_session`, the loop that maps reward-zone labels to integers, and the nested summary loop over all output arrays are all straightforward vectorization targets.

ii.
```python
for i in range(n_trials):
    ...
    trial_neural = neural_all[start:stop, :].T.astype(np.float32)
    ...
    dist_bins = discretize_distance_to_reward(dist_to_rz)
```

```python
for i in range(n_trials):
    if rz_labels[i] == 'A':
        rz_label_idx[i] = 0
    elif rz_labels[i] == 'B':
        rz_label_idx[i] = 1
    elif rz_labels[i] == 'C':
        rz_label_idx[i] = 2
```

iii. The notes do not discuss optimization in detail, but the structure of the code makes these repeated Python loops obvious hotspots.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly performs per-trial slicing, broadcasting of per-trial scalars across every timepoint, and later re-traverses all outputs to build summary histograms after already constructing them. At the workflow level documented in the notes, the agent also reran conversion, verification, and training multiple times after the sessions-dict fix.

ii.
```python
trial_input[1, :] = float(env_type)
trial_input[2, :] = float(trial_number)
trial_input[3, :] = float(prev_out)
...
trial_output[4, :] = rz_loc
trial_output[5, :] = rew_out
```

```python
for out_idx, out_name in enumerate(data['output_names']):
    vals = []
    for sess in output_list:
        for trial in sess:
            vals.append(trial[out_idx, :])
```

iii. `CONVERSION_NOTES.md` records multiple full reruns after the reward-zone bug, and the final code clearly does an extra full pass over outputs just for console summaries.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several arrays that it never uses in the final converted dataset (`trial number`, `environment`, `Reward/data`, `scanning`, `autoreward`). It also returns large `session_info` metadata and computes console-only output distributions that are not needed by the decoder.

ii.
```python
trial_num = behav['trial number']['data'][()]
env_data = behav['environment']['data'][()]
reward_data = behav['Reward']['data'][()]
scanning = behav['scanning']['data'][()]
autoreward = behav['autoreward']['data'][()]
```

```python
'session_info': session_infos,
...
for out_idx, out_name in enumerate(data['output_names']):
    ...
    print(f"\n{out_name}:")
```

iii. The trajectory shows the AI explored these fields during reverse engineering, but the final decoder format only consumes a subset of them.
