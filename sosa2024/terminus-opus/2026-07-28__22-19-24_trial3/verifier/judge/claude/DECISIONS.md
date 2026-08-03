# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by iterating through sorted subdirectories of `data/` that start with `sub-`. For each subject directory, it finds all `.nwb` files, extracts the subject ID and session number from the filename. Data is loaded using `h5py` (not `pynwb`). It uses a hardcoded `_all_sessions` dictionary (copied from the reference code's `sessions_dict`) to map subject/session to scene information.

ii.
```python
data_dir = 'data'
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
Loading:
```python
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']
```

iii. The AI confirmed 11 subjects and 152 sessions matching the paper. It used `h5py` instead of `pynwb` for performance reasons. The sessions_dict was imported from the reference code for scene-to-reward-zone mapping.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` matching `sub-mX`. The subject ID is extracted by stripping the `sub-m` prefix. A `subject_map` dictionary tracks unique subjects and assigns indices.

ii.
```python
subject_id = sub_dir.replace('sub-m', '')
...
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)
```

iii. Same approach as reference — subjects from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from filenames (`_ses-XX_`).

ii.
```python
ses_num = int(fname.split('_ses-')[1].split('_')[0])
```

iii. Same approach as reference — one NWB file per session.

## 1-d. How are the data split into trials?

i. Trial boundaries are found using `trial_start` (>0) and `teleport` (>0) signals. For each trial start, the next teleport index after that start is found. Trial data runs from start to teleport (exclusive of teleport endpoint).

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
```
Trial slicing:
```python
trial_len = stop - start
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Trial boundaries are matched by finding the next teleport after each trial start. This differs from the reference which uses the rising edge of teleport.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 3 timepoints are skipped. Sessions with fewer than 2 valid trials or fewer than 5 cells are skipped entirely.

ii.
```python
if trial_len < 3:
    continue
...
if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None
...
if len(neural_trials) < 2:
    print(f"  Skipping: only {len(neural_trials)} valid trials")
    return None
```

iii. Minimal filtering — short trials excluded, sessions with too few cells or trials skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `Deconvolved` data in the `ophys` processing module.

ii.
```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. Matches reference — deconvolved calcium events are used as neural data.

## 2-b. How is the `neural` data processed?

i. Multi-plane data is concatenated along the neuron axis. Cells are filtered by `iscell`. Data is cast to float32 and transposed to (n_cells, trial_len).

ii.
```python
deconv = np.concatenate(deconv_parts, axis=1)  # (n_timepoints, total_rois)
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The AI loads `iscell` from a single `ImageSegmentation/PlaneSegmentation` location rather than per-plane, and applies it to the concatenated data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Filtered by `iscell[:, 0] == 1` from Suite2P cell classification. Sessions with fewer than 5 cells are skipped.

ii.
```python
cell_mask = iscell[:, 0] == 1
n_cells = cell_mask.sum()
if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None
neural_all = deconv[:, cell_mask]
```

iii. Same iscell filter as reference. The reference loads iscell per-plane; the AI loads a combined iscell from a single location.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing from the trial start index. No additional alignment processing is needed since data starts at trial_start.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Same as reference — aligned to trial start via indexing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native imaging rate. It is computed as `1000.0 / imaging_rate` ms. No rebinning is applied.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
dt_ms = 1000.0 / imaging_rate
```

iii. The AI uses the raw imaging rate from the NWB metadata. This differs from the reference which computes `nplanes / plane_data.rate * 1000` to account for multi-plane scanning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the imaging rate and frame index — computed as `np.arange(trial_len) * dt` where `dt = 1.0 / imaging_rate`.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
time_from_start = np.arange(trial_len) * dt
```

iii. Instead of using timestamps from the behavioral data, the AI synthesizes time from the imaging rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as frame index multiplied by the frame duration (`dt`). No subtraction of initial timestamp needed since the index starts at 0.

ii.
```python
time_from_start = np.arange(trial_len) * dt
trial_input[0, :] = time_from_start
```

iii. The reference uses `timestamps_curr - timestamps_curr[0]` from the NWB behavioral timestamps. The AI's approach gives equivalent results if the timestamps are evenly spaced at the imaging rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data share the same indexing (same timepoints from start:stop), so alignment is implicit.

ii.
```python
# Both use same start:stop range
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
time_from_start = np.arange(trial_len) * dt  # trial_len = stop - start
```

iii. Same assumption as reference — neural and behavioral data are co-registered at the same time indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `scene` name in the hardcoded `_all_sessions` dictionary (copied from reference code), NOT from the `environment` behavioral time series in the NWB file.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
env_per_trial = get_environment_from_scene(scene, n_trials)
```
```python
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
    return env
```

iii. The AI chose to derive environment from the scene name to be consistent with the reference code's sessions_dict. The reference solution reads the `environment` behavioral time series directly from the NWB file.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine Env1 (0) or Env2 (1). For cross-environment switch sessions, environment changes at trial 30 (CHANGE_TRIAL).

ii. See 4-a code above.

iii. The value is per-trial, broadcast to all timepoints.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `i` over matched trial boundaries.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = i
    trial_input[2, :] = float(trial_number)
```

iii. Same approach as reference — sequential index within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — the loop index is used directly. The value is constant across all timepoints in a trial.

ii.
```python
trial_input[2, :] = float(trial_number)
```

iii. Same as reference.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward event timestamps are compared to behavioral timestamps to determine which trials had rewards.

ii.
```python
reward_data = behav['Reward']['data'][()]
reward_ts = behav['Reward']['timestamps'][()]
...
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
    rzone_active = np.any(rzone_data[start:stop+1] > 0)
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. Reward is determined per-trial and previous trial outcome is shifted by one trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if reward timestamps fall within the trial's time window AND if the reward zone was active. Previous trial outcome is simply the current trial's reward shifted by one: `prev_outcome[1:] = isreward[:-1]`, with the first trial set to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
trial_input[3, :] = float(prev_out)
```

iii. The AI adds an additional `rzone_active` check that the reference does not use. The first trial defaults to 0 (same as reference).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone coordinates determined from the `scene` name in the sessions_dict. The reward zone boundaries come from the hardcoded `REWARD_ZONE_DICT`.

ii.
```python
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Reward zone coordinates are from the reference code's sessions_dict, mapping scene names to zone locations A=[80,130], B=[200,250], C=[320,370].

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to nearest edge of reward zone. 0 if inside zone, negative if before, positive if past.

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

iii. Same logic as the reference implementation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean masking rather than `np.digitize`.

ii.
```python
def discretize_distance_to_reward(distances):
    bins = np.zeros(len(distances), dtype=int)
    bins[distances < -50] = 0
    bins[(distances >= -50) & (distances < -10)] = 1
    bins[(distances >= -10) & (distances < 0)] = 2
    bins[distances == 0] = 3
    bins[(distances > 0) & (distances <= 10)] = 4
    bins[(distances > 10) & (distances <= 50)] = 5
    bins[distances > 50] = 6
    return bins
```

iii. The bin boundaries match the instructions. The reference uses `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`, which produces equivalent results.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices (start:stop), so no additional alignment is needed.

ii.
```python
trial_pos = position[start:stop]
```

iii. Same approach as reference.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. Same as reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before binning, then discretized into 5 equal-sized bins using `np.linspace(0, 450, 6)` = `[0, 90, 180, 270, 360, 450]`.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The AI defines "equal-sized" as equal divisions of the track length [0, 450], giving 90cm bins. The reference uses bins `[-inf, 50, 150, 250, 350, inf]` which are 100cm wide but offset to account for the actual position range of roughly -50 to 450.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Uses `np.digitize` with edges `[90, 180, 270, 360, 450]` (from `np.linspace(0, 450, 6)[1:]`), then clips to [0, 4]. Position is first clipped to [0, 450].

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

iii. Different bin edges than reference: AI uses [0-90, 90-180, 180-270, 270-360, 360-450] vs reference [-inf-50, 50-150, 150-250, 250-350, 350-inf].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data — no additional alignment.

ii.
```python
trial_pos = position[start:stop]
```

iii. Same as reference.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error correction: if >35% of samples in a trial have lick values > 2, all lick values for that trial are set to 0. Otherwise, lick values > 1 are clipped to 1. The result is binary (0/1).

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The AI applied the reference code's `correct_lick_sensor_error` logic. The reference solution simply binarizes with `(licks_curr > 0).astype(int)` without error correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
trial_lick = lick_raw[start:stop].copy()
```

iii. Same as reference.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `scene` name in the hardcoded `_all_sessions` dictionary, NOT from the `reward_zone` behavioral time series.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. The AI parsed scene names like `Env1_LocationA_to_B` to determine reward zone. Switch sessions change at trial 30. The reference uses a Viterbi algorithm on observed reward zone positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing determines zone label per trial. For switch sessions, the zone changes at trial 30. Labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    if '_to_' in scene or '_to_Env' in scene:
        parts = scene.split('_to_')
        before_loc = parts[0][-1]
        after_loc = parts[1][-1]
        ...
        rz_coords[:change_trial] = REWARD_ZONE_DICT[before_zone]
        rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
        rz_coords[change_trial:] = REWARD_ZONE_DICT[after_zone]
        rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
    else:
        loc = scene[-1]
        zone = LOCATION_TO_ZONE[loc]
        rz_coords[:] = REWARD_ZONE_DICT[zone]
        rz_labels[:] = ZONE_TO_LABEL[zone]
    ...
    rz_label_idx[i] = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The AI uses the reference code's sessions_dict approach rather than inferring zones from the data. This is a more direct approach but assumes the sessions_dict is correct and the trial count matches expectations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps and the `reward_zone` behavioral time series.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. Same source as reference (Reward timestamps), but with additional reward_zone check.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamps fall within the trial's time window AND if reward_zone was active during the trial. Binary output: 1 if rewarded, 0 if not.

ii.
```python
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
...
trial_output[5, :] = rew_out
```

iii. The additional `rzone_active` check differs from the reference. The reference just checks if any reward event timestamp falls in the trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Short trials**: Trials with < 3 timepoints skipped
- **Too few cells**: Sessions with < 5 cells skipped
- **Too few trials**: Sessions with < 2 valid trials skipped
- **Lick sensor errors**: Trials with >35% lick values > 2 have licks zeroed
- **Processing errors**: Individual sessions wrapped in try/except, errors logged and session skipped

ii.
```python
if trial_len < 3:
    continue
if n_cells < 5:
    return None
if len(neural_trials) < 2:
    return None
try:
    result = process_session(...)
except Exception as e:
    print(f"  ERROR: {e}")
    continue
```

iii. The AI is more defensive with error handling than the reference, using try/except around session processing.

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files with `h5py` and reading large neural arrays. The AI reports full conversion takes about 90s for 152 sessions.

ii. N/A

iii. Using h5py is faster than pynwb. No survey step is needed since the AI uses the sessions_dict for reward zone info.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop processes each trial sequentially. The reward detection loop iterates over all trials. Some of the boolean masking for discretization could be done in batch before splitting into trials.

ii. N/A

iii. Variable trial lengths make full vectorization difficult.

## 13-c. What processing does the code repeat multiple times?

i. The AI's code does NOT have a separate survey step, so it only loads each NWB file once. This is more efficient than the reference which has a survey step that loads all files before conversion.

ii. N/A

iii. The sessions_dict approach avoids needing a survey/Viterbi pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `autoreward`, `scanning`, and `rzone_data` variables that are used minimally or not at all in the final output (rzone_data is only used for the reward outcome check). The `env_data` variable is loaded but never used (environment comes from scene name instead).

ii.
```python
autoreward = behav['autoreward']['data'][()]  # loaded but never used
scanning = behav['scanning']['data'][()]  # loaded but never used
env_data = behav['environment']['data'][()]  # loaded but not used
```

iii. These are minor inefficiencies.
