# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are found by iterating over subdirectories of the data directory that start with `sub-`, then collecting all `.nwb` files within each. Data is loaded using `h5py` (direct HDF5 access) rather than `pynwb`. Each NWB file corresponds to one session. Behavioral data, neural data, and metadata are all extracted from the HDF5 structure.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
def load_nwb_session(filepath):
    with h5py.File(filepath, 'r') as f:
        bts = f['processing/behavior/BehavioralTimeSeries']
        ophys = f['processing/ophys']
        ...
```

iii. The agent explored the directory structure and identified the pattern of `sub-*` directories containing `.nwb` files. It chose `h5py` for direct HDF5 access rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects correspond to `sub-*` subdirectories. The `sub-` prefix is stripped to get mouse names (e.g., `sub-m11` -> `m11`). A mapping from NWB subject IDs to internal GCAMP names is maintained via `SUBJECT_MAP`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'
if mouse_name not in subject_names:
    subject_names.append(mouse_name)
```

iii. The agent identified the directory naming convention and created a mapping to the paper's internal naming scheme.

## 1-c. How are the data split into sessions?

i. Each NWB file within a subject directory corresponds to one session. The experiment day is extracted from the filename (e.g., `ses-03` -> day 3).

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
ses_part = sess_file.split('_')[1]  # 'ses-03'
exp_day = int(ses_part.split('-')[1])
```

iii. The agent parsed the filename convention to extract session/day numbers.

## 1-d. How are the data split into trials?

i. Trial starts are identified by `trial_start > 0` and trial ends by `teleport > 0`. The minimum of the two counts is used as the number of trials. Trials with `end <= start` or fewer than 2 timepoints are skipped.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
teleport_idx = np.where(nwb_data['teleport'] > 0)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))
tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
for i in range(n_trials):
    start = tstart_idx[i]
    end = teleport_idx[i]
    if end <= start:
        continue
    n_timepoints = end - start
    if n_timepoints < 2:
        continue
```

iii. The agent identified `trial_start` and `teleport` as the relevant boundary signals from the behavioral data.

## 1-e. How are trials filtered based on quality controls?

i. Trials with `end <= start` or fewer than 2 timepoints are skipped. Additionally, a lick error correction step marks trials where >30% of samples have lick counts >2 as erroneous (licks set to NaN, then to 0). No minimum trial length filter beyond 2 timepoints is applied.

ii.
```python
if end <= start:
    continue
n_timepoints = end - start
if n_timepoints < 2:
    continue
...
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
        trial_licks = licks[start:end]
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > threshold:
            licks[start:end] = np.nan
            error_trials.append(i)
```

iii. The agent added lick error correction based on the paper's methods for cleaning lick sensor artifacts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's pre-computed `Deconvolved` field (suite2p's deconvolution). It is NOT recomputed from raw fluorescence and neuropil traces as the paper describes.

ii.
```python
deconv_keys = list(ophys['Deconvolved'].keys())
...
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
deconvolved = np.concatenate(deconv_list, axis=1)
```

iii. The agent stated "Use deconvolved activity" in its key design decisions (Step 41). It did not replicate the paper's custom dF/F and deconvolution pipeline.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied to the deconvolved data beyond filtering by `iscell` and removing putative interneurons. The raw suite2p deconvolved values are used directly. There is no neuropil subtraction, no custom dF/F computation, no baseline estimation, no smoothing, and no re-deconvolution.

ii.
```python
deconvolved = nwb_data['deconvolved'][:, cell_mask]
...
neural = deconvolved[:, neuron_mask]  # (timepoints, n_neurons)
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The agent chose to use the pre-stored deconvolved signal rather than reimplementing the paper's full preprocessing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) cells are restricted to those marked as valid by suite2p's `iscell` curation, and (2) putative interneurons are excluded based on correlation between fluorescence (not dF/F) and running speed exceeding 0.5.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
...
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]
...
is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
```

iii. The agent followed the paper's method of filtering by iscell and excluding interneurons via speed correlation. However, it uses raw fluorescence for the correlation rather than dF/F.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by simply slicing the neural array between `tstart_idx[i]` and `teleport_idx[i]`. No additional temporal shifting is applied.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The alignment is to trial start, which is the trial_start event index. No additional alignment needed since the data is already indexed at the same rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to `1000.0 / 15.5078125` ms (~64.5 ms), derived from the imaging rate. No temporal rebinning is applied. However, the imaging rate is hardcoded as 15.5078125 Hz rather than read per-session and adjusted for multi-plane recordings.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
...
frame_time = 1.0 / nwb_data['imaging_rate']
```

iii. The agent hardcoded the imaging rate. The `imaging_rate` read from the NWB is the scanner rate, which for two-plane sessions is ~31 Hz (the per-plane rate would be ~15.5 Hz). The code uses `imaging_rate` directly for frame_time without dividing by number of planes.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is computed from the imaging frame rate rather than from stored timestamps. It uses `np.arange(n_timepoints) * frame_time`.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The agent chose to synthesize timestamps from the imaging rate rather than using the behavioral timestamps stored in the NWB file.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A time array is generated by multiplying sample indices by the frame time (1/imaging_rate). No subtraction of initial timestamp is needed since the array starts at 0.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. This approach assumes uniform sampling at the imaging rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same sample indices (`start:end`), so they are inherently aligned. The time array has the same number of timepoints as the neural data slice.

ii.
```python
n_timepoints = end - start
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. Same indexing ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the session scene metadata (`SESSIONS_INFO` dictionary) rather than from the `environment` behavioral time series in the NWB file.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. The agent hardcoded the session scene information from the paper's `sessions_dict.py` file.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine environment type (Env1=0, Env2=1). For switch sessions (day 8), the environment changes after trial 30 (SWITCH_TRIAL).

ii.
```python
def parse_scene_environment(scene):
    if 'Env1' in scene and 'Env2' not in scene:
        return 0, None
    elif 'Env2' in scene and 'Env1' not in scene:
        return 1, None
    elif 'Env1' in scene and 'Env2' in scene:
        parts = scene.split('_to_')
        env_before = 0 if 'Env1' in parts[0] else 1
        env_after = 0 if 'Env1' in parts[1] else 1
        return env_before, env_after
```

iii. The agent parsed the scene name convention used in the paper's codebase.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop counter `i` in the per-trial processing loop. It starts from 0 for each session.

ii.
```python
input_arr[2, :] = float(i)
```

iii. The agent used a sequential index rather than the stored `trial number` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing. The trial loop index is directly assigned as a float.

ii.
```python
input_arr[2, :] = float(i)
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward timestamps are compared against trial start/end timestamps to determine which trials were rewarded.

ii.
```python
reward_ts = nwb_data['reward_ts']
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. The agent checked whether reward event timestamps fell within each trial's time range.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The previous trial's reward outcome is shifted by one trial. The first trial gets 0 (no previous trial).

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_arr[3, :] = float(prev_outcome[i])
```

iii. Standard one-trial lag.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries. The reward zone for each trial is determined from the hardcoded `SESSIONS_INFO` scene metadata and `parse_scene_reward_zones()`, which assigns zone labels based on scene name and a switch at trial 30.

ii.
```python
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The agent used the paper's session scene metadata to determine reward zone locations rather than deriving them from the NWB `reward_zone` behavioral time series.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if before the zone, 0 if inside, positive if after. The function computes `position - rz_start` for positions before the zone and `position - rz_end` for positions after.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end
    return dist
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Manual conditional assignment using boolean masks:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to < 0 cm
- 3: 0 cm (exactly 0, i.e. inside zone)
- 4: >0 to +10 cm
- 5: +10 to +50 cm
- 6: > +50 cm

ii.
```python
def discretize_distance_to_reward(distance):
    out = np.zeros_like(distance, dtype=int)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. Matches the bin specification in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same slice indices (`start:end`) are used for both neural and position data, ensuring alignment.

ii.
```python
pos = nwb_data['position'][start:end]
```

iii. Same indexing as neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = nwb_data['position'][start:end]
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

iii. Direct use of position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm before discretization. Then discretized into 5 equal-sized 90 cm bins using `floor(position / bin_size)`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The clipping ensures positions outside the track range are assigned to the first or last bin.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Uses `floor(position / 90)` clipped to [0, 4]:
- 0: 0-90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: 360-450 cm

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 450/5 = 90
out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. Equal-sized bins across the track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same slice indices as neural data.

ii.
```python
pos = nwb_data['position'][start:end]
```

iii. Same indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series, with additional lick error correction applied.

ii.
```python
lick = bts['lick/data'][:]
...
licks_corrected, error_trials = correct_lick_sensor_errors(
    nwb_data['lick'], tstart_idx, teleport_idx,
    threshold=LICK_ERROR_THRESHOLD
)
...
lck = licks_corrected[start:end]
```

iii. The agent added lick sensor error correction from the paper's methods.

## 9-b. What processing is involved in computing `output` *Lick*?

i. First, lick error correction is applied: trials where >30% of samples have lick count >2 have their lick data set to NaN, then NaN is converted to 0. Then lick is binarized: any value >0 becomes 1.

ii.
```python
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. Binarization matches the instructions (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same slice indices as neural data.

ii.
```python
lck = licks_corrected[start:end]
```

iii. Same indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the hardcoded `SESSIONS_INFO` scene metadata rather than from the NWB `reward_zone` behavioral time series. The scene name encodes the reward zone (A, B, or C) and whether a switch occurs.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The agent used the paper's session metadata to determine reward zones.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to identify reward zone labels. For single-zone sessions, all trials get the same zone. For switch sessions, trials before trial 30 get the old zone and trials after get the new zone. Zone labels are mapped to integers (A=0, B=1, C=2).

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    rz_labels = np.empty(n_trials, dtype='U1')
    if scene.endswith('_LocationA'):
        rz_labels[:] = 'A'
    ...
    else:
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after
    return rz_labels
```

iii. The switch at trial 30 is a hardcoded constant from the paper.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps.

ii.
```python
reward_ts = nwb_data['reward_ts']
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. Reward events are matched to trial boundaries using timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check whether any reward event timestamp falls within the trial's start and end timestamps. The result is binary (0=no reward, 1=reward). The value is constant across all timepoints in the trial.

ii.
```python
output_arr[5, :] = reward_per_trial[i]
```

iii. Per-trial binary outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is truncated to the minimum of the two.
- **Short trials**: Trials with end <= start or < 2 timepoints are skipped.
- **Lick sensor errors**: Trials with >30% high lick counts have lick data zeroed out.
- **Missing session info**: Sessions not found in `SESSIONS_INFO` are skipped with a warning.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    position = position[:min_len]
    ...
    deconvolved = deconvolved[:min_len]
```

iii. Defensive checks found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py (I/O bound, reading large arrays)
2. The interneuron identification loop (per-neuron correlation computation)
3. Iterating over all sessions for conversion

ii. N/A

iii. The NWB files contain large neural recordings.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `identify_interneurons` function loops over neurons individually to compute correlations. This could be vectorized with a matrix correlation computation. The per-trial loop for reward detection could also be vectorized.

ii.
```python
for c in range(n_neurons):
    neural_ts = neural_data[valid_mask, c]
    speed_ts = speed_data[valid_mask]
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

iii. Per-neuron correlation is the natural but slower approach.

## 13-c. What processing does the code repeat multiple times?

i. No survey/preprocessing pass is done separately -- the code processes each session once. However, behavioral and neural data arrays are loaded in full and then sliced per trial, which is done once per session.

ii. N/A

iii. The code is relatively efficient in this regard.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `fluorescence` data is loaded and used only for interneuron identification via speed correlation. Since the code uses the pre-stored `Deconvolved` data for the actual neural output, the fluorescence loading is only needed for this filtering step. Additionally, the lick error correction adds processing overhead for a variable that is simply binarized.

ii. N/A

iii. The fluorescence loading is a side-effect of using the stored deconvolution rather than recomputing it.
