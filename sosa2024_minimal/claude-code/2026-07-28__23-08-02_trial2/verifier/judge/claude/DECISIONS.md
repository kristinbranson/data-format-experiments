# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of the `data` directory that start with `sub-`. It uses `h5py` (not `pynwb`) to read NWB files directly. It has a hardcoded `SUBJECT_MAP` mapping NWB subject IDs to GCAMP names, and a `SESSIONS_INFO` dict mapping (subject, exp_day) to scene names. Only subjects present in `SUBJECT_MAP` are processed.

ii.
```python
def load_nwb_session(filepath):
    with h5py.File(filepath, 'r') as f:
        bts = f['processing/behavior/BehavioralTimeSeries']
        ophys = f['processing/ophys']
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick = bts['lick/data'][:]
        ...
        deconv_keys = list(ophys['Deconvolved'].keys())
        for pk in sorted(deconv_keys):
            deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
        deconvolved = np.concatenate(deconv_list, axis=1)
```

```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
for subj_dir_name in subjects:
    if subj_id not in SUBJECT_MAP:
        print(f"Warning: Unknown subject {subj_id}, skipping")
        continue
```

iii. The AI uses `h5py` for direct HDF5 access rather than `pynwb`. The `SUBJECT_MAP` and `SESSIONS_INFO` are hardcoded from the paper's `sessions_dict.py`. The AI validates subjects against this map.

## 1-b. How are the data split into subjects?

i. Subjects correspond to `sub-*` directories. The AI maps each directory name to a mouse name (e.g., `sub-m11` -> `m11`) and stores these as subject names. Only subjects in `SUBJECT_MAP` are included.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'
if mouse_name not in subject_names:
    subject_names.append(mouse_name)
```

iii. The AI checks against `SUBJECT_MAP` to ensure only known subjects are processed.

## 1-c. How are the data split into sessions?

i. Each NWB file within a subject directory corresponds to one session. The session/exp_day number is parsed from the filename (e.g., `ses-03` -> 3). The AI also looks up session metadata from `SESSIONS_INFO` using the GCAMP name and exp_day.

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
ses_part = sess_file.split('_')[1]  # 'ses-03'
exp_day = int(ses_part.split('-')[1])
scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. Files are sorted to ensure deterministic ordering. Sessions without entries in `SESSIONS_INFO` are skipped.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using `trial_start > 0` for starts and `teleport > 0` for ends. The number of trials is set to `min(len(tstart_idx), len(teleport_idx))`. Trials where `end <= start` or `n_timepoints < 2` are skipped.

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

iii. The AI uses all positive teleport indices rather than detecting rising edges. The `min()` approach handles any mismatch in counts. The verification output shows some sessions with non-80 trial counts (41, 50, 60, 75, 90, 100), suggesting the trial segmentation may not be fully correct.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if `end <= start` or if they have fewer than 2 timepoints. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
if end <= start:
    continue
n_timepoints = end - start
if n_timepoints < 2:
    continue
...
if len(trial_neural) < 2:
    print(f"  Skipping session: only {len(trial_neural)} valid trials")
    return None
```

iii. The minimum trial length threshold of 2 is much lower than the reference's 50 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field under `processing/ophys`. The AI also loads `Fluorescence` data for interneuron identification.

ii.
```python
deconv_keys = list(ophys['Deconvolved'].keys())
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
deconvolved = np.concatenate(deconv_list, axis=1)

fluor_keys = list(ophys['Fluorescence'].keys())
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. The deconvolved data matches the paper's description. Fluorescence is loaded for the interneuron exclusion step.

## 2-b. How is the `neural` data processed?

i. Neural data from multiple planes are concatenated (sorted by key name). The data is then filtered by `iscell` and by interneuron exclusion. The resulting deconvolved activity is cast to float32.

ii.
```python
deconvolved = np.concatenate(deconv_list, axis=1)
...
cell_mask = nwb_data['iscell'][:, 0] == 1
deconvolved = nwb_data['deconvolved'][:, cell_mask]
...
is_interneuron = identify_interneurons(fluorescence, nwb_data['speed'], valid_mask)
neuron_mask = ~is_interneuron
neural = deconvolved[:, neuron_mask]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. Multi-plane pooling follows the paper's statement that "planes were pooled for all analyses."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Suite2P `iscell` curation (`iscell[:, 0] == 1`), and (2) interneuron exclusion based on Pearson correlation > 0.5 between fluorescence and running speed.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]

def identify_interneurons(neural_data, speed_data, valid_mask, threshold=SPEED_CORR_THRESHOLD):
    for c in range(n_neurons):
        r = np.corrcoef(neural_ts, speed_ts)[0, 1]
        if r > threshold:
            is_interneuron[c] = True
    return is_interneuron

neuron_mask = ~is_interneuron
neural = deconvolved[:, neuron_mask]
```

iii. The interneuron exclusion follows the paper's Methods: "Pearson correlation of >0.5 between dF/F timeseries and the animal's running speed." The AI uses fluorescence as a proxy for dF/F.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Neural data for each trial is extracted using the trial_start and teleport indices.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The trial start event naturally provides the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is computed from the imaging rate as `1000.0 / 15.5078125` (~64.48 ms). No temporal rebinning is applied. The imaging rate is read from the NWB file.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
```

iii. The imaging rate value is hardcoded in metadata rather than dynamically computed across sessions, though the rate is read from the NWB file for computing frame_time.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Computed from the imaging rate rather than from timestamps. Time from start is `np.arange(n_timepoints) * frame_time`.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The AI uses the imaging rate to synthesize timestamps rather than using the stored behavioral timestamps directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as frame index multiplied by frame duration (1/imaging_rate). This starts at 0 for each trial.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The computation assumes uniform frame spacing at exactly the imaging rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data use the same sample indices, so alignment is inherent. The time array has the same number of elements as the neural data for each trial.

ii.
```python
n_timepoints = end - start
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
trial_n = neural[start:end, :].T
```

iii. Same indexing ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the **scene name** in `SESSIONS_INFO`, NOT from the NWB `environment` variable. The scene name is parsed to determine Env1 (0) or Env2 (1).

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

env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. The AI parses environment from the hardcoded session metadata rather than reading it from the NWB file directly.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene name string is parsed to extract environment type. For environment switch sessions (day 8), the first 30 trials get the pre-switch environment and remaining trials get the post-switch environment.

ii.
```python
input_arr[1, :] = env_per_trial[i]
```

iii. The switch point is hardcoded at trial 30 (`SWITCH_TRIAL = 30`).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `i` within the session processing loop.

ii.
```python
input_arr[2, :] = float(i)
```

iii. This is a 0-indexed sequential trial number within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index as a float. The value is constant across all timepoints within a trial.

ii.
```python
input_arr[2, :] = float(i)
```

iii. The trial number is simply the sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward` timestamps in the NWB file. Reward events are matched to trial boundaries using behavior timestamps.

ii.
```python
reward_ts = nwb_data['reward_ts']
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. The AI determines reward per trial by checking if any reward timestamp falls within the trial time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, previous trial outcome is the reward outcome of trial i-1. First trial gets 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_arr[3, :] = float(prev_outcome[i])
```

iii. Simple shift of the reward outcome array.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone location. Reward zone location is determined from `SESSIONS_INFO` scene names (NOT from the NWB `reward_zone` variable). On switch days, the zone switches at trial 30.

ii.
```python
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The AI uses hardcoded session metadata and the paper's reward zone coordinates to determine zone identity per trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before zone, 0 inside zone, positive after zone.

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

iii. Standard signed distance computation to nearest zone boundary.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic matching the instruction bins.

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

iii. The bin edges match the instructions. Bin 3 uses `distance == 0` which corresponds to being inside the zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
pos = nwb_data['position'][start:end]
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. Both use the same `start:end` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = nwb_data['position'][start:end]
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

iii. Position is clipped to [0, 450] before discretization.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, TRACK_LENGTH=450], then discretized into 5 equal bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360-450).

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 450/5 = 90
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The AI defines "equal-sized" bins as dividing the track length (0-450) by 5, giving 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each: 0-90, 90-180, 180-270, 270-360, 360-450 cm. Position is first clipped to [0, 450].

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 90
out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. The output_values labels confirm: `['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm']`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
pos = nwb_data['position'][start:end]
```

iii. Same `start:end` slice as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = bts['lick/data'][:]
...
lck = licks_corrected[start:end]
```

iii. The raw lick data goes through lick sensor error correction before use.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two-step processing: (1) Lick sensor error correction: trials where >30% of samples have lick count >2 have their lick data set to NaN, then NaN values are replaced with 0. (2) Binarization: any positive value maps to 1.

ii.
```python
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
        trial_licks = licks[start:end]
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > threshold:
            licks[start:end] = np.nan
    return licks, error_trials

lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. The lick error correction follows the paper's description of handling lick sensor artifacts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
lck = licks_corrected[start:end]
```

iii. Same `start:end` slice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the hardcoded `SESSIONS_INFO` scene names, NOT from the NWB `reward_zone` variable. Scene names encode which reward zone is active (A, B, or C) and when switches occur.

ii.
```python
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
output_arr[4, :] = rz_loc
```

iii. The scene names from `sessions_dict.py` encode the reward zone schedule for each session.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing determines zone labels per trial. For switch sessions (e.g., `LocationA_to_B`), trials 0-29 get zone A and trials 30+ get zone B. The zone letter is mapped to an integer (A=0, B=1, C=2).

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
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

iii. The switch point at trial 30 comes from the paper.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file.

ii.
```python
reward_ts = nwb_data['reward_ts']
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. Reward events are matched to trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward timestamp falls within the trial's time range, 0 otherwise. The value is constant across all timepoints in a trial.

ii.
```python
output_arr[5, :] = reward_per_trial[i]
```

iii. Per-trial binary outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Neural/behavior length mismatch**: Data truncated to minimum length.
- **Short trials**: Trials with < 2 timepoints skipped.
- **Lick sensor errors**: Trials with >30% high lick counts have lick data zeroed out.
- **Missing session info**: Sessions without entries in `SESSIONS_INFO` are skipped.
- **Trials where end <= start**: Skipped.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    position = position[:min_len]
    ...
    deconvolved = deconvolved[:min_len]
```

iii. The neural/behavior mismatch affects one session (m18 ses-01) per the CONVERSION_NOTES.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with h5py and reading large arrays (neural data)
2. **Interneuron identification** - requires computing correlations for each neuron
3. **Full conversion loop** - iterating over all sessions and trials
4. **Saving the pickle file**

ii. N/A

iii. The h5py loading and array concatenation are I/O bound.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `identify_interneurons` function loops over each neuron individually to compute correlations. This could be vectorized with matrix operations. The per-trial loop in `process_session` could partially be vectorized for operations like position discretization.

ii.
```python
for c in range(n_neurons):
    neural_ts = neural_data[valid_mask, c]
    speed_ts = speed_data[valid_mask]
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

iii. Computing correlations one neuron at a time is inefficient.

## 13-c. What processing does the code repeat multiple times?

i. The code does not have a separate survey step, so it loads each NWB file only once. However, the `parse_scene_reward_zones` and `parse_scene_environment` functions are called per session, doing redundant string parsing. The reward zone coordinate lookup is repeated per trial.

ii. N/A

iii. No major repeated processing since the code does a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads **fluorescence** data for every session to compute interneuron correlations, which adds significant I/O overhead. The `run_sanity_checks` function processes data that is not part of the output. The lick error correction processes all lick data even for trials that pass the check.

ii.
```python
fluor_keys = list(ophys['Fluorescence'].keys())
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. Fluorescence data is large and only used for interneuron identification, which the reference solution does not perform.
