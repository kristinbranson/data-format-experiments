# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from all NWB files found in `sub-*` subdirectories of the data directory. It uses `h5py` to directly read the HDF5/NWB files (rather than the `pynwb` library). It loads behavioral time series, neural data (deconvolved and fluorescence), ROI metadata, imaging rate, and subject/session identifiers from each file. A hardcoded `SUBJECT_MAP` and `SESSIONS_INFO` dictionary provides metadata about each subject and session (mapping NWB IDs to GCAMP names and scene/reward-zone information from the paper's `sessions_dict.py`).

ii.
```python
def load_nwb_session(filepath):
    """Load data from a single NWB file."""
    with h5py.File(filepath, 'r') as f:
        bts = f['processing/behavior/BehavioralTimeSeries']
        ophys = f['processing/ophys']
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick = bts['lick/data'][:]
        trial_start = bts['trial_start/data'][:]
        teleport = bts['teleport/data'][:]
        # ... (loads all behavioral and neural variables)
        deconv_list = []
        for pk in sorted(deconv_keys):
            deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
        deconvolved = np.concatenate(deconv_list, axis=1)
```

```python
def convert_all_data(data_dir, sample_only=False):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    for subj_dir_name in subjects:
        session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
        for sess_file in session_files:
            nwb_data = load_nwb_session(filepath)
            result = process_session(nwb_data, scene, exp_day)
```

iii. The AI loads all sub-* directories and all .nwb files within each, matching the paper's 11 switch-task mice. It uses h5py for direct HDF5 access rather than pynwb. Session metadata (scene, environment, reward zone) is derived from a hardcoded `SESSIONS_INFO` dictionary transcribed from the paper's code.

## 1-b. How are the data split into subjects?

i. Subjects correspond to `sub-*` subdirectories of the data directory. The AI maps each to a short name (e.g., `sub-m11` -> `m11`) and also maintains a mapping to GCAMP names for looking up session metadata.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
# ...
mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'
if mouse_name not in subject_names:
    subject_names.append(mouse_name)
subj_idx = subject_names.index(mouse_name)
```

iii. The directory structure directly encodes subject identity. The AI verified 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session day is parsed from the filename (e.g., `ses-03` -> day 3).

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
for sess_file in session_files:
    ses_part = sess_file.split('_')[1]  # 'ses-03'
    exp_day = int(ses_part.split('-')[1])
```

iii. NWB filenames encode session number. The total of 152 sessions matches expectations (14 days x 11 mice minus m11's missing days 1-2).

## 1-d. How are the data split into trials?

i. Trial boundaries are defined by `trial_start > 0` (start) and `teleport > 0` (end). The AI uses `np.where` to find all positive indices for both signals, then pairs them by index position.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
teleport_idx = np.where(nwb_data['teleport'] > 0)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))
tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
```

iii. The AI finds all positive values of trial_start and teleport and pairs them. This differs from the reference approach which detects teleport onset transitions. If teleport stays positive for multiple consecutive frames, `np.where(teleport > 0)` would yield many more indices than trials. However, the AI obtains ~80 trials per session (matching the paper), suggesting teleport may be a single-frame pulse in the data.

## 1-e. How are trials filtered based on quality controls?

i. Trials with `end <= start` or fewer than 2 timepoints are skipped. No minimum trial length threshold comparable to the reference's 50-timepoint minimum is applied.

ii.
```python
if end <= start:
    continue
n_timepoints = end - start
if n_timepoints < 2:
    continue
```

iii. The AI applies a minimal filter (n_timepoints >= 2) rather than a more conservative threshold. This means very short trials (e.g., 3-49 timepoints) that the reference would exclude are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field in the NWB ophys processing module.

ii.
```python
deconv_keys = list(ophys['Deconvolved'].keys())
deconv_list = []
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
deconvolved = np.concatenate(deconv_list, axis=1)
```

iii. The paper specifies using deconvolved calcium activity (OASIS algorithm). This matches the reference.

## 2-b. How is the `neural` data processed?

i. Neural data from multiple imaging planes is concatenated. ROIs are filtered by the `iscell` flag. Additionally, putative interneurons are identified by computing the Pearson correlation between fluorescence and running speed, and excluded if correlation > 0.5.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]

is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
neural = deconvolved[:, neuron_mask]
```

iii. The AI applies two filtering steps: iscell (from Suite2P curation) and interneuron exclusion (from the paper's methods). The reference only applies iscell filtering.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two quality filters are applied: (1) Suite2P's `iscell` flag to select manually curated ROIs, and (2) interneuron exclusion based on fluorescence-speed correlation > 0.5.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
# ...
def identify_interneurons(neural_data, speed_data, valid_mask, threshold=SPEED_CORR_THRESHOLD):
    for c in range(n_neurons):
        r = np.corrcoef(neural_ts, speed_ts)[0, 1]
        if r > threshold:
            is_interneuron[c] = True
    return is_interneuron
```

iii. The paper mentions excluding putative interneurons, which the AI implements. The reference code does not implement this filter. The impact is small (typically 0-4 neurons per session, occasionally up to 26 for sub-m3).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by extracting the slice from `tstart_idx[i]` to `teleport_idx[i]`. No additional temporal shifting is needed since trial start IS the alignment event.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The instructions specify alignment to trial start, so no offset is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is ~64.48 ms (1/15.5078125 Hz). No rebinning is applied; data is kept at the native imaging rate. For multi-plane recordings, the effective rate accounts for the number of planes.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
# ...
'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
```

iii. The imaging rate is consistent across sessions. The AI hardcodes the time bin size in metadata rather than computing it dynamically from the data (as the reference does by taking the mode of all session rates).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is computed from the imaging rate rather than from the actual timestamps in the data. It uses `np.arange(n_timepoints) * frame_time` where `frame_time = 1.0 / imaging_rate`.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
# ...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The AI assumes perfectly regular sampling intervals based on the imaging rate. The reference uses actual timestamps from the behavioral data (`timestamps_curr - timestamps_curr[0]`).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. An array of evenly spaced time values is generated using the imaging frame time, starting at 0.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. This produces regularly spaced time values. The reference subtracts the first timestamp from all timestamps in the trial, using actual recorded times.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are assumed to be sampled at the same rate on the same time grid. If there is a length mismatch between neural and behavioral data, both are cropped to the minimum length.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    # ... crop all arrays to min_len
```

iii. The AI verifies length consistency and truncates if needed, similar to the reference approach.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the `SESSIONS_INFO` hardcoded dictionary (transcribed from the paper's `sessions_dict.py`), NOT from the `environment` behavioral time series in the data.

ii.
```python
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. The AI parses the scene name (e.g., "Env1_LocationA") to determine environment type. The reference reads the environment variable directly from the NWB behavioral data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to extract environment type (0 for Env1, 1 for Env2). For day 8 (environment switch), trials before trial 30 use the original environment and trials 30+ use the new environment.

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

iii. This approach should produce the same values as reading from the data, assuming the SESSIONS_INFO is correct and the switch happens at trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-based loop index over trials within a session, not from any stored variable.

ii.
```python
input_arr[2, :] = float(i)
```

iii. This matches the reference approach, which also uses the loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the loop index `i` is used directly as a float value, constant across all timepoints within a trial.

ii.
```python
input_arr[2, :] = float(i)
```

iii. Same as reference.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward event timestamps are matched to behavior timestamps to determine which trials were rewarded.

ii.
```python
reward_ts = nwb_data['reward_ts']
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1

prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. The AI determines reward per trial by checking if any reward timestamp falls within the trial's time range, then shifts by one trial. First trial's previous outcome is 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward outcome is determined per trial by timestamp matching, then shifted: the previous trial's reward outcome becomes the current trial's input. First trial defaults to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
# ...
input_arr[3, :] = float(prev_outcome[i])
```

iii. The reference uses a similar approach but checks reward indices within the trial's index range rather than using timestamp comparison. Both produce binary per-trial values.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries. Reward zone identity per trial comes from the hardcoded `SESSIONS_INFO` dictionary (parsed from scene names), NOT from the `reward_zone` time series in the data.

ii.
```python
rz_labels = parse_scene_reward_zones(scene, n_trials)
# ...
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The AI uses scene metadata to determine which reward zone (A, B, or C) applies to each trial, with switch at trial 30. The reference uses a Viterbi algorithm on the `reward_zone` position data to assign zone labels.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if before the zone, positive if past the zone, 0 if inside. Zone boundaries come from `REWARD_ZONES` dict (A: 80-130, B: 200-250, C: 320-370 cm).

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

iii. This is functionally equivalent to the reference's `compute_distance_to_reward_zone` function.

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

iii. The bin boundaries match the instructions. The reference uses `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. Both produce the same 7-bin discretization.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial. Both use the same `start:end` slice.

ii.
```python
pos = nwb_data['position'][start:end]
trial_n = neural[start:end, :].T
```

iii. Neural and behavioral data share the same sampling grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = nwb_data['position'][start:end]
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before discretization.

ii.
```python
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

iii. The reference does not clip position before discretizing. Clipping means negative positions (e.g., -500 during teleport, though these should be excluded by trial boundaries) are mapped to 0.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each across a 450 cm track: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 450/5 = 90
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The AI uses 90 cm bins starting at 0. The reference uses `np.digitize` with edges `[-inf, 50, 150, 250, 350, inf]`, producing 100 cm bins centered at different positions: (<50, 50-150, 150-250, 250-350, >350). These are different bin boundaries.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
pos = nwb_data['position'][start:end]
```

iii. Same indexing approach.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lck = licks_corrected[start:end]
```

iii. Same source as reference, but the AI uses corrected lick data (after lick sensor error correction).

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first applies lick sensor error correction: trials where >30% of samples have lick count >2 have their lick data set to NaN. NaNs are then replaced with 0. Finally, lick values are binarized (>0 = 1).

ii.
```python
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
        trial_licks = licks[start:end]
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > threshold:
            licks[start:end] = np.nan
            error_trials.append(i)
    return licks, error_trials
# ...
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. The paper describes this lick error correction procedure. The reference code does NOT implement it, simply binarizing the raw lick data.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
lck = licks_corrected[start:end]
```

iii. Same indexing approach.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the hardcoded `SESSIONS_INFO` dictionary, which encodes the scene name (including reward zone) for each subject and session day. NOT derived from the `reward_zone` time series in the data.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
rz_labels = parse_scene_reward_zones(scene, n_trials)
# ...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
output_arr[4, :] = rz_loc
```

iii. The AI transcribed the session-to-scene mapping from the paper's code. The reference uses a Viterbi algorithm on the position-when-in-reward-zone data to determine zone labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene names are parsed to extract reward zone identity. For switch sessions, the zone changes at trial 30. Zone labels (A/B/C) are mapped to integers (0/1/2).

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    if scene.endswith('_LocationA'):
        rz_labels[:] = 'A'
    # ... (similar for B, C)
    else:
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after
    return rz_labels
```

iii. This is a deterministic approach based on metadata, while the reference uses a data-driven Viterbi approach. Both should produce the same results if the metadata is correct and the switch always occurs at trial 30.

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
# ...
output_arr[5, :] = reward_per_trial[i]
```

iii. Same source as the reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls within the trial's time range (start_t to end_t). If so, reward_outcome = 1, else 0. The value is constant across all timepoints in the trial.

ii.
```python
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. The reference uses index-based matching (`np.any(isreward[idx])`), while the AI uses timestamp-based matching. Both achieve the same goal of determining per-trial binary reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is cropped to the minimum of the two lengths.
- **Short trials**: Trials with < 2 timepoints are skipped (vs. reference's threshold of 50).
- **Lick sensor errors**: Trials with excessive lick counts are set to 0 (reference does not implement this).
- **NaN lick values**: Replaced with 0 via `np.nan_to_num`.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    # crop all to min_len
# ...
if n_timepoints < 2:
    continue
# ...
lck = np.nan_to_num(lck, nan=0.0)
```

iii. The AI handles several edge cases. The key difference from the reference is the much lower minimum trial length threshold (2 vs 50 timepoints).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py (I/O bound, reading large arrays)
2. Interneuron identification (computing correlation for each neuron)
3. The full conversion loop over all 152 sessions
4. Saving the large pickle file (~9.8 GB)

ii. N/A

iii. The h5py-based loading is generally faster than pynwb. The interneuron correlation check adds computation not present in the reference.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron identification loop iterates over each neuron individually to compute correlations. This could be vectorized using matrix operations. The per-trial processing loop could also partially be vectorized by applying discretization to full session arrays before splitting into trials.

ii.
```python
def identify_interneurons(neural_data, speed_data, valid_mask, threshold):
    for c in range(n_neurons):
        r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

iii. The per-neuron correlation loop is the clearest candidate for vectorization.

## 13-c. What processing does the code repeat multiple times?

i. The code does NOT have a separate survey step like the reference. It loads each NWB file only once during conversion. However, it reads both deconvolved AND fluorescence data for every session (fluorescence is used for interneuron identification), which doubles the neural data I/O.

ii.
```python
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. Reading fluorescence data is additional work compared to the reference, which only reads deconvolved data.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs interneuron exclusion and lick sensor error correction, which the reference does not. While mentioned in the paper's methods, these are not part of the reference conversion pipeline and thus represent extra processing. The fluorescence data is loaded solely for interneuron identification and is otherwise discarded. The lick error correction zeros out lick data for affected trials, but since lick is binarized anyway, the downstream effect is that affected trials have all-zero lick values.

ii.
```python
fluorescence = np.concatenate(fluor_list, axis=1)
# Only used in:
is_interneuron = identify_interneurons(fluorescence, ...)
```

iii. The fluorescence data loading and interneuron filtering represent unnecessary processing relative to the reference solution.
