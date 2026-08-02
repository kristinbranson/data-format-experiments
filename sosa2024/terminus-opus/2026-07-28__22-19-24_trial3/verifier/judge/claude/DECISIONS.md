# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by iterating over sorted `sub-*` directories in `data/`, then sorted `.nwb` files within each. It parses the subject ID from the directory name (`sub-mX` -> `X`) and session number from the filename (`ses-YY`). Each NWB file is loaded using `h5py` (not `pynwb`). All behavioral and neural data arrays are read into memory per-session via direct HDF5 dataset access.

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
```

Loading via h5py:
```python
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']
```

iii. The AI's CONVERSION_NOTES indicate it found 152 NWB files across 11 subjects, matching expected counts. Using `h5py` instead of `pynwb` is a valid alternative for reading NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are identified by parsing the directory name `sub-mX`, extracting `X` as the subject ID. A `subject_map` dictionary tracks unique subjects and assigns indices.

ii.
```python
subject_id = sub_dir.replace('sub-m', '')
...
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)
```

iii. The AI confirmed 11 subjects matching the paper's count.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from filenames (`ses-YY`).

ii.
```python
ses_num = int(fname.split('_ses-')[1].split('_')[0])
```

iii. Each NWB file contains one session's data. 152 total sessions were found.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined from `trial_start` and `teleport` behavioral signals. Trial starts are indices where `trial_start > 0`. For each trial start, the code finds the next timepoint where `teleport > 0` to define the trial end. Trial data spans from `start` to `stop` (exclusive).

ii.
```python
trial_start_inds = np.where(trial_start_signal > 0)[0]
teleport_inds = np.where(teleport_signal > 0)[0]

for i in range(len(trial_start_inds)):
    start = trial_start_inds[i]
    future_teleports = teleport_inds[teleport_inds > start]
    if len(future_teleports) > 0:
        matched_starts.append(start)
        matched_teleports.append(future_teleports[0])
```

iii. The AI's trajectory discusses matching trial starts to the next teleport event, which is consistent with how trials are defined in the reference code (trial_start to teleport).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 3 timepoints are skipped. Sessions with fewer than 2 valid trials or fewer than 5 cells are also skipped.

ii.
```python
trial_len = stop - start
if trial_len < 3:
    continue
...
if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None
...
if n_trials < 2:
    print(f"  Skipping: only {n_trials} trials")
    return None
```

iii. The AI applies minimal trial filtering (only very short trials < 3 timepoints). No explicit quality filtering based on trial content.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` data in the ophys processing module. For multi-plane animals, data from all planes is concatenated.

ii.
```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. The CONVERSION_NOTES confirm that deconvolved calcium events are used, matching the reference paper's decoder approach.

## 2-b. How is the `neural` data processed?

i. Deconvolved data from all planes is concatenated along the neuron axis. Cells are filtered using the `iscell` classification. The neural data is transposed to (n_neurons, n_timepoints) and cast to float32.

ii.
```python
deconv = np.concatenate(deconv_parts, axis=1)
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The AI notes that multi-plane concatenation is needed for m17/m18. However, the `iscell` is read from a single `PlaneSegmentation` table rather than per-plane, which may cause issues for multi-plane animals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered based on the `iscell` variable from `ImageSegmentation/PlaneSegmentation`. Only ROIs with `iscell[:,0] == 1` are kept.

ii.
```python
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
```

iii. The `iscell` filter comes from Suite2P manual curation, matching the reference approach. However, reading iscell from a single PlaneSegmentation table (rather than per-plane) could be problematic for multi-plane animals if the table only covers one plane's ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Each trial's neural data starts at the trial_start index and ends at the teleport index, so alignment to trial start is inherent in the trial extraction.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Since trials are extracted starting from trial_start, the first timepoint is the trial start, matching the required alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is computed from the imaging rate stored in the NWB file: `dt = 1.0 / imaging_rate`. No temporal rebinning is applied; the native imaging rate (~15.5 Hz, ~64.5 ms) is used.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
dt_ms = 1000.0 / imaging_rate
```

iii. The CONVERSION_NOTES confirm the native imaging rate is used with no additional binning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is computed from the imaging rate, NOT from the behavioral timestamps. It uses `np.arange(trial_len) * dt` where `dt = 1.0 / imaging_rate`.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
time_from_start = np.arange(trial_len) * dt
```

iii. The AI derives time from the imaging rate rather than from the stored timestamps, assuming uniform sampling.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A linearly spaced time vector is created using the frame index multiplied by the time per frame (`dt`). The first timepoint is 0.

ii.
```python
time_from_start = np.arange(trial_len) * dt
trial_input[0, :] = time_from_start
```

iii. This assumes perfectly uniform sampling, which should be a good approximation for imaging data.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data are indexed using the same trial boundaries (`start:stop`), and time is computed from the frame index, so alignment is inherent.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
time_from_start = np.arange(trial_len) * dt
```

iii. Since both use the same indices, they are aligned by construction.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is NOT derived from the NWB behavioral data. Instead, it is derived from the `scene` name in the hardcoded `_all_sessions` dictionary (copied from reference code's `sessions_dict`).

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
env_per_trial = get_environment_from_scene(scene, n_trials)

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

iii. The AI used the reference code's sessions dictionary to determine environment type from the scene name, rather than reading the `environment` variable directly from the NWB file.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine if it's Env1 (0) or Env2 (1). For cross-environment switch sessions (containing `_to_Env`), trials before trial 30 use the first environment and trials 30+ use the second.

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
```

iii. This follows the reference code's logic for determining environment from scene names.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index (0-indexed), derived from the loop counter over matched trial boundaries.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = i
    trial_input[2, :] = float(trial_number)
```

iii. The trial number is simply the sequential index of the trial within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
trial_number = i
trial_input[2, :] = float(trial_number)
```

iii. The trial number is the 0-indexed position in the trial sequence.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps. Reward event timestamps are compared to trial time boundaries to determine if a reward was delivered.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
...
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
    rzone_active = np.any(rzone_data[start:stop+1] > 0)
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The AI also checks that the reward zone was active during the trial, adding an extra condition beyond just checking for reward events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome is determined by checking if any reward timestamp falls within the trial's time range AND the reward zone was active. The previous trial outcome is then a simple shift: `prev_outcome[i] = isreward[i-1]`. The first trial defaults to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
trial_input[3, :] = float(prev_out)
```

iii. This vectorized shift approach is equivalent to checking the previous trial's reward status.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and reward zone coordinates. The reward zone coordinates come from the hardcoded `_all_sessions` dictionary (scene name parsing), NOT from the `reward_zone` behavioral variable in the NWB file.

ii.
```python
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI used the reference code's sessions dictionary to determine reward zone locations from scene names, applying the switch at trial 30 for switch sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone. Distance is negative before the zone, 0 inside the zone, and positive after the zone.

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

iii. This is the standard signed distance computation, matching the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments:
- 0: < -50 cm
- 1: [-50, -10)
- 2: [-10, 0)
- 3: == 0
- 4: (0, 10]
- 5: (10, 50]
- 6: > 50

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

iii. The bin boundaries match the instructions. Minor differences in boundary inclusivity compared to using `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data use the same trial indices (`start:stop`), so alignment is inherent.

ii.
```python
trial_pos = position[start:stop]
trial_neural = neural_all[start:stop, :].T
```

iii. Both are indexed identically within each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. The position variable directly records the animal's position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, TRACK_LENGTH]` (i.e., `[0, 450]`) before discretization.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

iii. The clipping ensures positions outside the track range are mapped to the extreme bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.linspace(0, 450, 6)` giving edges `[0, 90, 180, 270, 360, 450]`:
- 0: [0, 90)
- 1: [90, 180)
- 2: [180, 270)
- 3: [270, 360)
- 4: [360, 450]

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The AI uses 90cm-wide bins spanning [0, 450]. This differs from the reference which uses bins with edges at [-inf, 50, 150, 250, 350, inf] (100cm-wide central bins).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial indices as neural data. No additional alignment needed.

ii.
```python
trial_pos = position[start:stop]
```

iii. Aligned by construction via shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error detection from the reference code: if >35% of samples in a trial have lick values > 2, all lick values for that trial are set to 0. Otherwise, values > 1 are clipped to 1. The result is binary (0/1).

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The AI incorporated lick sensor error correction from the reference code's `correct_lick_sensor_error` function.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial indices as neural data. No additional alignment needed.

ii.
```python
trial_lick = lick_raw[start:stop].copy()
```

iii. Aligned by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the `scene` name in the hardcoded `_all_sessions` dictionary, NOT from the `reward_zone` behavioral variable in the NWB file. The scene name encodes the reward zone location (A, B, or C) and whether a switch occurs.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)

def get_reward_zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    if '_to_' in scene or '_to_Env' in scene:
        before_loc = parts[0][-1]
        after_loc = parts[1][-1]
        ...
        rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
        rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
    else:
        loc = scene[-1]
        ...
        rz_labels[:] = ZONE_TO_LABEL[zone]
```

iii. The AI imported the sessions dictionary from the reference code to deterministically assign reward zones based on session scene names.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene names are parsed to extract location letters (A, B, C). For switch sessions (containing `_to_`), the first 30 trials use the pre-switch location and remaining trials use the post-switch location. Labels are mapped to indices: A=0, B=1, C=2.

ii.
```python
rz_label_idx = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    if rz_labels[i] == 'A':
        rz_label_idx[i] = 0
    elif rz_labels[i] == 'B':
        rz_label_idx[i] = 1
    elif rz_labels[i] == 'C':
        rz_label_idx[i] = 2
```

iii. The fixed switch at trial 30 matches the paper's description.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps and the `reward_zone` behavioral variable.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. Combines reward event timestamps with reward zone activation status.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks (1) if any reward timestamp falls within the trial's time range AND (2) if the reward zone was active during the trial. Both conditions must be true for a reward outcome of 1.

ii.
```python
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The extra `rzone_active` check is meant to distinguish true omissions from trials where the mouse never entered the reward zone.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Very short trials**: Trials with < 3 timepoints are skipped.
- **Sessions with few cells**: Sessions with < 5 cells are skipped.
- **Sessions with few trials**: Sessions with < 2 valid trials are skipped.
- **Lick sensor errors**: Trials with >35% samples having lick > 2 have all lick values set to 0.
- **Processing errors**: Individual sessions that raise exceptions are skipped with error logging.

ii.
```python
if trial_len < 3:
    continue
if n_cells < 5:
    return None
if n_trials < 2:
    return None
...
try:
    result = process_session(...)
except Exception as e:
    print(f"  ERROR: {e}")
    continue
```

iii. The AI uses defensive coding with try/except around session processing to handle unexpected data issues.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files via h5py** - reading large neural data arrays from disk
2. **Processing each session** - iterating over all 152 sessions
3. **Saving the pickle file** - the final dataset is ~9.8 GB

ii. N/A

iii. The AI reports total processing time and per-session timing. The full conversion took about 90 seconds.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial sequentially to extract data, compute inputs/outputs. Some operations (discretization, distance computation) could be applied to full-session arrays before splitting. The reward determination loop over trials could be vectorized using broadcasting.

ii.
```python
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    ...
```

iii. The per-trial loop is natural given variable-length trials but adds overhead.

## 13-c. What processing does the code repeat multiple times?

i. The code does not have a separate survey step that re-reads files. Each NWB file is read once during `process_session`. However, the `_all_sessions` dictionary is a large hardcoded block that could have been imported from the reference code.

ii. N/A

iii. The AI's approach of processing in a single pass is more efficient than the reference's survey-then-convert approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads several behavioral variables that are not used in the final output:
- `autoreward` is loaded but never used
- `scanning` is loaded but never used
- `env_data` (the NWB environment variable) is loaded but the AI uses scene-derived environment instead

ii.
```python
autoreward = behav['autoreward']['data'][()]
scanning = behav['scanning']['data'][()]
env_data = behav['environment']['data'][()]  # loaded but not used - scene-based env used instead
```

iii. These are minor inefficiencies from loading all available behavioral variables.
