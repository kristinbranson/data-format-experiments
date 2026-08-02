# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by globbing `data/sub-*//*.nwb` and iterates over them one at a time. Each NWB file corresponds to one session. Data is loaded using `h5py` to read HDF5/NWB format. Behavioral timeseries (position, speed, lick, reward_zone, trial number, trial_start, teleport, environment, scanning) and neural data (Fluorescence, Neuropil, Deconvolved per plane, iscell, planeIdx) are loaded from each file.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
# ...
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
```

Inside `process_session`:
```python
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    # ... loads all behavioral and neural data
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that NWB files are organized as `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`, each containing behavioral timeseries and multi-plane optical physiology data.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is read from each NWB file's `general/subject/subject_id` field. A set of unique subjects is accumulated across all sessions, then sorted to create the `subjects` list. Each session is assigned a `subject_idx` mapping to this list.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
# ...
subjects_set = set()
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        subjects_set.add(result['subject_id'])
subjects = sorted(subjects_set)
# ...
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The AI identified 11 subjects (m3, m4, m7, m11-m15, m17-m19) matching the paper's "n = 11 mice" for the switch group.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. The AI processes each file independently via `process_session()` and collects results into lists. Sessions that fail quality checks (< 2 cells, < 2 trials) are skipped.

ii.
```python
for nwb_path in nwb_files:
    fname = os.path.basename(nwb_path)
    session_label = fname.replace('_behavior+ophys.nwb', '')
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. The AI documented finding 152 total sessions (14 per subject, except m11 with 12 starting from ses-03), matching the paper.

## 1-d. How are the data split into trials?

i. Trials are identified using binary flags `trial_start_flag` and `teleport_flag` from the behavioral timeseries. Trial boundaries are from the first timepoint where `trial_start_flag > 0` to the corresponding `teleport_flag > 0` timepoint. The number of trials is the minimum of the number of starts and teleports.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
# ...
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
```

iii. The AI documented using trial_start and teleport flags as trial boundaries, consistent with the reference code's approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if `e <= s` (teleport before or at trial start) or `(e - s) < 2` (fewer than 2 timepoints). There is no additional quality-based trial filtering (e.g., no speed-based exclusion of entire trials). Lick sensor errors are handled per-trial by zeroing out lick data rather than excluding the trial.

ii.
```python
if e <= s or (e - s) < 2:
    # Still need to track prev_trial_rewarded
    trial_start_time = pos_timestamps[s] if s < len(pos_timestamps) else 0
    trial_end_time = pos_timestamps[min(e, len(pos_timestamps) - 1)]
    was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
    prev_trial_rewarded = int(was_rewarded)
    continue
```

iii. The AI noted in CONVERSION_NOTES.md Step 5 that "No speed filtering for conversion" is applied because speed is an output to be decoded. Trial exclusion is minimal, only removing degenerate trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the pre-computed **deconvolved calcium events** stored in the NWB files at `processing/ophys/Deconvolved/plane{N}/data`. The AI also loads raw Fluorescence and Neuropil data, but only for the purpose of computing dF/F for interneuron detection.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]  # (n_timepoints, n_rois_plane)
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
# ...
deconv_all = np.concatenate(deconv_list, axis=1)
# ...
neural_data = deconv_cells[final_cell_mask]  # (n_final_cells, n_timepoints)
```

iii. The AI justified this in CONVERSION_NOTES.md Step 4: "The NWB data already has deconvolved events pre-computed via the reference pipeline (suite2p OASIS). We should use these directly." And Step 5: "Neural data = deconvolved events: The paper's decoder uses deconvolved events, and NWB has them pre-computed."

## 2-b. How is the `neural` data processed?

i. The deconvolved events are used directly from the NWB files without additional processing. They are concatenated across planes for multi-plane sessions, filtered by iscell and interneuron masks, then sliced per trial. Data is cast to float32 for storage.

ii.
```python
deconv_all = np.concatenate(deconv_list, axis=1)
# ...
deconv_cells = deconv_all[:, cell_mask].T  # (n_cells, n_timepoints)
# ...
neural_data = deconv_cells[final_cell_mask]  # (n_final_cells, n_timepoints)
# ...
trial_neural = neural_data[:, s:e]  # (n_cells, n_timepoints)
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The AI chose to use pre-computed deconvolved events since they were already available in the NWB files, matching what the reference code decoder used (`sess.timeseries['events']`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Suite2p `iscell` classification (column 0 == 1), and (2) interneuron exclusion based on Pearson correlation between dF/F and speed exceeding 0.5. dF/F is computed per-trial using the maximin baseline method (neuropil subtraction with coefficient 0.7, Gaussian smooth sigma=15, min-max filter with 300-sample window, then post-dF/F smooth sigma=2).

ii.
```python
# Stage 1: iscell
cell_mask = iscell[:, 0].astype(bool)

# Stage 2: Interneuron detection
dff_full = np.full_like(F_cells, np.nan)
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])

is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

iii. The AI documented matching the reference code's interneuron detection: "Pearson corr >0.5 with speed" and "0.42+/-0.85% of cells excluded".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the trial. For each trial, neural data is sliced from the trial start index to the teleport index: `neural_data[:, s:e]` where `s = trial_start_inds[i]` and `e = teleport_inds[i]`.

ii.
```python
trial_neural = neural_data[:, s:e]  # (n_cells, n_timepoints)
```

iii. The instructions specify "Temporally align based on start of the trial." The AI's metadata confirms: `'temporal_alignment_event': 'Start of trial (entry to linear track)'` and `'off_start': 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate (~15.5 Hz, ~64.48 ms per frame). No temporal rebinning is applied. For multi-plane sessions (m17, m18), the effective per-plane rate is `imaging_rate / n_planes`.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. The AI documented: "Time bin = imaging frame: ~64.5 ms per frame at ~15.5 Hz. This matches the native sampling rate."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is not derived from any raw data variable directly. It is computed as the frame index within the trial divided by the effective imaging rate.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The AI's mapping plan states: "Time from trial start | input[0] | (t - trial_start) / imaging_rate, continuous seconds | Time-varying".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple computation: create integer indices 0 to n_t-1, divide by the effective imaging rate (Hz) to convert to seconds.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. No complex processing needed. The AI uses frame indices and the known imaging rate for conversion.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural data and time are indexed from the same trial start index `s` to teleport index `e`. The time array has the same number of elements (`n_t = e - s`) as neural timepoints, ensuring 1:1 alignment.

ii.
```python
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
# neural: trial_neural = neural_data[:, s:e]  # same n_t timepoints
```

iii. Alignment is inherent because both use the same trial window indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral timeseries in the NWB file at `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = bts['environment/data'][:]
# ...
trial_env = environment[s:e]
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
```

iii. The AI mapped: "Environment (morph) | input[1] | 0=ENV1, 1=ENV2, per-trial scalar | From `environment` behavioral TS".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the environment values within the trial window are extracted. Negative values are excluded, and the median of the remaining values is taken as the per-trial environment type. This is then broadcast across all timepoints in the trial.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
input_arr[1, :] = env_type  # broadcast
```

iii. Using median of non-negative values is a robust way to get the per-trial environment type value (0 or 1).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the loop index `i` (0-indexed) over trials within each session, NOT from the `trial number` behavioral timeseries in the NWB file.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number  # broadcast
```

iii. The AI uses the loop iteration counter as trial number, which represents the 0-indexed trial position within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the 0-indexed loop variable `i`, cast to float32 and broadcast across all timepoints in the trial.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number  # broadcast
```

iii. No further processing. This counts all trials including those that get skipped (invalid trials increment `i` but don't produce output), so the trial numbers may have gaps.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral timeseries (reward delivery timestamps) by checking whether any reward timestamp falls within the previous trial's time window.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
# ...
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
# ...
prev_trial_rewarded = int(was_rewarded)
prev_outcome = np.float32(prev_trial_rewarded)
input_arr[3, :] = prev_outcome  # broadcast
```

iii. The AI tracks the reward outcome of the previous trial via `prev_trial_rewarded`, initialized to 0 for the first trial. Updated at the end of each trial loop iteration.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward is detected by checking if any entry in `reward_timestamps` falls within the trial's time window `[trial_start_time, trial_end_time]`. The previous trial's reward outcome (0=omitted, 1=rewarded) is stored and used as input for the current trial. The first trial defaults to 0 (no previous reward).

ii.
```python
prev_trial_rewarded = 0  # For the first trial, assume no previous reward
# ...
for i in range(n_trials):
    # ... at end of loop:
    prev_trial_rewarded = int(was_rewarded)
```

iii. The AI correctly tracks reward state across trials. Skipped trials also update `prev_trial_rewarded`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from: (1) `position` behavioral timeseries, and (2) reward zone coordinates determined from the session's scene name (parsed from the NWB `identifier` field) using the `REWARD_ZONE_DICT`.

ii.
```python
trial_pos = position[s:e]
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI mapped: "Distance to reward zone | output[0] | Discretized into 7 bins, time-varying | Compute from position and reward zone coords."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone: negative if before the zone, 0 if inside, positive if after.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end  # positive
    return dist
```

iii. This matches the concept of "distance to any location in the reward zone" from the instructions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins as specified in the instructions:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 cm to < 0 cm
- 3: 0 cm (inside reward zone)
- 4: >0 cm to +10 cm
- 5: +10 to +50 cm
- 6: > +50 cm

ii.
```python
def discretize_distance_to_rz(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. Exactly matches the bin specifications in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from the position values at the same trial indices (s:e) as the neural data, ensuring frame-by-frame alignment.

ii.
```python
trial_pos = position[s:e]
# ...
output_arr[0, :] = dist_to_rz_binned  # same n_t as neural
```

iii. Behavioral and neural data share the same timebase in the NWB file (both at the imaging frame rate).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral timeseries at `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = bts['position/data'][:]
# ...
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. Direct use of the position variable from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw position values (in cm, range [0, 450]) are discretized into 5 equal-sized bins using `np.digitize` with bin edges at [0, 90, 180, 270, 360, 450].

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The instructions specify "Absolute position in corridor, discretized into 5 equal-sized bins, time-varying."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal bins of 90 cm each:
- 0: 0-90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: 360-450 cm

ii.
```python
POSITION_BINS = 5
POSITION_BIN_EDGES = np.linspace(0, TRACK_LENGTH, POSITION_BINS + 1)  # [0, 90, 180, 270, 360, 450]
```

iii. Matches the instruction to use "5 equal-sized bins" over the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same alignment as other behavioral variables - position is extracted from the same trial window indices as neural data.

ii.
```python
trial_pos = position[s:e]
output_arr[1, :] = pos_binned  # same n_t as neural
```

iii. All data share the same timebase.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral timeseries at `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick_raw = bts['lick/data'][:]
# ...
trial_lick = lick[s:e].copy()
```

iii. The AI documented this as "From `lick` behavioral TS, cap at 1".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two-step processing: (1) Lick sensor error correction - if >35% of samples in a trial have cumulative lick count > 2, set all licks in that trial to 0. (2) Cap all lick values at 1 to produce a binary (0/1) output.

ii.
```python
# Lick sensor error correction
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0  # set to 0 instead of NaN for decoder output
# Cap licks at 1 (binary)
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The AI noted this matches the reference code's lick sensor error correction, though the reference code set bad lick trials to NaN. The AI chose 0 instead for decoder compatibility.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data is extracted from the same trial indices as neural data.

ii.
```python
trial_lick = lick[s:e].copy()
output_arr[3, :] = lick_binned
```

iii. Same timebase alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `identifier` field of the NWB file (which encodes the session scene name, e.g., "Env1_LocationB_to_A"), combined with the trial index and the default change trial (30).

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
# ...
rz_loc = rz_label_to_idx(rz_labels[i])  # 0=A, 1=B, 2=C
output_arr[4, :] = rz_loc  # broadcast
```

iii. The AI implemented `get_reward_zones()` to mirror the reference code's `behavior.get_reward_zones()`, mapping scene names to reward zone coordinates with switch at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine pre-switch and post-switch reward zone locations. Trials before index 30 get the first zone, trials at or after index 30 get the second zone. Zone labels (A, B, C) are mapped to indices (0, 1, 2). For single-location sessions, all trials get the same zone.

ii.
```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    if 'Location' in scene and '_to' not in scene:
        loc = scene.split('Location')[-1]
        rz_coords[:] = REWARD_ZONE_DICT[loc]
        rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    # ... similar for other switch combinations
```

iii. The AI documented using the same logic as the reference code for zone assignment with change_trial=30.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` and `Reward/data` from the NWB file's behavioral timeseries, combined with the trial's time window from `position/timestamps`.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
reward_data = bts['Reward/data'][:]
# ...
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
reward_outcome = int(was_rewarded)
```

iii. The AI checks whether any reward timestamp falls within the trial's time window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any entry in the global reward timestamps array falls within `[trial_start_time, trial_end_time]`. If yes, the trial is marked as rewarded (1), otherwise not (0). This is broadcast as a per-trial scalar across all timepoints.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
# ...
output_arr[5, :] = reward_outcome  # broadcast
```

iii. The AI validated that ~84.3% of trials are rewarded and ~15.7% are omitted, matching the paper's ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Neural/behavioral length mismatch**: When neural and behavioral data differ in length (by 1 frame in multi-plane recordings), both are truncated to the minimum length.
- **Lick sensor errors**: Trials with >35% of samples having lick > 2 get lick data zeroed out.
- **Invalid trials**: Trials where teleport index <= start index or trial length < 2 are skipped.
- **Missing environment values**: Negative environment values are excluded; if all negative, defaults to 0.
- **Session quality**: Sessions with < 2 cells or < 2 valid trials are skipped entirely.

ii.
```python
# Length mismatch
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    position = position[:n_timepoints_total]
    # ...

# Invalid trials
if e <= s or (e - s) < 2:
    continue

# Lick errors
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
```

iii. The AI documented handling multi-plane length mismatches in CONVERSION_NOTES.md Step 10.

## 13-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, the most time-consuming steps are:
1. **NWB file I/O**: Loading large HDF5 files (especially for multi-plane sessions like m18 with ~2000+ neurons, taking 8-12s per session).
2. **dF/F computation for interneuron detection**: Computing dF/F per trial for all iscell neurons, involving Gaussian smoothing and min/max filtering.
3. **Interneuron detection**: Vectorized Pearson correlation across all cells.

ii.
```python
# dF/F computation (per trial loop)
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

iii. The AI noted optimizing vectorized interneuron detection from 18.2s to 7.1s for 2 sessions. Total conversion time was 681.4s (~11 minutes) for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the per-trial dF/F computation loop. Currently it loops over all trials to compute dF/F for interneuron detection. This could potentially be done on the full continuous trace rather than per-trial, though the per-trial approach matches the reference code's behavior.

ii.
```python
# This loop computes dF/F per trial
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    if e <= s:
        continue
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

iii. The AI noted vectorizing interneuron detection but the dF/F per-trial loop remains.

## 13-c. What processing does the code repeat multiple times?

i. The dF/F computation is done once for all iscell neurons for interneuron detection, but the result is not reused. The interneuron detection computes correlations which is done only once. The trial loop processes each trial independently, which is inherently sequential but not repeated.

ii. No significant repeated processing identified - each major computation happens once.

iii. The AI optimized the interneuron detection to be vectorized rather than per-cell.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary processing is:
1. **dF/F computation**: Computed for all iscell neurons across all trials, but only used for interneuron detection (correlation with speed). The actual neural output uses deconvolved events.
2. **Loading Fluorescence and Neuropil data**: Loaded for all planes but only needed for dF/F computation for interneuron detection.
3. **Reward data variable** (`reward_data`): Loaded but never used (only `reward_timestamps` is used).

ii.
```python
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]  # Only used for dF/F -> interneuron detection
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]   # Only used for dF/F -> interneuron detection
reward_data = bts['Reward/data'][:]  # Never used
rzone_cumul = bts['reward_zone/data'][:]  # Never used in processing
scanning = bts['scanning/data'][:]  # Never used in processing
```

iii. The dF/F computation is necessary for matching the reference code's interneuron detection, but it is computationally expensive and only needed for the correlation test. The fluorescence data loading is the most impactful unnecessary I/O.
