# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by iterating over `data/sub-mX/` directories, parsing subject IDs and session numbers from filenames matching pattern `sub-mX_ses-YY_behavior+ophys.nwb`. Each NWB file is opened with `h5py.File()`. Behavioral data is loaded from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys`. All 152 NWB files across 11 subjects are loaded sequentially.

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

iii. The AI identified 11 subjects with 152 total NWB files by exploring the data directory structure. This matches the reference paper's description of 11 switch-condition mice.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the `sub-mX` directory name. A subject mapping dictionary tracks unique subjects and assigns indices. Subject names are stored as `mX` strings (e.g., `m11`, `m3`).

ii.
```python
subject_map = {}
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)
subject_idx.append(subject_map[sub_name])
```

iii. The AI correctly identified the GCAMP_N to m_N naming convention from the reference code's sessions_dict (e.g., GCAMP11 -> m11).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from the filename (e.g., `ses-03`). The session number corresponds to `exp_day` in the reference code's `sessions_dict`. The AI processes all 152 sessions (m11 has 12, others have 14 each).

ii.
```python
ses_num = int(fname.split('_ses-')[1].split('_')[0])
```

iii. The AI verified that session numbers in NWB files map to exp_day in the sessions_dict, and that m11 starts from session 03 (12 sessions total) while other subjects have 14 sessions.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by finding indices where the `trial_start` signal is > 0 (trial starts) and the `teleport` signal is > 0 (trial ends). Each trial start is matched to the next teleport after it. Trial data runs from `start` to `stop` (exclusive of stop).

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

iii. The AI matched the reference code's approach of using `sess.trial_start_inds` and `sess.teleport_inds` to define trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if they have fewer than 3 timepoints (`trial_len < 3`). Sessions are skipped if they have fewer than 2 valid trials or fewer than 5 cells. There is no filtering based on autoreward status or scanning signal, despite these variables being loaded from NWB.

ii.
```python
trial_len = stop - start
if trial_len < 3:
    continue

if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None

if n_trials < 2:
    print(f"  Skipping: only {n_trials} trials")
    return None
```

iii. The AI applied minimal trial filtering. The reference code does not explicitly filter short trials but the lick sensor error correction and speed masking effectively remove low-quality data. The AI noted it chose not to apply speed masking since the decoder needs continuous time series.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` data in the NWB ophys processing group, which contains OASIS-deconvolved calcium events. For multi-plane animals (m17, m18), data from multiple planes are concatenated.

ii.
```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. The AI identified that `sess.timeseries['events']` in the reference code corresponds to the `Deconvolved` data in NWB files. The reference code processes raw fluorescence through dF/F and OASIS deconvolution, but these steps are already applied in the NWB files.

## 2-b. How is the `neural` data processed?

i. The deconvolved events are filtered by `iscell` (Suite2P cell classification), keeping only ROIs classified as cells. The data is transposed from (timepoints, neurons) to (neurons, timepoints) and cast to float32. No additional processing (no speed masking, no NaN masking, no smoothing) is applied.

ii.
```python
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]  # (n_timepoints, n_cells)
# ...per trial:
trial_neural = neural_all[start:stop, :].T.astype(np.float32)  # (n_cells, trial_len)
```

iii. The AI decided not to apply speed masking or NaN masking that the reference code applies, reasoning that the decoder needs continuous time series data. The reference code's `get_timeseries_data` masks out timepoints where speed < 2 cm/s and where lick data is NaN.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cell filtering via `iscell[:, 0] == 1` from Suite2P manual curation is applied. Sessions with fewer than 5 cells are skipped. No interneuron exclusion is performed.

ii.
```python
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
cell_mask = iscell[:, 0] == 1
n_cells = cell_mask.sum()
if n_cells < 5:
    return None
```

iii. The AI noted that the reference paper mentions 155-2172 putative pyramidal neurons per session after manual curation and interneuron exclusion. The AI's range of 155-2341 is slightly wider, which the AI attributed to the max being pre-interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data spans from the `trial_start` index to the `teleport` index (exclusive). Time point 0 in each trial corresponds to the trial start signal.

ii.
```python
start = trial_start_inds[i]
stop = teleport_inds[i]
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The AI followed the instruction to "temporally align based on start of the trial" and set `temporal_alignment_event` to `'trial_start'` in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging rate (~15.5 Hz, ~64.5 ms per frame). No temporal rebinning is applied.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate  # seconds per frame
dt_ms = 1000.0 / imaging_rate  # in metadata
```

iii. The AI noted the imaging rate of 15.5078125 Hz from the NWB file, giving ~64.48 ms per frame. This matches the reference code which also uses the native imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This is computed from the frame index within each trial and the imaging rate (dt = 1/imaging_rate). No raw data variable is directly used; it is synthesized from the trial boundary indices and the known frame duration.

ii.
```python
time_from_start = np.arange(trial_len) * dt  # (trial_len,)
```

iii. The AI computed time as frame indices multiplied by the frame duration, starting from 0 at trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple linear ramp is created: `frame_index * dt` where `dt = 1.0 / imaging_rate`. The first timepoint is 0.0 seconds and each subsequent timepoint increments by dt (~0.0645 seconds).

ii.
```python
time_from_start = np.arange(trial_len) * dt
```

iii. No special processing or alignment is needed since time is defined relative to trial start, which is the alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time from trial start is inherently aligned with the neural data since both share the same frame indices within each trial. Frame 0 of neural data corresponds to time 0.0 seconds.

ii.
```python
trial_input[0, :] = time_from_start  # shape matches trial_len
```

iii. The alignment is exact because both neural data and time share the same indexing (start:stop within each trial).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the `scene` attribute in the reference code's `sessions_dict`, NOT from the NWB `environment` behavioral time series. The scene name encodes whether the session is in Env1 or Env2.

ii.
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

iii. The AI used the reference code's approach of deriving environment from the scene name in sessions_dict. The reference code's `env_morph_dict` maps Env1=0, Env2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine environment type (0=Env1, 1=Env2). For cross-environment switch sessions (containing `_to_Env`), the environment changes at trial 30 (`CHANGE_TRIAL`). The environment is set per-trial but broadcast across timepoints in the input array.

ii.
```python
env_per_trial = get_environment_from_scene(scene, n_trials)
# ...
trial_input[1, :] = float(env_type)  # broadcast per-trial value
```

iii. The AI matched the reference code's `env_morph_dict` mapping and the change at trial 30 for switch sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `i` when iterating over matched trial start/teleport pairs. It is NOT derived from the NWB `trial number` behavioral time series.

ii.
```python
trial_number = i  # 0-indexed loop variable
trial_input[2, :] = float(trial_number)
```

iii. The AI used the 0-indexed loop index as the trial number. The NWB `trial number` field was loaded but not used for this input.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the 0-indexed loop variable `i` over matched trial starts and teleports. It is broadcast across all timepoints in the input array. Note that if short trials (< 3 frames) are skipped, the trial numbers may have gaps.

ii.
```python
trial_number = i
trial_input[2, :] = float(trial_number)
```

iii. The AI used a simple 0-based index. The reference code's `trial_ids` in `get_timeseries_data` similarly uses the loop index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the `isreward` array, which itself is computed from: (1) the `Reward` event timestamps, (2) the behavioral timestamps, and (3) the `reward_zone` data in the NWB file. The previous trial's reward status becomes the current trial's input.

ii.
```python
isreward = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
    rzone_active = np.any(rzone_data[start:stop+1] > 0)
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0

prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
```

iii. The AI matched the reference code's `get_trial_types` logic, which checks both `reward > 0` AND `rzone > 0` to determine reward delivery. The first trial defaults to 0 (no previous outcome).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the AI checks if any reward events fall within the trial's time window AND if the reward zone was active (rzone > 0). The previous trial's binary reward outcome (0=omission, 1=rewarded) is shifted forward by one trial. The first trial has previous outcome = 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
trial_input[3, :] = float(prev_out)
```

iii. The AI's logic for determining reward delivery matches the reference code's `get_trial_types`, which uses `np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from: (1) the `position` behavioral time series, and (2) reward zone coordinates determined from the `sessions_dict` scene mapping (reward zones A=[80,130], B=[200,250], C=[320,370]).

ii.
```python
trial_pos = position[start:stop]
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI used the reference code's `reward_zone_dict` with zone labels X=[80,130], Y=[200,250], Z=[320,370] mapped to A, B, C.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if position is before the reward zone start, 0 if within the zone, positive if past the zone end. The distance is to the nearest edge of the reward zone.

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

iii. The AI computed signed distance to reward zone edges, consistent with the instruction's specification of "distance to any location in the reward zone."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is discretized into 7 bins matching the instruction specification exactly.

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

iii. The bin edges match the instruction exactly: < -50, -50 to -10, -10 to 0, 0, >0 to 10, 10 to 50, >50.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance to reward zone is computed from position data at the same frame indices as the neural data (start:stop within each trial), so it is inherently aligned.

ii.
```python
trial_pos = position[start:stop]
# ... same start:stop as neural data
trial_output[0, :] = dist_bins
```

iii. Alignment is exact because behavioral and neural data share the same timestamps in the NWB file.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the `position` behavioral time series in the NWB file.

ii.
```python
trial_pos = position[start:stop]
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
```

iii. Position data comes directly from `behav['position']['data']` in the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to [0, 450] (track length) before discretization.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

iii. The clipping ensures position values stay within the 450 cm track length before binning.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is discretized into 5 equal-sized bins using `np.linspace(0, 450, 6)` to create edges, then `np.digitize`.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. This creates bins [0-90), [90-180), [180-270), [270-360), [360-450] cm, matching the instruction for "5 equal-sized bins."

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position bins are computed from the same frame indices (start:stop) as neural data, ensuring frame-by-frame alignment.

ii.
```python
trial_pos = position[start:stop]  # same indices as neural
trial_output[1, :] = pos_bins
```

iii. Same alignment mechanism as distance to reward zone.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick data is derived from the `lick` behavioral time series in the NWB file.

ii.
```python
lick_raw = behav['lick']['data'][()]
trial_lick = lick_raw[start:stop].copy()
```

iii. The NWB `lick` field corresponds to `sess.timeseries['licks']` in the reference code.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick sensor error correction is applied: if >35% of samples in a trial have cumulative lick count > 2, all licks in that trial are set to 0. Remaining lick values > 1 are clipped to 1, creating a binary output.

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0  # Set to 0 for error trials
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The AI matched the reference code's lick sensor error threshold of 0.35. However, the reference code sets error trials to NaN (which are then masked out), while the AI sets them to 0 to maintain continuous time series for the decoder.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data uses the same frame indices (start:stop) as neural data.

ii.
```python
trial_lick = lick_raw[start:stop].copy()
trial_output[3, :] = trial_lick
```

iii. Same alignment mechanism as other behavioral outputs.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the `scene` attribute in the reference code's `sessions_dict`, NOT from the NWB `reward_zone` behavioral time series. The scene name encodes the reward zone location (A, B, or C).

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
rz_label_idx[i] = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The AI correctly used the sessions_dict to determine reward zone labels, matching the reference code's `get_reward_zones` function which also derives zones from `sess.scene`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract location (A, B, or C). For switch sessions (containing `_to_`), the location changes at trial 30. Labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    if '_to_' in scene or '_to_Env' in scene:
        parts = scene.split('_to_')
        before_loc = parts[0][-1]
        after_loc = parts[1][-1]
        # ... assign before/after zones at change_trial boundary
    else:
        loc = scene[-1]
        zone = LOCATION_TO_ZONE[loc]
        rz_coords[:] = REWARD_ZONE_DICT[zone]
        rz_labels[:] = ZONE_TO_LABEL[zone]
```

iii. The AI replicated the reference code's `get_reward_zones` logic, including the switch at `change_trial=30`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward` event data (timestamps and values) and the `reward_zone` behavioral time series in the NWB file.

ii.
```python
reward_data = behav['Reward']['data'][()]
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
```

iii. The AI used both reward events and reward zone activity to determine reward outcome, matching the reference code's `get_trial_types` approach.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks if any reward event timestamps fall within the trial's time window AND if the reward zone was active (rzone > 0). Both conditions must be true for `isreward = 1`. This distinguishes genuine reward delivery from spurious signals.

ii.
```python
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
    rzone_active = np.any(rzone_data[start:stop+1] > 0)
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The AI's logic matches the reference `get_trial_types` which checks `np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of data issues are handled:
- **Lick sensor errors**: Trials where >35% of samples have cumulative lick > 2 have licks set to 0 (reference sets to NaN).
- **Short trials**: Trials with < 3 timepoints are skipped.
- **Missing sessions**: m11 is missing sessions 01-02; the code handles this gracefully by only processing existing NWB files.
- **Multi-plane data**: m17 and m18 have two imaging planes, handled by concatenating plane data.
- **Low cell count**: Sessions with < 5 cells are skipped.
- **Autoreward data**: Loaded but not used for filtering.
- **Scanning data**: Loaded but not used for filtering.

ii.
```python
# Short trial handling
if trial_len < 3:
    continue

# Lick error handling
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0

# Multi-plane handling
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. The AI documented handling of multi-plane animals and lick sensor errors in CONVERSION_NOTES.md. The initial implementation had a bug with multi-plane data (only loading plane0) which was fixed during Step 10 review. The sessions_dict was also initially manually copied with errors, which was corrected.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files from disk via `h5py`, particularly reading the full deconvolved neural data arrays. Each `h5py.File()` open + read of behavioral and neural data takes the bulk of per-session processing time. Total processing for 152 sessions takes approximately 90 seconds.

ii.
```python
f = h5py.File(nwb_path, 'r')
# Loading all data arrays at once:
position = behav['position']['data'][()]
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. The AI reported ~0.6s per session on average, with total processing time of ~90s for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop in `process_session` iterates over each trial to extract data slices and compute outputs. The reward determination loop (checking reward timestamps per trial) could potentially be vectorized. The trial matching loop (matching starts to teleports) could also be vectorized using searchsorted.

ii.
```python
# Per-trial reward computation - could be vectorized
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
    rzone_active = np.any(rzone_data[start:stop+1] > 0)
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0

# Trial matching loop - could use np.searchsorted
for i in range(len(trial_start_inds)):
    start = trial_start_inds[i]
    future_teleports = teleport_inds[teleport_inds > start]
```

iii. The AI did not optimize these loops since total processing time was already ~90s, well within the 15-minute target.

## 13-c. What processing does the code repeat multiple times?

i. The `sessions_dict` lookup (`get_scene_for_session`) is called once per session, which is not repeated. However, the reward zone coordinate lookup pattern is repeated: first to get reward zone coords/labels, then to get environment type - both parse the same scene name independently.

ii.
```python
# Both called with the same scene:
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
env_per_trial = get_environment_from_scene(scene, n_trials)
```

iii. This duplication is minor and has negligible performance impact since it's simple string parsing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads several data variables that are never used in the output:
- `scanning` data: loaded but never referenced
- `autoreward` data: loaded but never used
- `env_data` (NWB environment field): loaded but environment is derived from sessions_dict instead
- `trial_num` (NWB trial number field): loaded but trial number is computed from loop index

Additionally, the Speed output variable is computed and included, but the instructions list it as a decoder output. The AI includes all 6 specified outputs.

ii.
```python
# Loaded but never used:
scanning = behav['scanning']['data'][()]
autoreward = behav['autoreward']['data'][()]
env_data = behav['environment']['data'][()]
trial_num = behav['trial number']['data'][()]
```

iii. The AI loaded these variables during exploration but did not remove the unnecessary loads from the final code. This wastes some memory but does not affect correctness.
