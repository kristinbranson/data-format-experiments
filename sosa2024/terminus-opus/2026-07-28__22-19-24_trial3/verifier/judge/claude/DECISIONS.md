# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by iterating over `sub-*` directories in the `data` folder and listing `.nwb` files. It uses `h5py` to open each NWB file directly rather than `pynwb`. Subject IDs and session numbers are parsed from directory and file names. All behavioral and neural data are read from the HDF5 structure.

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
        all_files.append({...})
```
```python
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']
```

iii. The AI's CONVERSION_NOTES states 11 subjects and 152 sessions were found, matching the reference paper's description of n=11 switch-condition mice. Using h5py instead of pynwb is a valid choice for reading NWB files.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories named `sub-mX` in the data directory. Subject IDs are extracted by stripping the `sub-m` prefix.

ii.
```python
subject_id = sub_dir.replace('sub-m', '')
...
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)
```

iii. The number of subjects (11) matches the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from the filename pattern `sub-mX_ses-YY_behavior+ophys.nwb`.

ii.
```python
ses_num = int(fname.split('_ses-')[1].split('_')[0])
```

iii. 152 sessions were found across all subjects, matching the expected total.

## 1-d. How are the data split into trials?

i. Trials are identified by finding indices where `trial_start` > 0, then matching each trial start to the next teleport signal (`teleport > 0`) that occurs after it. Trial data spans from `trial_start` to (but not including) the matched teleport index.

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

iii. This approach produces trials from trial_start to the first teleport, consistent with the paper's definition. The AI found 12,216 total trials, close to the paper's 12,376 (difference attributed to 3 fixed-condition mice not in the NWB data).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 3 timepoints are skipped. Sessions with fewer than 5 cells or fewer than 2 valid trials are also skipped.

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

iii. The minimum trial length threshold (3) is very permissive compared to the reference (50). No other trial quality filtering is applied beyond this minimum length check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `Deconvolved` field in the NWB file's `ophys` processing module. This is suite2p's own deconvolution of raw fluorescence, NOT the paper's custom dF/F and OASIS deconvolution pipeline.

ii.
```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
```

iii. The AI's CONVERSION_NOTES (Step 5, Key Decision 1) states: "Use deconvolved events from NWB (matches reference code sess.timeseries['events'])". However, this is incorrect -- the NWB `Deconvolved` field is suite2p's deconvolution, while the paper's `sess.timeseries['events']` comes from the paper's own `preprocessing.dff()` function applied to raw `Fluorescence` and `Neuropil` traces.

## 2-b. How is the `neural` data processed?

i. No processing is applied. The raw suite2p `Deconvolved` values from the NWB file are used directly. There is no neuropil subtraction, no dF/F computation, no maximin baseline estimation, no smoothing, and no OASIS deconvolution -- all of which are steps in the paper's pipeline.

ii.
```python
neural_all = deconv[:, cell_mask]  # (n_timepoints, n_cells)
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The AI treats the stored Deconvolved data as equivalent to the paper's processed events, but the paper's Methods describe a specific pipeline: neuropil subtraction (coef=0.7), maximin baseline (20s window), dF/F, Gaussian smoothing (2-sample), and OASIS deconvolution (tau=0.7). The reference solution copies this entire pipeline and applies it to the raw Fluorescence and Neuropil traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `iscell` filter is applied (suite2p manual curation, keeping cells where `iscell[:, 0] == 1`). Putative interneurons are NOT filtered out.

ii.
```python
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
```

iii. The AI applies the iscell filter but does not implement the paper's interneuron exclusion step (filtering cells whose dF/F correlates with running speed at r > 0.5). The CONVERSION_NOTES mention "Neuron curation rules: iscell=1 (Suite2P manual curation)" but do not mention interneuron filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced from trial_start to teleport for each trial. Since alignment is to trial start, no additional temporal shifting is needed.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The alignment is straightforward since the alignment event is trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is computed as `1000.0 / imaging_rate` where `imaging_rate` is read from the first session's `ImagingPlane` metadata. No rebinning is applied. The reported time bin is ~64.48 ms.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
dt_ms = 1000.0 / imaging_rate
```

iii. The AI uses the `ImagingPlane` rate rather than the per-series `rate` used by the reference. For multi-plane animals (m17, m18), the series rate is the scanner rate (~31 Hz), and the per-plane rate is half that (~15.5 Hz). The AI's approach of reading from `ImagingPlane` may give a different value than the series rate. The reference accounts for multi-plane by computing `nplanes/frame_rate*1000`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is computed from the frame index and the imaging dt, NOT from stored timestamps.

ii.
```python
dt = 1.0 / imaging_rate
...
time_from_start = np.arange(trial_len) * dt
```

iii. The AI uses frame-count-based time rather than the actual behavioral timestamps stored in the NWB file.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The time is computed as `np.arange(trial_len) * dt`, where `dt = 1.0 / imaging_rate`. No subtraction of trial start time is needed since the range starts at 0.

ii.
```python
time_from_start = np.arange(trial_len) * dt
trial_input[0, :] = time_from_start
```

iii. This produces evenly spaced time values starting from 0. The reference uses actual timestamps and subtracts the first timestamp, which should produce similar but not identical values (actual timestamps may have slight jitter).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and time data use the same frame indices (`start:stop`), so they are inherently aligned.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
time_from_start = np.arange(trial_len) * dt  # trial_len = stop - start
```

iii. Same indexing ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the hardcoded `_all_sessions` dictionary (copied from reference code's `sessions_dict`) using the scene name, NOT from the `environment` behavioral variable in the NWB file.

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

iii. The AI imports the sessions dictionary from the reference code and uses scene names to determine environment type. The CONVERSION_NOTES explain the mapping: Env1=0, Env2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to determine if it's Env1 or Env2. For cross-environment switch sessions, the first 30 trials get the "before" environment and trials after 30 get the "after" environment.

ii.
```python
env_per_trial = get_environment_from_scene(scene, n_trials)
...
trial_input[1, :] = float(env_type)
```

iii. The AI uses a hardcoded switch trial (30) and scene parsing rather than reading the environment directly from the NWB data. The reference reads the `environment` behavioral time series directly.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop counter (0-indexed) over valid trials within a session.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = i
    trial_input[2, :] = float(trial_number)
```

iii. This is a sequential index of the trial within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing -- just the loop index. The value is constant across all timepoints within a trial.

ii.
```python
trial_input[2, :] = float(trial_number)
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward event timestamps are compared against trial boundaries (in timestamp space) to determine if a reward occurred.

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

iii. The AI checks both whether reward timestamps fall within the trial AND whether the reward zone was active, adding an extra condition compared to the reference.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is used. The first trial defaults to 0.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
trial_input[3, :] = float(prev_out)
```

iii. Standard shift-by-one approach. The reference checks the previous trial's time range for reward events; the AI pre-computes per-trial reward status and shifts.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone coordinates. Reward zone coordinates are determined from the hardcoded `_all_sessions` dictionary (scene name) and `REWARD_ZONE_DICT`, NOT from the `reward_zone` behavioral variable in the NWB file.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI uses the sessions dictionary to determine reward zones, which provides a clean mapping from session/day to zone. The reference uses a Viterbi algorithm on the actual `reward_zone` behavioral data to infer zone assignments.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative when before the zone, 0 when inside, positive when past the zone. The distance is to the nearest edge of the zone.

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

iii. This computation matches the paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using conditional array indexing with explicit boundary comparisons.

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

iii. The bin edges match the instructions. Minor boundary differences from the reference at exactly +10 and +50 cm (AI uses `<=` for upper bounds in bins 4 and 5, reference uses `<` via np.digitize). The AI uses `distances == 0` for bin 3, while the reference uses a small epsilon (1e-6) with np.digitize.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (`start:stop`) used for both neural and position data, ensuring alignment.

ii.
```python
trial_pos = position[start:stop]
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Same indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. Direct extraction of position data from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, TRACK_LENGTH]` (i.e., `[0, 450]`) before discretization.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

iii. The clipping ensures positions stay within the track boundaries. The reference does not clip, instead using open-ended bins (`-inf` and `inf`) to absorb edge values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins using `np.linspace(0, 450, 6)` as bin edges, then `np.digitize` and clipping.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The bin edges are `[0, 90, 180, 270, 360, 450]`, effectively creating 90 cm bins. The reference uses `[-inf, 90, 180, 270, 360, inf]`. The practical result is very similar since positions are already clipped to [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices used for neural and position data.

ii.
```python
trial_pos = position[start:stop]
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. Direct extraction of lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick data is processed in two steps: (1) lick sensor error detection -- if >35% of samples in a trial have lick values > 2, all licks in that trial are set to 0; (2) remaining lick values > 1 are clipped to 1, producing binary output.

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The lick sensor error detection comes from the reference code's `correct_lick_sensor_error` function. The reference solution simply binarizes with `(licks > 0).astype(int)` without lick sensor error correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices used.

ii.
```python
trial_lick = lick_raw[start:stop].copy()
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the hardcoded `_all_sessions` dictionary using the scene name for each session. The scene name encodes the reward zone location (e.g., `Env1_LocationA` -> zone A).

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
...
rz_label_idx[i] = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The AI uses the sessions dictionary from the reference code, which provides a deterministic mapping from session to reward zone. For switch sessions, the zone changes after trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine the reward zone. For switch sessions (containing `_to_`), the first 30 trials get the "before" location and subsequent trials get the "after" location. Encoded as 0=A, 1=B, 2=C.

ii.
```python
def get_reward_zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    if '_to_' in scene or '_to_Env' in scene:
        before_loc = parts[0][-1]
        after_loc = parts[1][-1]
        rz_labels[:change_trial] = ZONE_TO_LABEL[LOCATION_TO_ZONE[before_loc]]
        rz_labels[change_trial:] = ZONE_TO_LABEL[LOCATION_TO_ZONE[after_loc]]
    else:
        loc = scene[-1]
        rz_labels[:] = ZONE_TO_LABEL[LOCATION_TO_ZONE[loc]]
    return rz_coords, rz_labels
```

iii. The reference uses a data-driven Viterbi algorithm on the `reward_zone` behavioral data. The AI's approach assumes the switch always happens at trial 30 and that the sessions dictionary is correct.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The reward timestamps are compared against trial boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks if any reward timestamp falls within the trial's time range AND if the reward zone was active. If both conditions are met, the trial is marked as rewarded (1), otherwise not (0).

ii.
```python
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
...
trial_output[5, :] = rew_out
```

iii. The extra `rzone_active` check is not present in the reference. The reference simply checks if any reward event occurred in the trial's time range using `np.any(isreward[idx])`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Short trials**: Trials with < 3 timepoints are skipped.
- **Few cells**: Sessions with < 5 cells are skipped.
- **Few trials**: Sessions with < 2 valid trials are skipped.
- **Lick sensor errors**: Trials with >35% lick samples > 2 have lick set to 0.
- **Processing errors**: Try/except wraps session processing, with errors logged and session skipped.

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

iii. The error handling is mostly defensive. The reference handles neural/behavior length mismatches (cropping to minimum) and uses a higher minimum trial length (50 timepoints).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py and reading large arrays
2. Processing each session sequentially (all 152 sessions)
3. Saving the final pickle file (~9.8 GB)

ii. N/A

iii. The AI's code is simpler (no dF/F pipeline) and thus faster than the reference.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` (lines 536-623) iterates sequentially over trials. Operations like distance computation, discretization, and input construction could be applied to full-session arrays before splitting into trials.

ii. N/A

iii. Variable trial lengths make full vectorization awkward but not impossible.

## 13-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Unlike the reference (which has a separate survey step), the AI processes each NWB file only once.

ii. N/A

iii. The AI's approach is more efficient in this regard.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads some behavioral variables that are checked but not used in the final output:
- `autoreward` is loaded but not used
- `scanning` is loaded but not used
- `rzone_data` (reward_zone from NWB) is loaded only for the `rzone_active` check in reward detection

ii.
```python
autoreward = behav['autoreward']['data'][()]
scanning = behav['scanning']['data'][()]
```

iii. These are minor inefficiencies with negligible performance impact.
