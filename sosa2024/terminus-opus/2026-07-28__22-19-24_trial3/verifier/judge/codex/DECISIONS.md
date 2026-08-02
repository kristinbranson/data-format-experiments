# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for `sub-*` directories, scans each subject directory for `.nwb` files, and opens each file directly with `h5py`. Within each file it loads behavioral arrays from `processing/behavior/BehavioralTimeSeries` and neural arrays from `processing/ophys`.

ii.
```python
for sub_dir in sorted(os.listdir(data_dir)):
    if not sub_dir.startswith('sub-'):
        continue
    ...
    for fname in sorted(os.listdir(sub_path)):
        if not fname.endswith('.nwb'):
            continue
```

```python
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']
```

iii. In `CONVERSION_NOTES.md` the AI says it mapped NWB contents directly to the reference `sess`-like structure and processed all 152 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined by directory names under `data/`, and the subject id is parsed from the directory name `sub-mX`.

ii.
```python
for sub_dir in sorted(os.listdir(data_dir)):
    if not sub_dir.startswith('sub-'):
        continue
    subject_id = sub_dir.replace('sub-m', '')
```

iii. The notes say the dataset is organized by subject directories and that this matches the 11 mice in the paper.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session, and the session number is parsed from the filename `..._ses-YY_...`.

ii.
```python
for fname in sorted(os.listdir(sub_path)):
    if not fname.endswith('.nwb'):
        continue
    ses_num = int(fname.split('_ses-')[1].split('_')[0])
```

iii. The notes say NWB session numbers correspond to `exp_day` in the reference `sessions_dict`.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` to the next `teleport`. The code finds all positive `trial_start` indices, all positive `teleport` indices, then for each start chooses the next later teleport.

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

iii. The notes say this matches the trial structure used in the reference code.

## 1-e. How are trials filtered based on quality controls?

i. The code skips sessions with fewer than 2 trials, skips sessions with fewer than 5 cells, and skips individual trials shorter than 3 frames. It does not implement the reference solution’s `<50` timepoint trial filter.

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

iii. The notes justify the short-trial skip as an edge-case cleanup and record “Short trials (<3 timepoints): Skipped.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from `processing/ophys/Deconvolved/*/data`, with the ROI quality mask from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
```

iii. The notes say this matches the reference use of deconvolved events and Suite2P cell curation.

## 2-b. How is the `neural` data processed?

i. The code concatenates all imaging planes along the ROI axis, filters to `iscell[:, 0] == 1`, then slices each trial and transposes to `(n_neurons, n_timepoints)`.

ii.
```python
deconv = np.concatenate(deconv_parts, axis=1)
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
...
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The notes say multi-plane sessions were handled by concatenating planes and then applying the combined cell mask.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The main neural QC is `iscell[:, 0] == 1`. The code also rejects whole sessions with fewer than 5 retained cells.

ii.
```python
cell_mask = iscell[:, 0] == 1
n_cells = cell_mask.sum()

if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None
```

iii. The notes cite Suite2P `iscell` filtering as the intended curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start implicitly by cutting each trial from its `trial_start` index to its matching `teleport` index.

ii.
```python
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. The notes state the target alignment event is trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The code uses `dt = 1.0 / imaging_rate` seconds per frame and stores `time_bin_size = 1000.0 / imaging_rate` in metadata. For multi-plane sessions this is 2x finer than the reference behavior/neural bin.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
dt_ms = 1000.0 / imaging_rate
...
'time_bin_size': dt_ms,
```

iii. The notes claim the native bin is about 64.5 ms with no extra binning, but the code computes it directly from `imaging_rate`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from the NWB timestamps. It is derived from the trial length and the scalar `dt = 1 / imaging_rate`.

ii.
```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
...
time_from_start = np.arange(trial_len) * dt
```

iii. The notes justify this as using the native imaging frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial the code creates `0, dt, 2*dt, ...` up to trial length minus one frame. There is no use of recorded timestamps and no subtraction of the observed trial-start timestamp.

ii.
```python
time_from_start = np.arange(trial_len) * dt
...
trial_input[0, :] = time_from_start
```

iii. The notes say the input is “frame_index * dt.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction to the neural frame count within each trial: the time vector has exactly `trial_len` bins and is stored alongside the trial’s neural slice.

ii.
```python
trial_len = stop - start
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
time_from_start = np.arange(trial_len) * dt
trial_input = np.zeros((4, trial_len), dtype=np.float32)
```

iii. The notes treat the neural frame rate as the master clock.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The code derives environment from the session scene in the copied `sessions_dict`, not from the NWB `environment` time series it loaded.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
...
env_per_trial = get_environment_from_scene(scene, n_trials)
```

iii. The notes say environment mapping came from the reference code’s scene metadata, with `Env1=0` and `Env2=1`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code converts the scene name into a per-trial binary environment vector, splitting at trial 30 for cross-environment switch scenes; then it broadcasts the per-trial value across all time bins in the trial.

ii.
```python
def get_environment_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    env = np.zeros(n_trials, dtype=int)
    if '_to_Env' in scene:
        ...
        env[:change_trial] = before_env
        env[change_trial:] = after_env
```

```python
env_type = env_per_trial[i]
trial_input[1, :] = float(env_type)
```

iii. The notes justify this as following the reference scene schedule.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the trial loop index after trial segmentation, not from the loaded `trial number` array.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = i
```

iii. The notes say trial number is a 0-indexed within-session counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is broadcast as a constant value across all bins of the trial.

ii.
```python
trial_input[2, :] = float(trial_number)
```

iii. No separate justification appears beyond the notes’ mapping table.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The code first derives a per-trial `isreward` from `Reward.timestamps`, the behavior `position` timestamps, and whether `reward_zone` was active during the trial. `previous_trial_outcome` is then the lagged version of that per-trial reward label.

ii.
```python
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
```

iii. The notes describe it as “0=omission, 1=rewarded.”

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is set to 0. Every later trial gets the binary reward outcome of the previous trial, broadcast across the full trial.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
prev_out = prev_outcome[i]
trial_input[3, :] = float(prev_out)
```

iii. The notes explicitly record the sanity check `prev_outcome[i] = reward_outcome[i-1]`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from raw `position` plus reward-zone start/stop coordinates inferred from scene metadata, not from the per-frame `reward_zone` labels in the NWB stream.

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

iii. The notes say reward-zone location came from scene mapping imported from the reference code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside, positive after.

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

iii. The notes describe this as signed distance relative to the hidden reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code uses manual boolean thresholds for 7 categories. Exact boundary handling differs from the reference `np.digitize` scheme at `10` and `50` cm.

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

iii. The notes say these bins were chosen to match the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial slice `position[start:stop]` used to define the neural trial, so it has the same number of bins as the neural matrix.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
trial_output[0, :] = dist_bins
```

iii. The notes say the whole dataset is aligned to trial start.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw `position` behavioral time series.

ii.
```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. No separate justification beyond the variable-mapping table.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code clips position to `[0, 450]` cm before binning.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

iii. The notes describe this as using the 450 cm track length.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code divides `0..450` cm into five equal-width bins using `np.linspace` and `np.digitize`, after clipping values into that range.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The notes justify this from the 450 cm corridor length and the instruction to use 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is taken from the same per-trial index range as the neural data and stored with one value per neural frame.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_pos = position[start:stop]
trial_output[1, :] = pos_bins
```

iii. No additional justification beyond the trial-based alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` behavioral time series.

ii.
```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. The notes call this the lick sensor output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code applies a trial-level lick sensor error rule: if more than 35% of samples are `>2`, it sets the whole trial to zero. Otherwise it clips values `>1` down to `1`, then casts to integer.

ii.
```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The notes cite the reference code’s lick sensor correction and say “Lick sensor errors: Detected and set to 0.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same `[start:stop]` frame range as the neural data, so it has one value per neural frame.

ii.
```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
trial_lick = lick_raw[start:stop].copy()
trial_output[3, :] = trial_lick
```

iii. The notes treat all outputs as trial-start aligned framewise series.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session scene metadata through the copied `sessions_dict`, not from the noisy per-frame `reward_zone` stream.

ii.
```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. The notes say scene-based reward-zone determination was taken from the reference code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses the scene string, assigns reward zone A/B/C for each trial, switches after trial 30 on switch days, converts labels to integers `A->0`, `B->1`, `C->2`, then broadcasts that trial label across all frames.

ii.
```python
if '_to_' in scene or '_to_Env' in scene:
    ...
    rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
    rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
```

```python
if rz_labels[i] == 'A':
    rz_label_idx[i] = 0
elif rz_labels[i] == 'B':
    rz_label_idx[i] = 1
elif rz_labels[i] == 'C':
    rz_label_idx[i] = 2
...
trial_output[4, :] = rz_loc
```

iii. The notes justify the fixed switch trial from the paper and sessions table.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward event timestamps and the trial boundaries, with an extra requirement that `reward_zone` be active somewhere in the trial.

ii.
```python
reward_ts = behav['Reward']['timestamps'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The notes frame this as rewarded vs omission trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial the code counts reward timestamps inside that trial’s time window and labels the trial rewarded if at least one reward occurred and the reward-zone indicator was active. The result is then broadcast across all trial bins.

ii.
```python
isreward = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    ...
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
...
rew_out = isreward[i]
trial_output[5, :] = rew_out
```

iii. The notes say the binary trial outcome matched expected omission rates.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles only a few cases explicitly: skip sessions with too few cells or trials, skip trials shorter than 3 frames, zero out lick-error trials, and skip sessions that raise exceptions. It does not implement the reference solution’s neural/behavior length cropping or missing reward-zone inference from data.

ii.
```python
if n_cells < 5:
    return None
if n_trials < 2:
    return None
...
if trial_len < 3:
    continue
...
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
```

iii. The notes list multi-plane handling, short-trial skipping, and lick sensor error handling as the main edge-case checks.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are opening every NWB file, reading large behavior and deconvolved arrays, concatenating multi-plane data, iterating over trials, and writing a very large pickle.

ii.
```python
for i, file_info in enumerate(all_files):
    ...
    result = process_session(...)
...
deconv = np.concatenate(deconv_parts, axis=1)
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes estimate full conversion at about 90 seconds and highlight the full-dataset pass as the main runtime cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the loop that matches each trial start to the next teleport, the per-trial reward/outcome loop, the reward-zone label conversion loop, and the main per-trial slicing/broadcast loop.

ii.
```python
for i in range(len(trial_start_inds)):
    future_teleports = teleport_inds[teleport_inds > start]
```

```python
for i in range(n_trials):
    ...
    isreward[i] = ...
```

```python
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
```

iii. The notes mention general speedups, but do not give a more specific vectorization analysis than the final implementation.

## 13-c. What processing does the code repeat multiple times?

i. Within a run, the code repeatedly rescans the teleport array for each trial start, repeatedly slices session arrays trial by trial, and repeatedly broadcasts per-trial constants to full trial-length matrices. Unlike the human reference solution, it does not do a separate whole-dataset survey pass before conversion.

ii.
```python
for i in range(len(trial_start_inds)):
    future_teleports = teleport_inds[teleport_inds > start]
```

```python
for i in range(n_trials):
    ...
    trial_input[1, :] = float(env_type)
    trial_input[2, :] = float(trial_number)
    trial_input[3, :] = float(prev_out)
```

iii. The notes emphasize a single conversion pass over all 152 sessions rather than the reference solution’s extra survey phase.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several arrays it never uses downstream (`trial_num`, `env_data`, `scanning`, `reward_data`, `autoreward`), computes `n_timepoints` and an initial `n_trials = min(...)` that are later overwritten, and creates a 1D `trial_input` array that is immediately replaced by a 2D array.

ii.
```python
trial_num = behav['trial number']['data'][()]
env_data = behav['environment']['data'][()]
scanning = behav['scanning']['data'][()]
reward_data = behav['Reward']['data'][()]
autoreward = behav['autoreward']['data'][()]
```

```python
n_timepoints = len(position)
...
n_trials = min(len(trial_start_inds), len(teleport_inds))
...
trial_input = np.array([time_from_start[0], float(env_type), float(trial_number), float(prev_out)], dtype=np.float32)
trial_input = np.zeros((4, trial_len), dtype=np.float32)
```

iii. There is no explicit justification for these extra computations in the notes.
