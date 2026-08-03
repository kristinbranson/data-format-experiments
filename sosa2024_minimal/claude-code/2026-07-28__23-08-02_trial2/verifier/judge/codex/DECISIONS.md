# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all NWB files by iterating over every `sub-*` directory in the data root, then every `.nwb` file in each subject directory. Each file is opened directly with `h5py`, relevant behavioral and ophys arrays are read eagerly into memory, and each file is treated as one session.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
with h5py.File(filepath, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
    ophys = f['processing/ophys']
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
```

iii. The trajectory says the agent concluded that the NWB files already contain neural, behavioral, ROI, and reward-event streams needed for conversion, and it chose direct NWB traversal rather than `pynwb`. The notes justify this as using the NWB archive for the 11 switch-task mice.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name. The stored subject names in the output are the `sub-mN` directory names with the `sub-` prefix removed, and `subject_idx` is assigned in that directory iteration order.

ii.
```python
for subj_dir_name in subjects:
    subj_path = os.path.join(data_dir, subj_dir_name)
    subj_id = subj_dir_name  # e.g., 'sub-m11'
    ...
    mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'

    if mouse_name not in subject_names:
        subject_names.append(mouse_name)
    subj_idx = subject_names.index(mouse_name)
```

iii. The notes describe a manual mapping from NWB IDs like `sub-m3` to paper/code IDs like `GCAMP3`, but the saved subject identities are the mouse IDs derived from directory names.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is one session. The session/day index is parsed from the filename component like `ses-03`, and the scene metadata are then looked up from a hard-coded `SESSIONS_INFO` table keyed by mapped GCAMP name and day number.

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
ses_part = sess_file.split('_')[1]  # 'ses-03'
exp_day = int(ses_part.split('-')[1])
...
scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. The trajectory explicitly lists “subject mapping: `sub-mN -> GCAMPN`, session number = exp_day” as a design decision so session/day metadata could be matched to the paper’s reward-zone schedule.

## 1-d. How are the data split into trials?

i. Trials are split using sample indices where `trial_start > 0` as starts and sample indices where `teleport > 0` as ends. The code pairs these arrays positionally, truncates to the shorter count, and slices each trial as `start:end`.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
teleport_idx = np.where(nwb_data['teleport'] > 0)[0]

n_trials = min(len(tstart_idx), len(teleport_idx))
tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The notes say the agent intended trials to span the “on-track portion only, excluding teleport zone” from `trial_start` to `teleport`. The trajectory also names `trial_start` and `teleport` as the chosen segmentation markers.

## 1-e. How are trials filtered based on quality controls?

i. The code does very limited trial filtering: it skips sessions with fewer than 2 detected trials, skips trials where `end <= start` or trial length is `< 2` samples, and skips sessions with fewer than 2 remaining valid trials. It does not implement the reference solution’s `< 50` timepoint filter.

ii.
```python
if n_trials < 2:
    print(f"  Skipping session: only {n_trials} trials")
    return None
...
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

iii. The notes emphasize lick-error correction and basic session validity rather than aggressive trial curation. No explicit justification was given for omitting the reference short-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved `neural` signal comes from `processing/ophys/Deconvolved/.../data`, with `processing/ophys/Fluorescence/.../data` used only to identify putative interneurons for exclusion. ROI selection also depends on `iscell`.

ii.
```python
iscell = seg['iscell'][:]
...
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
...
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. The notes state that the agent used deconvolved calcium activity because the paper’s analyses used OASIS-deconvolved activity, while fluorescence was used as a proxy for the dF/F-based interneuron filter described in the methods.

## 2-b. How is the `neural` data processed?

i. Neural data are concatenated across planes, cropped to the behavior length if needed, filtered first by `iscell[:, 0] == 1`, then filtered again to remove putative interneurons identified by fluorescence-speed correlation. The remaining deconvolved traces are transposed per trial to `(neurons, time)`.

ii.
```python
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
...
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    ...
    deconvolved = deconvolved[:min_len]
    fluorescence = fluorescence[:min_len]
...
cell_mask = nwb_data['iscell'][:, 0] == 1
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]
...
is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
...
neural = deconvolved[:, neuron_mask]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The notes justify plane pooling from the paper’s multi-plane description, `iscell` filtering from Suite2P curation, and additional speed-correlation filtering from the paper’s interneuron-exclusion rule.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered by `iscell[:, 0] == 1`, then additional cells are excluded if their fluorescence trace has Pearson correlation `> 0.5` with running speed over timepoints where `position > 0`.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
...
valid_mask = nwb_data['position'] > 0  # exclude teleport period (pos = -500)
...
r = np.corrcoef(neural_ts, speed_ts)[0, 1]
if r > threshold:
    is_interneuron[c] = True
```

iii. The notes explicitly justify these as the manual Suite2P curation plus the paper’s extra interneuron exclusion criterion based on speed correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural matrices are aligned to trial start by slicing from each `trial_start` index to the paired `teleport` index. No extra shifting or interpolation is applied.

ii.
```python
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The notes say the temporal alignment event is the `trial_start` event itself, so the agent treated the left edge of each extracted slice as time zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code uses the native imaging-frame resolution with no rebinning or interpolation. It assumes a session frame time of `1 / imaging_rate`, and the saved metadata hard-code `1000.0 / 15.5078125` ms.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / nwb_data['imaging_rate']
...
'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
```

iii. The notes justify this as matching the approximately 15.5 Hz imaging rate reported by the paper, with no additional temporal rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from the behavioral timestamps array. Instead it is derived from the trial length and the imaging rate: `np.arange(n_timepoints)` scaled by `1 / imaging_rate`.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The notes justify this by treating the imaging frame rate as the session clock and the trial start as time zero.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial the code builds a uniformly spaced elapsed-time vector starting at 0 and increasing by one frame duration per sample, then copies that vector into the first input row.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
...
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. The notes state that `off_start = 0.0` because alignment is to the trial-start event itself; the trajectory likewise says the data are aligned to trial start at ~15.5 Hz.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction to the same trial slices as the neural data, with one time value per neural frame in that trial. Alignment is sample-count based rather than timestamp based.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
...
n_timepoints = end - start
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The notes say the session is aligned to trial start and uses the imaging frame rate as the common temporal grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved environment type is derived from hard-coded session scene metadata in `SESSIONS_INFO`, parsed by `parse_scene_environment`. The raw NWB `environment` time series is loaded but not used for the saved decoder input.

ii.
```python
environment = bts['environment/data'][:]
...
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. The notes justify this as determining environment type from scene names, especially for the day-8 environment switch session.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code parses each session scene name into `Env1` or `Env2`, assigns that code to all trials in the session, and for scene strings containing an environment switch assigns the post-switch value from trial 30 onward.

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
...
input_arr[1, :] = env_per_trial[i]
```

iii. The notes explicitly describe this rule: `Env1` only gives 0, `Env2` only gives 1, and day-8 switch sessions use the first 30 trials in the original environment and the rest in the new environment.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The saved trial number is not derived from the NWB `trial number` series. It is derived from the per-session trial-loop index `i` after the code forms paired trial boundaries.

ii.
```python
trial_num = bts['trial number/data'][:]
...
for i in range(n_trials):
    ...
    input_arr[2, :] = float(i)
```

iii. The notes and trajectory both describe trial number as a 0-indexed trial index within session, rather than a raw NWB field.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the current trial index and repeating it across all timepoints in the trial.

ii.
```python
input_arr[2, :] = float(i)
```

iii. The notes justify this as a simple within-session trial counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward event timestamps `reward_ts` and the behavior `timestamps` array. The code first determines a per-trial current reward outcome, then shifts that vector by one trial.

ii.
```python
timestamps = nwb_data['timestamps']
reward_ts = nwb_data['reward_ts']
...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. The notes justify this by defining a trial as rewarded when any reward timestamp falls inside its boundaries, with the first trial’s previous outcome set to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code creates `reward_per_trial` by checking whether any reward event falls between each trial’s start and end timestamps, then fills a `prev_outcome` vector whose first element is 0 and whose later elements are the previous trial’s reward result. That value is repeated across each trial’s timepoints.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
...
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
...
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_arr[3, :] = float(prev_outcome[i])
```

iii. The notes say the intended binary meaning is omitted `0` versus rewarded `1`, with the first trial defaulting to `0`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the raw `position` series plus per-trial reward-zone labels inferred from session scene metadata, not from the raw `reward_zone` time series. The code turns the metadata label into zone start/end coordinates and computes signed distance from position to that zone.

ii.
```python
position = bts['position/data'][:]
reward_zone = bts['reward_zone/data'][:]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The notes justify this by saying reward-zone location is determined from `sessions_dict.py` scene metadata and that switch days use trial 30 as the zone-change point.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint the code computes signed distance to the nearest boundary of the active reward zone: negative before the zone, zero inside it, positive after it. The continuous distance is then discretized with a dedicated rule function.

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
...
dist_disc = discretize_distance_to_reward(dist)
```

iii. The notes describe exactly this interpretation: zone A/B/C coordinates from the paper, distance 0 inside the zone, and signed distance before versus after the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The agent uses a hand-written thresholding function with 7 bins matching the requested categories.

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

iii. The notes say these bins were chosen to match the decoder specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance output is aligned to neural data by using the same `start:end` trial slices from the session arrays and producing one discretized distance value per neural frame.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
...
pos = nwb_data['position'][start:end]
...
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
output_arr[0, :] = dist_disc
```

iii. The notes justify this alignment by stating that all decoder variables are aligned to trial start on the imaging-frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the raw `position` behavioral time series.

ii.
```python
position = bts['position/data'][:]
...
pos = nwb_data['position'][start:end]
```

iii. The notes identify this as the animal’s corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code clips raw positions to `[0, 450]` cm and then discretizes them into five equal 90 cm bins across the 450 cm track using floor division.

ii.
```python
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
...
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The notes justify this as matching the requested “5 equal-sized bins” over the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five bins: `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
...
'output_values': [
    ...
    ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
```

iii. The notes explicitly list these five equal-width track bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same sample slice as the neural data for each trial, then one categorical position value is produced per neural frame.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
output_arr[1, :] = pos_disc
```

iii. The notes say all trial-wise variables are aligned to trial start at the imaging frame rate.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the raw `lick` behavioral time series.

ii.
```python
lick = bts['lick/data'][:]
...
lck = licks_corrected[start:end]
```

iii. The notes identify the lick stream as the source, with extra correction for lick-sensor artifacts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code first marks entire trials as lick-error trials when more than 30% of that trial’s samples have raw lick values `> 2`, sets those trial samples to `NaN`, converts `NaN` back to `0`, and finally binarizes remaining values as `> 0`.

ii.
```python
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    licks = np.copy(lick_data)
    ...
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > threshold:
        licks[start:end] = np.nan
...
lck = licks_corrected[start:end]
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. The notes justify this as an attempt to follow the paper’s lick-sensor error correction, although the notes describe cumulative lick counts while the implemented code thresholds raw lick values directly.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick values are aligned by slicing the corrected lick vector on the same `start:end` boundaries used for neural data, then emitting one binary lick label per neural frame.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
...
lck = licks_corrected[start:end]
...
output_arr[3, :] = lck_binary
```

iii. The notes say all time-varying outputs are aligned to trial start on the imaging frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from hard-coded session scene metadata in `SESSIONS_INFO` and parsed scene strings, not from the raw NWB `reward_zone` time series.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The notes justify this as using the paper/code session schedule for zone A/B/C and applying the 30-trial switch rule on switch sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses the session scene name into a per-trial sequence of reward-zone labels, using one label for stable sessions and a before/after split at trial 30 for switch sessions, then maps `A/B/C` to `0/1/2`.

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    rz_labels = np.empty(n_trials, dtype='U1')
    ...
    else:
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The notes and trajectory both say the reward-zone schedule came from `sessions_dict.py` scene metadata plus the paper’s “switch after 30 trials” rule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward event timestamps `reward_ts` together with the behavioral `timestamps` array and the trial boundary indices.

ii.
```python
reward_ts = bts['Reward/timestamps'][:]
...
start_t = timestamps[tstart_idx[i]]
end_t = timestamps[teleport_idx[i]]
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
```

iii. The notes justify this as classifying a trial as rewarded if any reward event occurs within its boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial the code checks whether any reward timestamp falls between that trial’s start and end timestamps, stores a binary per-trial result in `reward_per_trial`, and repeats that scalar across all timepoints of the trial in the final output.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
output_arr[5, :] = reward_per_trial[i]
```

iii. The notes describe reward outcome as a binary per-trial label and use the same reward-boundary rule for previous trial outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles a few issues defensively: it crops neural and behavioral arrays to the minimum common length when they differ, skips sessions with too few trials or no valid neurons, skips degenerate trials with non-positive length or fewer than 2 samples, marks lick-artifact trials by setting their lick samples to `NaN` and then converts those `NaN`s to zeros, and skips sessions missing subject/session metadata from the hard-coded tables.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    ...
    deconvolved = deconvolved[:min_len]
...
if n_trials < 2:
    ...
if n_neurons < 1:
    ...
if end <= start:
    continue
if n_timepoints < 2:
    continue
...
if frac_high > threshold:
    licks[start:end] = np.nan
...
lck = np.nan_to_num(lck, nan=0.0)
...
if subj_id not in SUBJECT_MAP:
    print(f"Warning: Unknown subject {subj_id}, skipping")
    continue
...
if exp_day not in SESSIONS_INFO[gcamp_name]:
    print(f"Warning: No session info for {gcamp_name} day {exp_day}, skipping")
    continue
```

iii. The notes explicitly justify the neural/behavior truncation and lick-artifact handling. The rest are implicit safeguards in the conversion code rather than decisions discussed in detail.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work is likely full-session NWB I/O, eager loading of all behavioral and imaging arrays, per-neuron correlation screening for interneuron exclusion, and the per-trial loop that constructs output arrays for every session.

ii.
```python
with h5py.File(filepath, 'r') as f:
    ...
    deconvolved = np.concatenate(deconv_list, axis=1)
    fluorescence = np.concatenate(fluor_list, axis=1)
...
for c in range(n_neurons):
    ...
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
...
for i in range(n_trials):
    ...
    output_arr = np.zeros((6, n_timepoints), dtype=np.int64)
```

iii. There is no explicit justification in the notes beyond using direct full-session conversion and validation. The trajectory shows the agent prioritized a straightforward implementation over optimization.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-neuron loop in `identify_interneurons`, the per-trial loop computing `reward_per_trial`, the per-trial loop building neural/input/output trial arrays, and the lick-error loop over trials.

ii.
```python
for c in range(n_neurons):
    ...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
for i in range(n_trials):
    ...
    trial_neural.append(trial_n)
    trial_input.append(input_arr)
    trial_output.append(output_arr)
...
for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
    ...
```

iii. No explicit optimization discussion appears in the notes. This is an inference from the code structure.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial-boundary traversal several times per session: once for lick-error correction, once for reward-outcome assignment, and once for the main trial extraction loop. It also makes additional full-dataset passes during `run_sanity_checks` after conversion.

ii.
```python
licks_corrected, error_trials = correct_lick_sensor_errors(
    nwb_data['lick'], tstart_idx, teleport_idx,
    threshold=LICK_ERROR_THRESHOLD
)
...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
...
for i in range(n_trials):
    start = tstart_idx[i]
    end = teleport_idx[i]
    ...
for s in range(n_sessions):
    ...
    for t in range(n_trials):
        ...
```

iii. The notes do not discuss this explicitly. It follows from the agent’s straightforward, multi-pass implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several fields that are not used in the saved decoder dataset (`trial_num`, raw `environment`, raw `reward_zone_ts`, `autoreward`, `plane_idx`, `reward_data`, `subject_id`, `session_id`), constructs unused temporary arrays like `input_per_trial` and a placeholder `input_arr`, and stores detailed `session_info` plus prints sanity summaries that are not used by downstream decoding itself.

ii.
```python
trial_num = bts['trial number/data'][:]
environment = bts['environment/data'][:]
reward_zone = bts['reward_zone/data'][:]
autoreward = bts['autoreward/data'][:]
reward_data = bts['Reward/data'][:]
...
plane_idx = seg['planeIdx'][:]
...
subject_id = f['general/subject/subject_id'][()]
session_id = f['general/session_id'][()]
...
input_arr = np.array([
    time_from_start[0] if False else 0,  # placeholder
])
...
input_per_trial = np.array([
    float(env_per_trial[i]),
    float(i),  # trial number
    float(prev_outcome[i]),
], dtype=np.float32)
```

iii. No explicit justification is given for these extra loads and temporaries. They appear to come from exploratory or convenience-oriented implementation choices rather than downstream requirements.
