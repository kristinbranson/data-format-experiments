# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI lists every `sub-*` directory under the data root, then lists every `.nwb` file inside each subject directory, and loads each session file with `h5py` by reading named NWB/HDF5 datasets directly. Trials are then created later inside `process_session()`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
with h5py.File(filepath, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
    ophys = f['processing/ophys']
```

iii. In the trajectory the agent explicitly chose directory scanning plus direct NWB/HDF5 reads, and summarized the plan as loading all `sub-mN` mice, all session files, then processing trials from those session arrays. `CONVERSION_NOTES.md` also states the source is the NWB dataset from DANDI.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the `sub-*` directories. The AI then strips the `sub-` prefix and stores subject names like `m11`, with `subject_idx` pointing sessions back to those names.

ii.
```python
for subj_dir_name in subjects:
    subj_path = os.path.join(data_dir, subj_dir_name)
    subj_id = subj_dir_name
    ...
    mouse_name = subj_id.replace('sub-', '')
    if mouse_name not in subject_names:
        subject_names.append(mouse_name)
```

iii. The trajectory says the intended subject mapping was `sub-mN -> GCAMPN`, and `CONVERSION_NOTES.md` includes a subject-mapping table built around those NWB subject directories.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session/day number is parsed from the filename `ses-XX` token and then used to look up scene metadata in the hard-coded `SESSIONS_INFO` table.

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
...
ses_part = sess_file.split('_')[1]
exp_day = int(ses_part.split('-')[1])
scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. The agent justified this in the trajectory as “session number = exp_day,” and `CONVERSION_NOTES.md` uses the same subject/day table.

## 1-d. How are the data split into trials?

i. Trial starts are all indices where `trial_start > 0`; trial ends are all indices where `teleport > 0`. The code truncates both lists to the same minimum length, then defines each trial as the slice `start:end`.

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
```

iii. `CONVERSION_NOTES.md` says trials are defined by `trial_start` and `teleport`, and the trajectory repeatedly notes that the agent chose “trial_start to teleport” segmentation after inspecting the NWB structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference trial-length filter. It only skips degenerate trials where `end <= start` or where the resulting trial has fewer than 2 timepoints, and skips entire sessions if fewer than 2 valid trials remain.

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

iii. I did not find an explicit justification for dropping only degenerate trials. The notes mention the decoder requirement that sessions need at least two trials, but do not describe a stricter per-trial QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices come from `processing/ophys/Deconvolved`, but the selection of which neurons survive also depends on `ImageSegmentation/PlaneSegmentation/iscell` and on `processing/ophys/Fluorescence` for interneuron exclusion.

ii.
```python
iscell = seg['iscell'][:]
...
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
for fk in sorted(fluor_keys):
    fluor_list.append(ophys['Fluorescence'][fk]['data'][:])
...
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]
```

iii. The trajectory states the decision as “use deconvolved activity, filter by iscell, exclude interneurons (speed corr > 0.5),” and `CONVERSION_NOTES.md` repeats that same rationale.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, filters ROIs by `iscell`, computes a fluorescence-speed correlation to mark putative interneurons, removes those neurons, and finally slices the remaining deconvolved traces per trial and transposes to neuron-by-time.

ii.
```python
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
...
cell_mask = nwb_data['iscell'][:, 0] == 1
...
is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
neural = deconvolved[:, neuron_mask]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The notes justify this as matching the paper’s pooled multi-plane analysis plus a putative-interneuron exclusion rule based on fluorescence/speed correlation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: `iscell[:, 0] == 1` from Suite2P curation, then exclusion of putative interneurons whose fluorescence is too correlated with running speed.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
...
is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD
)
neuron_mask = ~is_interneuron
```

iii. `CONVERSION_NOTES.md` explicitly lists both QC rules and quotes the paper for the second one. The trajectory summary at script-writing time makes the same claim.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start. The AI aligns neural data by segmenting the continuous session trace at `trial_start` and then taking `start:end` slices, so each trial begins at the trial-start boundary.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
...
start = tstart_idx[i]
end = teleport_idx[i]
...
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says “Aligned to start of trial (trial_start event),” and the trajectory lists “Temporal alignment: Align to trial start” as one of the core design decisions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging sampling and does not rebin. It uses `1 / imaging_rate` for per-frame timing and stores metadata time-bin size as `1000 / 15.5078125`, about 64.5 ms.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. The notes justify this as the paper’s native imaging rate, stating “Time bin size: ~64.48 ms ... no rebinning.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI does not derive this input from NWB timestamps. It derives it from trial length and the imaging rate, using frame indices multiplied by `1 / imaging_rate`.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The notes justify fixed-rate timing by citing the imaging rate and a constant ~64.48 ms bin size, rather than behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code creates a fresh `0, 1, 2, ...` frame index and multiplies by the frame duration. It does not subtract the first recorded timestamp from a behavioral clock.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
...
input_arr[0, :] = time_from_start
```

iii. The AI’s stated justification is implicit: the notes treat the data as uniformly sampled at the imaging frame rate, so a regular frame grid is used.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time-from-start vector is aligned by construction: it is created with exactly the same `n_timepoints` as the per-trial neural slice and stored as one row of the same `(4, n_timepoints)` input matrix.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
...
n_timepoints = end - start
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
...
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. `CONVERSION_NOTES.md` says the dataset is aligned to trial start at fixed imaging bins; I did not find a more detailed alignment justification beyond that.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from hard-coded session scene metadata, not from the NWB `environment` time series.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
```

iii. The trajectory says reward-zone and environment metadata come from `sessions_dict.py`, and `CONVERSION_NOTES.md` explicitly says environment type is “Determined from scene name.”

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code parses `Env1` versus `Env2` out of the scene string. For day-8 environment-switch sessions it assigns the first 30 trials to the first environment and the remaining trials to the second.

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
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. `CONVERSION_NOTES.md` gives exactly this explanation, including the “first 30 trials / remaining trials” rule for switch sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not read from a stored NWB variable. It is derived from the loop index `i` after the code has already split the session into trials.

ii.
```python
for i in range(n_trials):
    ...
    input_arr[2, :] = float(i)
```

iii. The notes justify this as a “0-indexed trial within session,” and the trajectory summary also describes trial number as the within-session index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond filling the same scalar trial index across every timepoint in the trial.

ii.
```python
input_arr[2, :] = float(i)
```

iii. `CONVERSION_NOTES.md` states that trial number is a constant, per-trial decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward/timestamps` event stream, combined with each trial’s start and end timestamps.

ii.
```python
timestamps = nwb_data['timestamps']
reward_ts = nwb_data['reward_ts']
...
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
```

iii. The notes justify reward determination by saying a trial is rewarded if any reward timestamp falls inside its boundaries, and previous-trial outcome is derived from that per-trial reward label.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes a binary `reward_per_trial` vector, then shifts it by one trial. The first trial is forced to 0.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
...
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
...
input_arr[3, :] = float(prev_outcome[i])
```

iii. `CONVERSION_NOTES.md` states that the first trial has no previous outcome and is therefore set to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output uses the raw `position` time series plus a reward-zone label inferred from the hard-coded session scene metadata. The NWB `reward_zone` time series is loaded but not used to define the converted reward zone.

ii.
```python
position = bts['position/data'][:]
reward_zone = bts['reward_zone/data'][:]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
pos = nwb_data['position'][start:end]
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
```

iii. The trajectory shows that the agent inspected the NWB `reward_zone` field, decided it was messy, and then switched to using `sessions_dict.py` scene metadata plus a 30-trial switch rule. The notes repeat that choice.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest edge of the current trial’s reward zone: negative before the zone, zero inside it, positive after it.

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
```

iii. `CONVERSION_NOTES.md` presents this as the intended signed distance variable for decoding.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI manually thresholds the continuous distance into 7 classes using the requested breakpoints at `-50`, `-10`, `0`, `10`, and `50` cm.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. The notes describe the same seven bins and say they follow the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The code computes distance from the per-trial `position[start:end]` slice and writes it into an output array with the same `n_timepoints` as the neural trial matrix, so alignment is by shared trial slicing.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
output_arr = np.zeros((6, n_timepoints), dtype=np.int64)
output_arr[0, :] = dist_disc
```

iii. I did not find a separate justification beyond the general “aligned to trial start” claim in the notes.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is taken from the behavioral `position` time series.

ii.
```python
position = bts['position/data'][:]
...
pos = nwb_data['position'][start:end]
```

iii. No special justification was given beyond using the VR position signal directly.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Before discretization, the code clips position values into `[0, 450]` cm and then bins the clipped positions.

ii.
```python
pos = nwb_data['position'][start:end]
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped)
```

iii. `CONVERSION_NOTES.md` describes the output as corridor position over a 450 cm track; it does not separately justify the clipping step.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses five equal-width bins over `0-450` cm, i.e. `0-90`, `90-180`, `180-270`, `270-360`, `360-450`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. The notes justify this choice directly from the decoder prompt’s “5 equal-sized bins” wording.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position is taken from the same `start:end` trial slice as the neural data and stored as a time-varying row in the output matrix with the same length.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
pos = nwb_data['position'][start:end]
...
output_arr[1, :] = pos_disc
```

iii. The notes rely on the general per-trial alignment-by-slicing design rather than a separate argument for position.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = bts['lick/data'][:]
...
lck = licks_corrected[start:end]
```

iii. `CONVERSION_NOTES.md` says the lick output comes from the lick signal after sensor-error correction.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first runs a lick-sensor QC pass that marks some whole trials as `NaN` when too many samples have lick values above 2. It then converts those NaNs to 0 and binarizes the remaining lick values with `> 0`.

ii.
```python
def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    ...
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > threshold:
        licks[start:end] = np.nan
...
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. `CONVERSION_NOTES.md` explicitly claims this follows the paper’s lick-sensor correction rule, then says the corrected lick counts are binarized for decoding.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same trial slice boundaries as neural and storing the binarized lick vector with the same number of timepoints.

ii.
```python
trial_n = neural[start:end, :].T.astype(np.float32)
lck = licks_corrected[start:end]
...
output_arr[3, :] = lck_binary
```

iii. The notes do not give a separate lick-alignment argument beyond the overall trial-start alignment scheme.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the scene metadata table (`SESSIONS_INFO`) and the parsed scene name, not from the NWB `reward_zone` signal.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
...
rz_labels = parse_scene_reward_zones(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The trajectory shows the agent explored the NWB `reward_zone` field, then decided to use session metadata instead. `CONVERSION_NOTES.md` says reward-zone locations are determined from `sessions_dict.py`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene name into zone labels `A/B/C`, applies a 30-trial before/after split for switch sessions, then encodes each trial’s zone as `0/1/2`.

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    ...
    zone_before, zone_after = parse_switch_zones(scene)
    ct = min(change_trial, n_trials)
    rz_labels[:ct] = zone_before
    rz_labels[ct:] = zone_after
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. `CONVERSION_NOTES.md` gives the same rule and cites the paper’s reward-zone switch after 30 trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward event timestamps, specifically `Reward/timestamps`, plus the trial start and end times from the session timestamp vector.

ii.
```python
timestamps = nwb_data['timestamps']
reward_ts = nwb_data['reward_ts']
...
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
```

iii. `CONVERSION_NOTES.md` says reward events have separate timestamps and a trial is rewarded if any event falls inside its boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code creates a per-trial binary label by checking whether any reward timestamp falls within that trial’s time span, then repeats that scalar across every timepoint in the trial output row.

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

iii. The notes justify this as a per-trial decoder target: `0=no`, `1=yes`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: it crops behavior and neural streams to a shared minimum length, truncates unequal start/end event lists to the shorter length, skips sessions with too few trials or no surviving neurons, skips degenerate trials, and replaces corrected-lick NaNs with zeros before binarization. It does not implement the reference survey/Viterbi handling for missing `reward_zone` information because it never uses that NWB signal to define trial labels.

ii.
```python
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    position = position[:min_len]
    ...
    deconvolved = deconvolved[:min_len]
...
n_trials = min(len(tstart_idx), len(teleport_idx))
...
if n_neurons < 1:
    return None
...
if end <= start:
    continue
if n_timepoints < 2:
    continue
...
lck = np.nan_to_num(lck, nan=0.0)
```

iii. The notes explicitly justify the one-sample neural/behavior truncation and the lick-sensor correction. The other defensive behaviors are visible in code but not separately explained.

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are full-session NWB reads, concatenating large deconvolved and fluorescence arrays, the per-neuron interneuron-correlation pass, the per-trial slicing/output construction loop, and writing the large pickle file.

ii.
```python
with h5py.File(filepath, 'r') as f:
    ...
    deconvolved = np.concatenate(deconv_list, axis=1)
    fluorescence = np.concatenate(fluor_list, axis=1)
...
for c in range(n_neurons):
    ...
for i in range(n_trials):
    ...
with open(output_file, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory shows the full conversion produced a multi-gigabyte pickle and that the agent repeatedly treated whole-file conversion/training runs as substantial jobs. I did not find a separate written performance analysis by the AI.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious non-vectorized loops are the neuron loop in `identify_interneurons`, the trial loop in `correct_lick_sensor_errors`, the trial loop used to assign `reward_per_trial`, the main per-trial construction loop in `process_session`, and the nested loops in `run_sanity_checks`.

ii.
```python
for c in range(n_neurons):
    ...
for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
    ...
for i in range(n_trials):
    ...
for s in range(n_sessions):
    for t in range(n_trials):
```

iii. This is an inference from the code structure; I did not find an explicit vectorization discussion in the notes or trajectory.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code makes multiple full passes over related data: one pass to identify interneurons, one pass to correct lick trials, one pass to compute per-trial reward outcomes, and one pass to materialize all per-trial arrays. Across end-to-end runs, the same conversion logic is rerun separately for sample and full outputs.

ii.
```python
is_interneuron = identify_interneurons(...)
licks_corrected, error_trials = correct_lick_sensor_errors(...)
for i in range(n_trials):
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
for i in range(n_trials):
    ...
```

iii. The trajectory shows the agent executed sample conversion, full conversion, verification, and training as separate stages; the code itself also clearly revisits the same session arrays several times.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several raw variables that are never used downstream (`trial_num`, `environment`, `reward_zone_ts`, `autoreward`, `plane_idx`, `subject_id`, `session_id`). Inside `process_session()` it also creates temporary inputs (`input_arr` placeholder, `input_time_varying`, `input_per_trial`) that are immediately overwritten or unused.

ii.
```python
trial_num = bts['trial number/data'][:]
environment = bts['environment/data'][:]
reward_zone = bts['reward_zone/data'][:]
autoreward = bts['autoreward/data'][:]
...
plane_idx = seg['planeIdx'][:]
...
subject_id = f['general/subject/subject_id'][()]
session_id = f['general/session_id'][()]
...
input_arr = np.array([
    time_from_start[0] if False else 0,
])
input_time_varying = time_from_start
input_per_trial = np.array([
    float(env_per_trial[i]),
    float(i),
    float(prev_outcome[i]),
], dtype=np.float32)
```

iii. I did not find any explicit justification for these unused reads or temporary arrays.
