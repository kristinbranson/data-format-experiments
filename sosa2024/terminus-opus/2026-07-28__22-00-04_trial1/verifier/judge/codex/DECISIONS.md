# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file matching `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`, then processes each file as one session. Within each file it reads the behavioral and ophys groups directly with `h5py` rather than `pynwb`.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, show_processing=args.show_processing, session_idx=i)
```

```python
f = h5py.File(nwb_path, 'r')
beh = f['processing']['behavior']['BehavioralTimeSeries']
deconv_group = f['processing']['ophys']['Deconvolved']
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose to "Load NWB files using h5py" and notes the NWB layout it found during exploration. The trajectory also records an explicit design choice to load all NWB files with `h5py`.

## 1-b. How are the data split into subjects (mice)?

i. The AI derives subject identity from each NWB file’s stored `subject_id`, then builds the final `subjects` list from the unique IDs seen across sessions.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
...
all_subj_ids.append(result['subj_id'])
...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
```

iii. The notes emphasize that the NWB files are organized by subject and session, and the trajectory shows the AI verified the `subject_id` metadata during NWB exploration before using it as the authoritative subject label.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. Session identity is taken from the file metadata and the output arrays append one top-level session entry per processed file.

ii.
```python
sess_id = f['general']['session_id'][()].decode()
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, show_processing=args.show_processing, session_idx=i)
    ...
    all_neural.append(result['neural_trials'])
```

iii. In the notes, the AI describes the dataset layout as one NWB file per session and reports 152 total sessions from the file tree. The trajectory shows it inspected file names and NWB metadata and then adopted that one-file/one-session mapping.

## 1-d. How are the data split into trials?

i. Trials are segmented using `trial_start` samples as trial starts and `teleport` samples as trial ends. For each trial, the AI slices all time-varying arrays from `t_start` to `t_end`.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
...
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    n_tp = t_end - t_start
```

iii. `CONVERSION_NOTES.md` lists "Trial boundaries: trial_start_inds and teleport_inds" as a key match to the reference logic. The trajectory shows the AI explored both `trial number` and `trial_start` and decided the `trial_start`/`teleport` pair was the correct boundary definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference script’s nominal `< 50` timepoint exclusion. It only drops trials with fewer than 2 timepoints, and separately skips entire sessions if they end up with fewer than 2 trials.

ii.
```python
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    n_tp = t_end - t_start
    
    if n_tp < 2:
        continue
```

```python
if len(result['neural_trials']) < 2:
    print(f"  SKIPPING: fewer than 2 trials")
    continue
```

iii. The notes justify the session-level requirement by citing the decoder format requirement that a session must contain at least two trials. I did not find an explicit note justifying the much weaker per-trial filter beyond that.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from the NWB `processing/ophys/Deconvolved/plane*/data` arrays, i.e. precomputed deconvolved calcium-event traces.

ii.
```python
deconv_group = f['processing']['ophys']['Deconvolved']
plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
if len(plane_keys) == 1:
    deconv_data = deconv_group[plane_keys[0]]['data'][:]
else:
    plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
    deconv_data = np.concatenate(plane_data, axis=1)
```

iii. The notes explicitly state "Neural: Deconvolved calcium events (OASIS via Suite2p)" and that the NWB files already contain the precomputed deconvolved events, so no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The AI concatenates multiple imaging planes along the ROI axis, filters ROIs with `iscell`, and transposes each trial to `(n_neurons, n_timepoints)`. It does not further smooth, normalize, or rebin the deconvolved traces.

ii.
```python
if len(plane_keys) == 1:
    deconv_data = deconv_group[plane_keys[0]]['data'][:]
else:
    plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
    deconv_data = np.concatenate(plane_data, axis=1)
...
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` lists "Filter by iscell, transpose" and "Multi-plane pooling: Concatenate plane0 and plane1 data for m17, m18" as core implementation choices. The trajectory shows the AI deliberately avoided extra calcium preprocessing because the NWB data already store deconvolved events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural-quality filter in the conversion code is `iscell == 1`.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
...
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. The notes say "Use iscell as-is" and explicitly note that interneuron exclusion was skipped because the AI believed the effect would be small. The trajectory records the same decision after reading the methods section.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by trial segmentation itself: each trial begins at the `trial_start` sample, and the per-trial matrix is the slice from that start to the matching teleport sample.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The notes explicitly say "Temporal alignment: per-trial, from trial_start to teleport" and later describe the alignment event in metadata as `trial_start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sampling interval from the behavioral timestamps, using `dt = median(diff(position.timestamps))`, and does not apply temporal rebinning.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
...
dt = np.median(np.diff(pos_timestamps))
...
'time_bin_size': median_dt * 1000,
```

iii. The notes say "Native frame rate: Use ~64.5 ms time bins (no resampling)" and report ~15.51 Hz / 64.48 ms from the explored data.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives this input from the `position` timestamps and the trial boundaries. It does not use the `trial number` timestamps directly.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
...
dt = np.median(np.diff(pos_timestamps))
...
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
n_tp = t_end - t_start
time_from_start = np.arange(n_tp) * dt
```

iii. The notes argue that all behavioral streams share the same frame timing and that the native frame duration is ~64.5 ms. The trajectory shows the AI settled on the behavior timestamp grid and then simplified to a constant-`dt` construction.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI computes a regularly spaced vector `0, dt, 2*dt, ...` with one entry per frame in the trial.

ii.
```python
n_tp = t_end - t_start
time_from_start = np.arange(n_tp) * dt
...
input_data[0, :] = time_from_start
```

iii. The notes justify this through the fixed frame rate and native time-bin decision. There is no separate complicated alignment step because the AI assumes evenly sampled behavior and neural frames.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Alignment is by shared trial indexing: the time vector length is `n_tp = t_end - t_start`, the same frame count used to slice the neural matrix.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
...
n_tp = t_end - t_start
time_from_start = np.arange(n_tp) * dt
input_data = np.zeros((4, n_tp), dtype=np.float32)
```

iii. The notes emphasize native-frame alignment and no resampling. The trajectory also records that the AI viewed the neural and behavior streams as already aligned in the NWB export.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
env_data = beh['environment']['data'][:]
...
env_vals = env_data[t_start:t_end]
```

iii. The notes identify `environment data` as the source variable for this input and summarize it as a binary 0/1 per-trial feature.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI removes negative codes (`-1` before valid trial periods), takes the median environment value within each trial, converts it to an integer, and then repeats that constant value across all timepoints in the trial.

ii.
```python
trial_env = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    ...
    env_vals = env_data[t_start:t_end]
    env_vals = env_vals[env_vals >= 0]
    if len(env_vals) > 0:
        trial_env[t] = int(np.median(env_vals))
...
input_data[1, :] = env_type
```

iii. The notes describe environment as a per-trial binary context variable, and the trajectory records that the AI saw `-1` values outside valid scanning/trial periods and treated the within-trial environment as effectively constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the per-trial loop index after defining trial boundaries from `trial_start` and `teleport`.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = float(t)
```

iii. The notes map "Trial index" to this input, and the trajectory shows the AI rejected the stored NWB `trial number` series as the primary trial definition in favor of the segmented trial order.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing beyond assigning the zero-based trial index and repeating it across the full trial.

ii.
```python
trial_num = float(t)
...
input_data[2, :] = trial_num
```

iii. The notes describe this variable as "0-indexed" and per-trial. I did not find a more elaborate justification beyond using the segmented within-session trial order.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The AI derives reward outcome from `Reward.timestamps` aligned onto the behavioral frame timestamps, then uses the previous trial’s reward flag.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
pos_timestamps = beh['position']['timestamps'][:]
...
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
...
trial_rewarded[t] = 1 if reward_in_trial else 0
```

iii. The notes map "Previous trial reward" to this input and describe reward delivery as sparse events that must be aligned to the imaging/behavior frame grid.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first marks each trial as rewarded or omitted based on whether any reward timestamp falls between `t_start` and `t_end`. It then fills the current trial’s previous-outcome input with `0` for the first trial and `trial_rewarded[t-1]` otherwise.

ii.
```python
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
...
if t == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(trial_rewarded[t - 1])
```

iii. The notes explicitly define this feature as binary previous reward outcome. The trajectory shows the AI verified this against its own spot checks in `CONVERSION_NOTES.md`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives distance from trial-wise `position` together with a trial reward-zone identity inferred from the `reward_zone` time series. It finds the minimum position where `reward_zone > 0` inside each trial, maps that start position to A/B/C with a tolerance, and fills omission trials from neighboring trials.

ii.
```python
position = beh['position']['data'][:]
rzone = beh['reward_zone']['data'][:]
...
rz_starts, rz_labels = get_reward_zone_start_per_trial(
    position, rzone, trial_start_inds, teleport_inds)
...
def get_reward_zone_start_per_trial(pos, rzone, trial_start_inds, teleport_inds):
    ...
    rz_active = trial_rzone > 0
    if np.any(rz_active):
        rz_start_pos = trial_pos[rz_active].min()
        rz_labels[t] = determine_reward_zone_label(rz_start_pos)
```

iii. The notes say "Determine reward zone from position data where rzone > 0" and later state that omission trials use the same reward zone as neighboring trials. The trajectory shows this was a deliberate simplification relative to the original analysis code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside it, and positive after it.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start):
    rz_end = rz_start + 50.0
    distance = np.zeros_like(position)
    before_mask = position < rz_start
    distance[before_mask] = position[before_mask] - rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    distance[in_mask] = 0.0
    after_mask = position > rz_end
    distance[after_mask] = position[after_mask] - rz_end
    return distance
```

iii. `CONVERSION_NOTES.md` says "Distance to reward zone: Computed from position relative to reward zone boundaries." The trajectory shows the AI adopted the paper’s 50 cm reward-zone width and fixed A/B/C start positions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI intends to use the seven instruction-defined bins and encodes them with chained comparisons rather than `np.digitize`.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The notes describe this output as "7-bin discretization" and clearly aim to follow the decoder specification. I did not find a separate discussion of the exact edge inclusivity choices.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same `t_start:t_end` trial slice that defines the neural matrix for the trial.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
...
trial_pos = position[t_start:t_end]
distance = compute_distance_to_reward_zone(trial_pos, rz_start)
```

iii. The notes repeatedly describe the conversion as native-frame and per-trial, with no temporal resampling between neural and behavioral streams.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[t_start:t_end]
```

iii. The notes map `position` directly to the absolute-position decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the trial’s raw position trace and discretizes it; there is no additional smoothing or interpolation.

ii.
```python
trial_pos = position[t_start:t_end]
...
pos_bins = discretize_position(trial_pos, n_bins=5)
```

iii. The notes describe the transform simply as "Position | 5-bin equal discretization" with no extra preprocessing.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI divides the 450 cm track into five equal-width bins using edges from `np.linspace(0, TRACK_LENGTH, 6)`, then clips out-of-range values into the end bins.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The notes explicitly say "Absolute position | 5-bin equal discretization" and treat the 450 cm track length as the relevant span, which is why the AI chose equal 90 cm bins over `[0, 450]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by taking the same per-trial frame slice as the neural data.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_pos = position[t_start:t_end]
```

iii. The justification is implicit in the notes’ repeated claim that all outputs are kept at the native frame rate and segmented with the same trial boundaries as neural activity.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick_data = beh['lick']['data'][:]
...
trial_licks = licks_processed[t_start:t_end]
```

iii. The notes list `lick` as the source for the lick output and mention the original code’s lick artifact logic as something to reproduce.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies the original analysis pipeline’s lick cleanup: per-trial artifact rejection if too many samples exceed 2, clipping values above 1, Gaussian smoothing with `sigma=2`, and finally thresholding the smoothed trace at `> 0.5` to obtain a binary lick output.

ii.
```python
def process_licks(lick_data, trial_start_inds, teleport_inds):
    ...
    if len(trial_licks) > 0:
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > LICK_ERROR_THRESHOLD:
            licks[t_start:t_end] = np.nan
    licks[licks > 1] = 1
    return licks
...
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
lick_binary[np.isnan(smoothed_licks)] = 0
```

iii. The notes say "Processes licks with error correction and smoothing" and explicitly cite the reference code’s 35% sensor-error threshold and Gaussian smoothing. The trajectory shows this was one of the AI’s strongest attempts to mimic the original analysis code.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The cleaned lick trace is aligned by session-wide preprocessing followed by the same `t_start:t_end` trial slicing used for neural activity.

ii.
```python
licks_processed = process_licks(lick_data, trial_start_inds, teleport_inds)
...
trial_licks = licks_processed[t_start:t_end]
...
output_data[3, :] = lick_binary
```

iii. The notes describe lick as remaining on the native frame grid, so alignment is handled by shared trial indices rather than separate resampling.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the `reward_zone` behavior signal together with the animal’s `position`, using the same trial-wise reward-zone inference used for distance-to-zone.

ii.
```python
position = beh['position']['data'][:]
rzone = beh['reward_zone']['data'][:]
...
rz_starts, rz_labels = get_reward_zone_start_per_trial(
    position, rzone, trial_start_inds, teleport_inds)
```

iii. The notes treat reward-zone location and distance-to-zone as two outputs built from the same inferred trial reward-zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers a label A/B/C from the minimum in-zone position within each trial, then replaces missing labels on omission trials by the nearest labeled neighboring trial. Finally it maps A/B/C to integers 0/1/2 and repeats the per-trial value across the whole trial.

ii.
```python
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None
...
def get_reward_zone_for_trial(rz_labels, trial_idx):
    if rz_labels[trial_idx] is not None:
        return rz_labels[trial_idx]
    for offset in range(1, len(rz_labels)):
        if trial_idx - offset >= 0 and rz_labels[trial_idx - offset] is not None:
            return rz_labels[trial_idx - offset]
        if trial_idx + offset < len(rz_labels) and rz_labels[trial_idx + offset] is not None:
            return rz_labels[trial_idx + offset]
    return 'A'
...
trial_rz_label[t] = rz_label_map.get(label, 0)
output_data[4, :] = rz_loc
```

iii. The notes justify this as a practical way to handle omission trials while keeping zone assignments stable enough for decoding. The trajectory shows the AI knew the original code used richer reward-zone logic but chose this simpler local heuristic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward.timestamps` aligned to the behavioral frame grid.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. The notes describe reward outcome as a binary per-trial variable derived from sparse reward delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI maps reward timestamps to frame indices, tests whether each trial interval contains any mapped reward event, stores that as `trial_rewarded[t]`, and repeats the result across all timepoints in the trial.

ii.
```python
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
...
reward_in_trial = np.any(
    (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
)
trial_rewarded[t] = 1 if reward_in_trial else 0
...
output_data[5, :] = reward_out
```

iii. The notes map "Reward delivery" to this output and report that reward rates looked consistent with the expected omission fraction, which the AI treated as a sanity check for this logic.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases defensively, but with different choices from the human reference: if `trial_start` and `teleport` counts disagree it truncates to the smaller count; omission trials with no detected reward-zone samples borrow the nearest neighboring trial’s reward-zone label, with a final fallback to `'A'`; lick-sensor-error trials are turned into `NaN` and later back into 0 after smoothing. It does not implement the reference crop-on-length-mismatch safeguard between neural and behavior streams.

ii.
```python
if n_trials != len(teleport_inds):
    print(f"  WARNING: trial_start ({n_trials}) != teleport ({len(teleport_inds)}) count")
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
```

```python
if frac_high > LICK_ERROR_THRESHOLD:
    licks[t_start:t_end] = np.nan
...
if trial_idx - offset >= 0 and rz_labels[trial_idx - offset] is not None:
    return rz_labels[trial_idx - offset]
...
return 'A'
```

iii. The notes justify these behaviors as practical edge-case handling, especially for omission trials and lick-sensor artifacts. The trajectory shows the AI was actively trying to preserve decodable outputs even when some raw trial annotations were missing.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s expensive steps are loading each NWB file, reading large neural and behavioral arrays, looping over all trials to build session outputs, and writing the final multi-gigabyte pickle. Optional processing plots add more time when enabled.

ii.
```python
f = h5py.File(nwb_path, 'r')
...
for t in range(n_trials):
    ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes report a full conversion time of about 70 seconds and a final pickle near 9.4 GB, which implicitly identifies file I/O and whole-dataset serialization as major costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized work is the repeated per-trial looping for reward-zone extraction, environment summarization, reward detection, lick artifact checking, and final trial assembly. Several of these could be partly vectorized or precomputed over full-session arrays before splitting into Python lists of variable-length trials.

ii.
```python
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    ...
```

```python
for t in range(len(trial_start_inds)):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    trial_licks = licks[t_start:t_end]
```

iii. I did not find an explicit efficiency discussion in the notes beyond runtime estimates. This answer is mostly inferred from the structure of the conversion code.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly scans trial boundaries over the same session-level arrays: once to find reward-zone starts, once to clean licks, once to determine per-trial rewards, once to summarize per-trial environment, and again to build the final neural/input/output trial objects.

ii.
```python
rz_starts, rz_labels = get_reward_zone_start_per_trial(
    position, rzone, trial_start_inds, teleport_inds)
licks_processed = process_licks(lick_data, trial_start_inds, teleport_inds)
...
for t in range(n_trials):
    ...
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
...
for t in range(n_trials):
    ...
    env_vals = env_data[t_start:t_end]
...
for t in range(n_trials):
    ...
    neural_trials.append(trial_neural)
```

iii. The notes focus more on correctness than efficiency, so this is largely a structural inference from the code rather than an explicitly documented design justification.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `scanning` but never uses it, collects `all_sess_ids` but never saves them, computes several plotting-only summaries when `--show-processing` is enabled, and prints detailed neuron/session summaries that are not part of the converted dataset.

ii.
```python
scanning = beh['scanning']['data'][:]
...
all_sess_ids = []
...
all_sess_ids.append(result['sess_id'])
```

```python
if show_processing and session_idx < 2:
    plot_processing(...)
```

iii. I did not find an explicit justification for these pieces beyond debugging and validation convenience. The notes make clear that plotting and extensive summary checks were used as sanity checks rather than as part of the final decoder input.
