# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `data/sub-*/*.nwb` file, sorted the paths, and processed each file as one session. It used `h5py` to read NWB datasets directly. Full mode uses every matched file; sample mode substitutes two files.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
files_to_process = all_files
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, ...)

with h5py.File(nwb_path, 'r') as f:
    behav = f['processing/behavior/BehavioralTimeSeries']
```

iii. The notes report 11 subjects and 152 sessions, matching the available dataset and paper. The agent chose direct HDF5 access for speed and confirmed the full conversion found all 152 files.

## 1-b. How are the data split into subjects?

i. The subject is parsed from the filename prefix (for example, `sub-m11`). A first-seen ordered subject list is built, and every session receives an index into it.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
all_subject_idx.append(subj_idx)
```

iii. The agent justified this from the stable `sub-{mouse}` directory/file convention and checked that 11 unique mice were produced.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session; its trials are appended as one element of each session-level list.

ii.
```python
for file_idx, nwb_path in enumerate(files_to_process):
    neural_trials, input_trials, output_trials, ... = process_session(nwb_path, ...)
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. This follows the dataset naming/organization. The notes state that the resulting 152 sessions match the expected count.

## 1-d. How are the data split into trials?

i. Trial starts are all positive `trial_start` samples and trial ends are all positive `teleport` samples. Corresponding arrays are paired in order and each emitted slice is `[start:stop)`. If counts differ, both are truncated to the smaller count.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
n_trials = min(len(trial_starts), len(teleports))
...
start = trial_starts[i]
stop = teleports[i]
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes identify `trial_start` to `teleport` as the reference trial definition. The agent inspected unusually long trials and concluded they reflected slow/stopped animals rather than bad boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Only non-positive-duration trials are dropped inside a session. Sessions with fewer than two retained trials are skipped. There is no minimum-50-sample trial filter.

ii.
```python
if stop <= start:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. The agent did not document a short-trial exclusion; it retained long and short valid slices and relied on format validation. This differs from the human conversion's stated short-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from every plane under the NWB `processing/ophys/Deconvolved` group; planes are concatenated across ROIs. It does not use raw `Fluorescence` and `Neuropil`.

ii.
```python
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The trajectory says the agent believed the stored field contained events already deconvolved after dF/F and used it to avoid recomputing the complex maximin baseline. This belief conflicts with the human finding that this is Suite2p's separate stored deconvolution, not the paper-analysis signal.

## 2-b. How is the `neural` data processed?

i. Stored deconvolved planes are concatenated, truncated to behavioral length if necessary, filtered by the cell mask, sliced into trials, transposed to neuron-by-time, and cast to `float32`. The paper's neuropil subtraction, trial-aware maximin dF/F, smoothing, and OASIS recomputation are not performed.

ii.
```python
deconv_data = np.concatenate(deconv_planes, axis=1)
deconv_data = deconv_data[:min_len, :]
neural_data = deconv_data[:, cell_indices]
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The agent explicitly called recomputing dF/F “complex” and treated the stored deconvolution as adequate. Its sanity check only established exact preservation of that stored field, not equivalence to the paper pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must have `iscell[:,0] == 1`. Putative interneurons are then approximated as cells whose stored deconvolved activity has Pearson correlation greater than 0.5 with nonnegative speed, using samples where raw `trial number >= 0`.

ii.
```python
cell_mask = iscell[:, 0] == 1
valid_mask = trial_number >= 0
r = np.corrcoef(deconv_data[valid_mask, c], speed_valid[valid_mask])[0, 1]
interneuron_mask = speed_corr > 0.5
cell_mask = cell_mask & ~interneuron_mask
```

iii. The threshold and `iscell` curation follow the paper, but the agent acknowledged that the paper correlates dF/F—not stored events—with speed and described its implementation as an approximation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are sliced beginning at the detected trial-start index, so column zero is the trial-start-aligned sample. No interpolation or padding is applied.

ii.
```python
start = trial_starts[i]
stop = teleports[i]
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The notes state that behavior and neural arrays share sampling indices and that alignment is therefore achieved by common slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The median difference of behavioral position timestamps is stored; the reported value is about 64.48 ms (15.5 Hz).

ii.
```python
dt = np.median(np.diff(timestamps))
...
'time_bin_size': session_infos[0]['dt'] * 1000
```

iii. The agent retained native imaging/behavior frames because the notes found a common approximately 15.5 Hz rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps` and trial starts from `trial_start/data`.

ii.
```python
timestamps = behav['position/timestamps'][()]
trial_starts = np.where(trial_start_signal > 0)[0]
```

iii. The agent verified the timestamps against converted time values and treated behavioral and neural sampling as index-aligned.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first selected timestamp is subtracted from every selected timestamp and the result is cast to `float32`.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. This directly implements elapsed seconds from the requested alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `[start:stop)` indices are used for timestamps and neural data. Length mismatches between the full behavior and neural arrays are first truncated to their common minimum.

ii.
```python
deconv_data = deconv_data[:min_len, :]
timestamps = timestamps[:min_len]
trial_neural = neural_data[start:stop, :].T
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. The agent's raw-versus-converted check reported exact time agreement and matching per-trial dimensions.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although the raw `environment` stream is loaded, the output environment is derived by parsing the NWB `identifier` scene name and assuming a switch at trial 30.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The agent found environment values 0/1 with -1 between trials and verified scene parsing and trial-30 switches against the raw stream.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene strings map `Env1` to 0 and `Env2` to 1. Environment-switch sessions change at zero-based trial index 30; each per-trial value is broadcast over time.

ii.
```python
env_map = {'Env1': 0, 'Env2': 1}
if is_env_switch and i >= change_trial:
    env_vals.append(env_map.get(env_after, 0))
...
np.full(n_tp, env_per_trial[i], dtype=np.float32)
```

iii. The trajectory reports checking all switch patterns and confirming the trial-30 assumption against NWB signals.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index of paired trial-start/teleport boundaries, not the loaded NWB `trial number` values.

ii.
```python
for i in range(n_trials):
    ...
    np.full(n_tp, i, dtype=np.float32)
```

iii. The notes define this as the within-session trial index. The raw trial-number stream is used only to form the interneuron-correlation validity mask.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond zero-based indexing and broadcasting the scalar across every trial timepoint.

ii.
```python
np.full(n_tp, i, dtype=np.float32)
```

iii. This supplies a per-trial variable in the required two-dimensional input layout.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, behavioral timestamps, and the preceding trial's start/end time interval.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
reward_in_trial = np.any((reward_timestamps >= trial_start_time) &
                         (reward_timestamps <= trial_end_time))
```

iii. The agent used timestamp intervals because rewards are an event series with a separate time base.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each current trial is first labeled rewarded if any reward timestamp lies between its start timestamp and its last included timestamp. Previous outcome is 0 for trial 0 and the preceding label otherwise, broadcast over time.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
...
np.full(n_tp, prev_outcome[i], dtype=np.float32)
```

iii. This implements the requested omitted/rewarded binary history, including the conventional zero for no previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw position samples plus reward-zone coordinates inferred from the scene identifier and a trial-30 switch rule. The loaded raw `reward_zone` signal is not used in conversion.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The agent used known paper zones A/B/C and reports validating assignments across all 77 switch sessions against observed reward-zone data with no mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to 0–450 cm. Distance is negative relative to the zone's leading edge before the zone, zero inside its inclusive interval, and positive relative to its trailing edge after it.

ii.
```python
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
dist[before_mask] = position[before_mask] - rz_start
dist[in_mask] = 0.0
dist[after_mask] = position[after_mask] - rz_end
```

iii. The agent interpreted “distance to any location in the reward zone” as signed distance to the nearest zone edge, with zero throughout the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven instructed categories, including exact zero as class 3.

ii.
```python
bins[dist < -50] = 0
bins[(dist >= -50) & (dist < -10)] = 1
bins[(dist >= -10) & (dist < 0)] = 2
bins[dist == 0] = 3
bins[(dist > 0) & (dist <= 10)] = 4
bins[(dist > 10) & (dist <= 50)] = 5
bins[dist > 50] = 6
```

iii. The agent stated these boundaries directly follow the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the same `[start:stop)` sample range, producing equal-length arrays without interpolation.

ii.
```python
trial_neural = neural_data[start:stop, :].T
trial_pos = position[start:stop]
```

iii. The notes report dimensional and raw-data checks confirming common sample alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = behav['position/data'][()]
trial_pos = position[start:stop].astype(np.float32)
```

iii. The agent treated this stream as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Each trial slice is cast to `float32`, clipped to the nominal 0–450 cm track, then discretized.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. Clipping was intended to enforce the known physical track range. The human reference instead lets open end bins absorb small out-of-range values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.linspace(0,450,6)` creates 90 cm intervals; `np.digitize` against the four internal edges returns classes 0–4 and values are clipped to that class range.

ii.
```python
bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(position, bin_edges[1:-1])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. Five equal bins over 450 cm directly follow the requested thresholds.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial indices slice position and neural samples.

ii.
```python
trial_neural = neural_data[start:stop, :].T
trial_pos = position[start:stop]
```

iii. The agent verified trial-level timepoint counts and position discretization against raw data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = behav['lick/data'][()]
trial_licks = lick_corrected[start:stop]
```

iii. Exploration showed cumulative lick values from 0 to 5 rather than an already binary stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Within each trial, if more than 35% of samples exceed 2, that trial's lick values become NaN. Values greater than 1 are capped at 1, and NaNs are then converted to 0. A final `>0` comparison yields binary classes.

ii.
```python
if frac_high > 0.35:
    lick_corrected[start:stop] = np.nan
lick_binary[lick_binary > 1] = 1
lick_binary[np.isnan(lick_binary)] = 0
lick_disc = (trial_lick > 0).astype(int)
```

iii. The agent followed the repository code's 0.35 sensor-error threshold rather than the paper's 0.30 wording, and chose zero so the decoder format would contain no NaNs. It later observed no examined trials crossing this threshold.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Corrected lick and neural arrays are sliced with the same trial start and stop indices.

ii.
```python
trial_neural = neural_data[start:stop, :].T
trial_lick = lick_binary[start:stop].astype(np.float32)
```

iii. Shared sample indexing provides the alignment; no event resampling is done.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB identifier's scene string, not directly from the loaded `reward_zone` stream.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
```

iii. The agent chose scene metadata because it encodes the intended zone even on omission trials and reports cross-checking it against the reward-zone signal.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene syntax is parsed into before/after labels. Switches occur at trial 30, labels A/B/C map to 0/1/2, and the resulting value is broadcast across the trial.

ii.
```python
if is_switch and i >= change_trial:
    label = rz_after
else:
    label = rz_before
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
np.full(n_tp, rz_label_map[rz_labels[i]], dtype=int)
```

iii. The trajectory records a check of 77 switch sessions with no detected mismatch and confirms trial 30 as the switch point.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` and each trial's position timestamp interval.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
trial_start_time = timestamps[start]
trial_end_time = timestamps[stop - 1]
```

iii. The event timestamps have a separate time base, so the agent tested membership in trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies inclusively from its first through last included behavioral timestamp, otherwise 0; that scalar is broadcast over time.

ii.
```python
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
...
np.full(n_tp, reward_out, dtype=int)
```

iii. The resulting omission rate was about 15.3%, matching the paper's approximately 15% rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length mismatches are silently truncated to the shorter length. Unequal start/end counts are warned about and truncated to the smaller count. Non-positive trials and sessions with fewer than two trials are skipped. Lick-sensor NaNs are replaced by zero. Several expected timestamp/dimension relationships are not asserted.

ii.
```python
min_len = min(n_behav, n_neural)
position = position[:min_len]
deconv_data = deconv_data[:min_len, :]
...
n_trials = min(len(trial_starts), len(teleports))
...
lick_binary[np.isnan(lick_binary)] = 0
```

iii. The notes specifically identify occasional one-sample neural/behavior mismatch and multi-plane dimensional issues discovered during conversion; truncation and plane concatenation were introduced to make those sessions usable.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large NWB arrays, correlating every curated ROI separately with speed, retaining/copying dense neuron-by-time trial matrices, serializing the roughly 9.4 GB pickle, and the optional decoder training dominate. The conversion log reports roughly 200 seconds total, while neural arrays dominate storage.

ii.
```python
deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
for c in range(deconv_data.shape[1]):
    r = np.corrcoef(...)[0, 1]
...
pickle.dump(data, f, protocol=4)
```

iii. The notes estimate about 1.3 seconds per session and explain that the dense `float32` neural data account for almost the entire 9.2–9.4 GB output.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell correlation loop could be vectorized using centered matrix operations; per-trial reward-event scans could use mapped reward indices; environment/reward-zone lists could be constructed with array assignment; output-distribution collection could use concatenation. Trial construction itself remains naturally looped because lengths vary.

ii.
```python
for c in range(deconv_data.shape[1]):
    ... np.corrcoef(...)
for i in range(n_trials):
    reward_in_trial = np.any(...)
for i in range(n_trials):
    ... neural_trials.append(trial_neural)
```

iii. The agent did not explicitly document vectorization opportunities. Its emphasis was on acceptable per-session runtime and the necessity of variable-length trial lists.

## 13-c. What processing does the code repeat multiple times?

i. It makes several separate passes over trials: reward outcomes, lick correction, previous outcomes, trial construction, optional plotting, and final output distributions. It also repeatedly creates trial-length constant arrays for per-trial variables.

ii.
```python
for i in range(n_trials):  # reward labels
for i in range(n_trials):  # lick correction
for i in range(1, n_trials):  # prior outcome
for i in range(n_trials):  # construct arrays
...
for sess_outputs in all_output:
    for trial_output in sess_outputs:
```

iii. The notes do not flag these repeated passes; they report conversion as fast enough relative to data loading and storage.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several loaded variables are unused in the final conversion (`reward_zone_signal`, raw `environment`, and most `trial_number` values). Reward-zone centers are computed but unused. `in_mask` is redundant because distance is initialized to zero. Optional plotting recomputes summaries, and the final distribution report materializes large Python lists solely for printing.

ii.
```python
REWARD_ZONE_CENTERS = {k: (v[0] + v[1]) / 2 ...}
reward_zone_signal = behav['reward_zone/data'][()]
environment = behav['environment/data'][()]
in_mask = (position >= rz_start) & (position <= rz_end)
dist[in_mask] = 0.0
all_vals.extend(trial_output[out_idx].tolist())
```

iii. The agent loaded the unused streams partly for exploration, truncation consistency, and intended sanity checking, but its code comment for reward-zone verification has no implementation. The notes do not identify this work as unnecessary.
