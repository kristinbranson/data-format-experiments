# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globbed every NWB file one directory below `/app/data/sub-*`, sorted the paths, and processed each file directly with `h5py`. In full mode it attempted all 152 files; sessions returning fewer than two valid trials were omitted.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
for fpath in all_files:
    result = process_session(fpath, ...)
```

iii. The notes justify this from the observed one-subdirectory-per-mouse layout, the 11 expected mice, and 152 expected sessions. Direct HDF5 access was chosen because the needed NWB arrays were already aligned.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each file's `general/subject/subject_id`. A first-seen unique-subject list is built while sorted files are processed, and each retained session receives its index into that list.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The notes report 11 subjects (`m3`, `m4`, `m7`, `m11`–`m19` as applicable), matching the paper and directory survey.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session. Session metadata are read from `general/session_id`; a session is dropped if it begins with fewer than two trials or retains fewer than two trials after filtering.

ii.
```python
session_id = f['general/session_id'][()].decode()
if len(neural_trials) < 2:
    return None
neural_all.append(result['neural'])
```

iii. The agent reasoned that each file represents one continuous, already-aligned recording. Its reported 152 retained sessions matches the expected dataset total.

## 1-d. How are the data split into trials?

i. Trial starts are all positive samples of `trial_start`; trial ends are all positive samples of `teleport`. The arrays are paired by order and silently trimmed to equal length if needed. Slices are half-open `[start:end)`.

ii.
```python
trial_start_inds = np.where(trial_start_data > 0)[0]
teleport_inds = np.where(teleport_data > 0)[0]
min_len = min(len(trial_start_inds), len(teleport_inds))
trial_pos = position[ts:te]
```

iii. The notes say this follows the reference trial-start/teleport definition and excludes inter-trial teleport periods. They do not justify using every positive teleport sample rather than its rising edge.

## 1-e. How are trials filtered based on quality controls?

i. Sessions require at least two trials. Individual trials are skipped if the end is not after the start, if they contain fewer than two raw samples, or if fewer than two samples remain after requiring speed at least 2 cm/s and a non-NaN value in the first neural column. Lick-sensor failures change lick values but do not drop the trial.

ii.
```python
valid_mask = (trial_speed >= SPEED_THRESHOLD) & ~np.isnan(trial_neural[:, 0])
if te <= ts or n_trial_tp < 2 or valid_mask.sum() < 2:
    continue
```

iii. The agent cites the paper's `<2 cm/s` exclusion and the decoder's two-trial minimum. It states that all trials otherwise remain included.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `processing/ophys/Deconvolved/plane*/data`, concatenated across planes, then restricted using `ImageSegmentation/PlaneSegmentation/iscell[:,0]`.

ii.
```python
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_all = np.concatenate(plane_data, axis=1)
deconv = deconv_all[:, cell_mask]
```

iii. The notes claim the stored deconvolved events are the same events used by the paper's decoder and are therefore ready to use, rather than recomputing them from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. Plane arrays are concatenated, `iscell` filtering is applied, trial/time masks are applied, remaining NaNs are replaced by zero, the array is transposed to neuron-by-time, and it is cast to `float32`. No dF/F, neuropil subtraction, smoothing, or new deconvolution is performed.

ii.
```python
trial_neural_valid = np.nan_to_num(trial_neural[valid_mask, :], nan=0.0)
neural_matrix = trial_neural_valid.T.astype(np.float32)
```

iii. The agent says the NWB events are precomputed and reference-compatible. It describes the speed mask as matching the paper's analysis threshold.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell == 1` are retained. Timepoints below 2 cm/s or with NaN in the first retained neuron are removed. The documented interneuron filter (dF/F–speed correlation >0.5) is deliberately skipped.

ii.
```python
cell_mask = iscell == 1
valid_mask = speed_mask & neural_nan_mask
```

iii. The notes justify skipping interneuron exclusion because dF/F was not stored and the expected effect was under 0.5% of cells; the resulting maximum cell count above the paper's range is acknowledged.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural and behavior are sliced at the same trial-start and teleport indices. Trial start is the alignment event, but low-speed removal makes the first retained neural sample potentially later than the event.

ii.
```python
trial_neural = deconv[ts:te, :]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The notes state that NWB behavior is already aligned to imaging frames and that trial start is time zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Retained points remain imaging frames, nominally 15.5 Hz or 64.5 ms, although low-speed filtering creates irregular gaps. Metadata use a fixed `64.5` ms rather than each session's measured rate.

ii.
```python
'time_bin_size': 64.5,
frame_period = 1.0 / imaging_rate
```

iii. The agent cites the paper's approximately 15.5 Hz per-field rate and says no additional binning is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
timestamps = bts['position/timestamps'][:]
trial_timestamps = timestamps[ts:te]
```

iii. The notes say all behavioral streams are already frame-aligned, so position timestamps provide the imaging-frame clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. After applying the valid-timepoint mask, the original timestamp at `trial_start` is subtracted, preserving elapsed wall-clock time and gaps caused by removed samples.

ii.
```python
trial_times_valid = trial_timestamps[valid_mask]
time_from_start = trial_times_valid - trial_timestamps[0]
```

iii. The agent says this makes time zero coincide with entry to the virtual track.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `valid_mask` is applied to timestamps and neural rows, so every retained time value corresponds to the same retained neural frame.

ii.
```python
trial_neural_valid = trial_neural[valid_mask, :]
trial_times_valid = trial_timestamps[valid_mask]
```

iii. The justification is the NWB's common frame indexing and use of one mask for both arrays.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although `environment/data` is loaded, the output environment is inferred from the NWB `identifier` scene string.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
env_per_trial = get_environment_from_scene(scene, n_trials)
```

iii. The notes say scene names encode environment and that cross-environment switch sessions change after trial 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene prefixes `Env1`/`Env2` become 0/1. Cross-environment scenes are split at hard-coded trial 30; the scalar is broadcast across each trial.

ii.
```python
env[:change_trial] = env1_num - 1
env[change_trial:] = env2_num - 1
input_data[1, :] = env_type
```

iii. The notes report checking that all 11 environment switches occurred at trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index over detected trial starts, not the loaded raw `trial number/data`.

ii.
```python
for i in range(n_trials):
    trial_num = float(i)
```

iii. The mapping plan said the raw field would be used, but the implemented choice is only annotated as a “0-indexed trial number.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is cast to float and broadcast to every retained timepoint in the trial. Skipped trials leave gaps in the sequence.

ii.
```python
input_data[2, :] = trial_num
```

iii. No additional justification is supplied beyond representing within-session trial order.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It comes from `Reward/timestamps`, position timestamps, and detected trial boundaries.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                          (reward_timestamps <= trial_time_end))
```

iii. The notes interpret any reward delivery during a trial as rewarded and no delivery as omitted.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is marked rewarded if at least one reward timestamp lies inclusively between its start and teleport timestamp. The reward vector is shifted by one; trial zero gets 0. Values are broadcast over time.

ii.
```python
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
prev_outcome[1:] = reward_per_trial[:-1]
input_data[3, :] = prev_out
```

iii. This directly implements the requested rewarded/omitted binary variable and defines “no previous trial” as 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position/data` plus reward-zone coordinates inferred from the scene identifier and hard-coded A/B/C ranges.

ii.
```python
REWARD_ZONE_DICT = {'A': (80, 130), 'B': (200, 250), 'C': (320, 370)}
dist = compute_distance_to_reward_zone(trial_pos_valid, rz_start, rz_end)
```

iii. The notes cite the paper/code zone coordinates and scene-name validation, including switches at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the active zone, zero inside it, and positive after it, measured to the nearest boundary. It is evaluated only at retained timepoints.

ii.
```python
distance[before] = position[before] - rz_start
distance[after] = position[after] - rz_end
distance[inside] = 0.0
```

iii. The agent says this is signed linear distance to the nearest point of the active 50-cm reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks create the seven requested categories with boundaries -50, -10, 0, 10, and 50 cm.

ii.
```python
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
```

iii. The notes state that these categories reproduce the task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural rows use the same trial slice and valid mask before distance is calculated.

ii.
```python
trial_pos_valid = trial_pos[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The agent relies on the common NWB imaging-frame index.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavioral `position/data`.

ii.
```python
position = bts['position/data'][:]
trial_pos_valid = trial_pos[valid_mask]
```

iii. Position is documented as centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is speed/NaN masked, then discretized; there is no interpolation, clipping, or normalization.

ii.
```python
pos_disc = discretize_position(trial_pos_valid)
```

iii. The notes say the raw position range is covered by five equal track bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Positions are assigned to `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360` cm.

ii.
```python
out[position < 90] = 0
out[(position >= 270) & (position < 360)] = 3
out[position >= 360] = 4
```

iii. Five 90-cm bins are justified by the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same `[ts:te]` slice and `valid_mask` are applied to position and neural activity.

ii.
```python
trial_pos_valid = trial_pos[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The justification is shared NWB frame alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from behavioral `lick/data`.

ii.
```python
lick_raw = bts['lick/data'][:]
```

iii. The agent interprets this as a cumulative/event lick sensor stream and invokes the reference lick-sensor correction.

## 9-b. What processing is involved in computing `output` *Lick*?

i. For each trial, if more than 35% of samples exceed 2, all lick samples in that trial are changed to NaN. Positive values are then 1, everything else—including those NaNs—is 0.

ii.
```python
if np.sum(trial_licks > 2) / len(trial_licks) > LICK_ERROR_THRESHOLD:
    lick_corrected[ts:te] = np.nan
lick_binary[lick_corrected > 0] = 1
lick_binary[np.isnan(lick_corrected)] = 0
```

iii. The agent cites the reference code's 0.35 stuck-sensor threshold, but gives no scientific justification for converting unknown/bad lick data to “no lick.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Corrected lick data receive the same trial slice and valid-timepoint mask as neural activity.

ii.
```python
trial_lick_valid = trial_lick[valid_mask]
trial_neural_valid = trial_neural[valid_mask, :]
```

iii. The common NWB frame index is the stated justification.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred solely from the NWB `identifier` scene string and the trial index, not the raw `reward_zone` stream.

ii.
```python
scene = parse_scene(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. The notes say scene names explicitly encode zone identities and switch structure.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Single-location scenes receive one zone throughout. Switch scenes use the first encoded zone before trial 30 and the second afterward. A/B/C map to 0/1/2 and are broadcast over time.

ii.
```python
rz_labels[:change_trial] = loc1
rz_labels[change_trial:] = loc2
rz_label = REWARD_ZONE_LABEL_MAP[rz_labels[i]]
```

iii. The agent reports confirming all observed switch sessions changed at trial 30 and that zone classes were balanced.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward/timestamps`, position timestamps, and trial-start/teleport indices.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
trial_time_start = timestamps[ts]
trial_time_end = timestamps[te]
```

iii. Reward events are treated as direct evidence of delivery.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. At least one event in the inclusive trial time window yields 1; otherwise 0. The result is broadcast over every retained trial sample.

ii.
```python
reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0
output_data[5, :] = reward_out
```

iii. The agent says this implements the requested per-trial binary outcome and reports an overall 82.8% reward rate consistent with expected omissions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Start/end count mismatches are silently trimmed; invalid/very short trials and sessions are skipped; first-neuron NaN frames are removed; remaining neural NaNs are zero-filled; failed lick trials are converted to all-zero lick output. Missing or unparsable scene metadata raises an error. There is no explicit reconciliation of neural and behavioral array lengths or timestamp validation.

ii.
```python
trial_start_inds = trial_start_inds[:min_len]
teleport_inds = teleport_inds[:min_len]
trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)
```

iii. The notes describe these as defensive filtering and report no NaNs in the final neural arrays. Most individual fallback choices are not separately justified.

## 13-a. What are the most time-consuming steps of the code?

i. The agent's code times each session and the total/save phase but does not explicitly profile operations. The likely dominant steps are reading and concatenating the large deconvolved plane arrays, building/copying thousands of per-trial arrays, and serializing the 8.2-GB pickle; optional plotting and decoder training are outside normal conversion.

ii.
```python
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_all = np.concatenate(plane_data, axis=1)
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report the dataset size and total conversion/save timing but provide no dedicated performance analysis.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The reward-per-trial loop, lick-error loop, and trial-building loop are Python loops. Reward assignment could use `searchsorted`/binning, lick statistics could use segmented reductions, and some full-session discretization could occur before trial slicing. Trial list construction remains naturally loop-based because lengths vary.

ii.
```python
for i in range(n_trials):
    rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                              (reward_timestamps <= trial_time_end))
for i in range(n_trials):
    frac_error = np.sum(trial_licks > 2) / len(trial_licks)
```

iii. The agent provides no explicit vectorization discussion; its implementation favors direct, readable per-trial processing.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trials three times: reward classification, lick correction, and final extraction. It also repeatedly allocates masks and slices the same session arrays, and repeatedly evaluates all reward timestamps for each trial.

ii.
```python
for i in range(n_trials):  # rewards
    ...
for i in range(n_trials):  # lick correction
    ...
for i in range(n_trials):  # conversion
    ...
```

iii. No explicit justification is documented; separating stages makes the derived previous-outcome and corrected-lick arrays simple to use later.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `trial number/data` and `environment/data` but never uses either, computes an unused `n_timepoints` and `frame_period`, collects unused `subjects_list`, and performs lick-error NaN assignment only to immediately convert those NaNs to zero. Optional plotting performs additional diagnostics only when requested.

ii.
```python
trial_number_data = bts['trial number/data'][:]
environment_data = bts['environment/data'][:]
n_timepoints = len(position)
frame_period = 1.0 / imaging_rate
subjects_list = []
```

iii. The notes planned to use the two raw behavioral fields, which likely explains their loading, but do not discuss the dead variables or discarded intermediate values.
