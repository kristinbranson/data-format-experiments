# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB file one directory below `data/sub-*`, sorts the paths, and processes each with `h5py`. In normal/full mode it does not intentionally subsample; `--sample` selects two files.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
```
```python
with h5py.File(nwb_path, 'r') as f:
    position = bts['position/data'][:]
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
```

iii. The notes state that the layout contains 11 subject directories and 152 NWB sessions, and report a full conversion of all 152 sessions. The agent chose direct HDF5 access because the required arrays were identifiable in the NWB hierarchy.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, unique IDs are sorted, and each retained session gets an index into that sorted list.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
subjects = sorted(subjects_set)
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes verified 11 subject directories/IDs matching the paper (m3, m4, m7, m11–m15, m17–m19).

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session. Invalid sessions would be omitted if they had fewer than two cells or trials after the implemented checks.

ii.
```python
result = process_session(nwb_path, ...)
if result is not None:
    all_sessions.append(result)
```

iii. The agent inferred session identity from the one-file-per-session naming/layout and confirmed 152 files/sessions and the expected per-subject counts.

## 1-d. How are the data split into trials?

i. Trial starts are every positive `trial_start` sample and ends are every positive `teleport` sample. The two lists are independently truncated to their shorter length, paired by order, and sliced as `[start:end)`.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
...
trial_neural = neural_data[:, s:e]
```

iii. The notes say the flags mark exact NWB frames and that spot checks produced positions near 0–450 cm while excluding teleport. The trajectory explicitly revisited the apparent one-sample difference from the repository code and judged the NWB flag convention equivalent.

## 1-e. How are trials filtered based on quality controls?

i. Only malformed trials with `end <= start` or fewer than two samples are skipped. Sessions with fewer than two remaining trials are dropped. The lick error rule modifies lick labels but does not remove trials, and low-speed frames are deliberately retained.

ii.
```python
if e <= s or (e - s) < 2:
    ...
    continue
...
if valid_trial_count < 2:
    return None
```

iii. The agent justified retaining low-speed frames because speed itself is a decoder output. It did not document a reason for using two samples rather than the human solution’s 50-sample trial-quality threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural matrices come from the NWB `processing/ophys/Deconvolved/plane*/data` arrays. Raw `Fluorescence` and `Neuropil` are loaded only to compute dF/F for interneuron detection; `iscell` supplies the initial ROI mask.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
...
neural_data = deconv_cells[final_cell_mask]
```

iii. The notes claim the stored deconvolved arrays were precomputed by the reference pipeline and that the paper’s decoder uses deconvolved events, so reusing them was considered appropriate.

## 2-b. How is the `neural` data processed?

i. Deconvolved arrays from all planes are concatenated by ROI, cropped with behavior to the common minimum length, filtered by `iscell` and the computed interneuron mask, transposed to neuron-by-time, and trial-sliced. The agent does not recompute final events. It separately estimates per-trial dF/F as `F - 0.7*Fneu`, maximin baseline, normalized dF/F, and Gaussian smoothing solely for the speed-correlation filter.

ii.
```python
deconv_all = np.concatenate(deconv_list, axis=1)
deconv_cells = deconv_all[:, cell_mask].T
neural_data = deconv_cells[final_cell_mask]
```
```python
F_corr = F_trial - NEUROPIL_COEF * Fneu_trial
smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
dff = gaussian_filter1d((F_corr - baseline) / abs_baseline, sigma=2, axis=1)
```

iii. The agent believed NWB deconvolution could substitute for recomputing the paper-specific events. It documented the dF/F parameters as matching the paper, but used this reconstruction only for interneuron classification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs first pass `iscell[:,0]`. Putative interneurons are then removed when their per-session dF/F Pearson correlation with speed exceeds 0.5. Sessions with fewer than two remaining cells are dropped.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

iii. The notes identify both filters as paper/reference curation: Suite2p/manual `iscell` and the Methods’ speed-correlation definition of putative interneurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural matrix begins at the `trial_start` flag and ends immediately before the paired teleport sample, so column zero is the trial-start alignment point.

ii.
```python
s = trial_start_inds[i]
e = teleport_inds[i]
trial_neural = neural_data[:, s:e]
```

iii. The required event is trial start. The notes report checking boundary positions and conclude the NWB flags identify the exact desired frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One native imaging frame is one time bin; there is no temporal rebinning. Effective rate is scanner rate divided by the number of planes for multiplane sessions, and metadata uses the first retained session’s rate (about 64.48 ms/bin).

ii.
```python
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. The agent verified that single- and two-plane recordings reduce to approximately 15.5 Hz per plane and chose native frames to preserve synchronized behavior and neural samples.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the number of samples since trial start and the effective imaging rate, rather than directly subtracting behavioral timestamps.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes describe all streams as synchronized at the imaging-frame rate and therefore regard frame count divided by rate as elapsed seconds.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based integer sequence of trial columns is divided by the session’s per-plane frame rate and cast to float32.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. This was chosen as the direct native-frame representation of elapsed time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created with exactly `e-s` elements, the same slice length and column ordering as `neural_data[:,s:e]`; scalar inputs are broadcast across those columns.

ii.
```python
n_t = e - s
trial_neural = neural_data[:, s:e]
input_arr = np.zeros((4, n_t), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. The agent relies on common frame indices after cropping neural and behavior to equal total lengths.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/environment/data` over the current trial.

ii.
```python
environment = bts['environment/data'][:]
trial_env = environment[s:e]
```

iii. The notes map this field directly to ENV1/ENV2 and report it as a per-trial contextual variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of nonnegative samples in a trial is taken (fallback 0), cast to float32, and broadcast over all trial columns.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
input_arr[1, :] = np.float32(env_val)
```

iii. The agent treats environment as constant within a trial, as required by the decoder specification.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although raw `trial number/data` is loaded, the converted value is the zero-based loop index obtained from ordered trial-start/teleport pairs.

ii.
```python
trial_num = bts['trial number/data'][:]
...
trial_number = np.float32(i)
```

iii. The mapping plan calls for a zero-indexed within-session number; the agent uses boundaries rather than trusting the raw trial-number values.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is cast to float32 and broadcast across the trial.

ii.
```python
input_arr[2, :] = trial_number
```

iii. It is intended as a per-trial scalar represented in the uniform `(variables,time)` array.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the separate `Reward/timestamps` stream and the previous trial’s first/last position timestamps.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
prev_outcome = np.float32(prev_trial_rewarded)
```

iii. The agent interprets any reward timestamp inside a trial window as a rewarded outcome and retains that result for the next trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned 0. After every raw trial (including a skipped malformed one), an inclusive timestamp-range test sets the state used by the following trial; that binary state is broadcast over all columns.

ii.
```python
prev_trial_rewarded = 0
...
return np.any((reward_timestamps >= trial_start_time) &
              (reward_timestamps <= trial_end_time))
...
input_arr[3, :] = prev_outcome
prev_trial_rewarded = int(was_rewarded)
```

iii. This implements omitted=0/rewarded=1 while preserving the temporal relationship even if a malformed trial is not emitted.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It combines raw `position/data` with reward-zone coordinates inferred from the NWB `identifier` scene string and a fixed switch trial (30), not from the raw `reward_zone` series (which is loaded but unused here).

ii.
```python
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say scene naming follows the repository’s `get_reward_zones` logic and that A/B/C are fixed at 80–130, 200–250, and 320–370 cm. The trajectory fixed its parser to cover cross-environment switch names.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus the near zone edge is used before the zone, zero inside the inclusive zone, and position minus the far edge after it; the resulting signed distance is discretized.

ii.
```python
dist[before] = position[before] - rz_start
dist[inside] = 0.0
dist[after] = position[after] - rz_end
```

iii. The agent follows the task’s requested signed distance to any point in the active reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit boolean masks produce seven categories with boundaries `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
out[dist < -50] = 0
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. The notes state these masks directly implement the decoder-task bins and isolate the in-zone zero class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data are sliced by the identical `[s:e)` frame range; one distance label is produced per neural column.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. The agent relies on the NWB’s synchronized frame indexing and prior common-length crop.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is directly derived from `position/data`.

ii.
```python
position = bts['position/data'][:]
trial_pos = position[s:e]
```

iii. The notes identify position as centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Trial position is digitized against six equally spaced edges from 0 to 450 cm and clipped to classes 0–4.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. Five 90-cm bins follow directly from the requested equal partition; clipping handles slight excursions outside track limits.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The effective edges are 0, 90, 180, 270, 360, and 450 cm, with out-of-range samples absorbed into the nearest end class.

ii.
```python
POSITION_BIN_EDGES = np.linspace(0, TRACK_LENGTH, POSITION_BINS + 1)
pos_binned = discretize_position(trial_pos)
```

iii. The agent explicitly ties this to five equal bins spanning the instructed 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural matrices use the same trial frame slice, so each output sample corresponds to one neural column.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. Alignment is by the common NWB frame index.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick_raw = bts['lick/data'][:]
trial_lick = lick[s:e].copy()
```

iii. The notes identify this as the native lick stream sampled with the imaging frames.

## 9-b. What processing is involved in computing `output` *Lick*?

i. If more than 35% of trial samples exceed 2, the entire trial’s lick vector is replaced by zeros. Otherwise values above 1 are capped, and finally every positive value becomes 1.

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The agent cites the repository’s lick-sensor error rule but substitutes zero for the paper code’s NaN because the decoder requires categorical outputs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data use the same `[s:e)` slice and have equal column counts.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
```

iii. The streams are assumed synchronized at the imaging-frame rate after common cropping.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the final component of the NWB `identifier` (scene name), trial order, fixed A/B/C coordinates, and switch trial 30. The loaded `reward_zone/data` is not used.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
```

iii. The agent says this mirrors the repository’s scene parsing and repaired the implementation after full-data validation exposed unrecognized cross-environment scenes.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene patterns select either one location for all trials or a before/after pair split at index 30. Labels A, B, C map to 0, 1, 2 and are broadcast over each trial.

ii.
```python
rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
rz_labels[:change_trial] = 'A'
rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
...
mapping = {'A': 0, 'B': 1, 'C': 2}
output_arr[4, :] = rz_loc
```

iii. The fixed switch point and coordinates were taken from the reference repository/paper; full-data checks found no remaining `-1` location values after parser correction.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`; `Reward/data` is loaded but not used.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
reward_data = bts['Reward/data'][:]
was_rewarded = detect_reward_in_trial(reward_timestamps, ...)
```

iii. The notes treat the presence of a reward delivery timestamp as the definitive rewarded/omitted outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. An inclusive interval test checks whether any reward timestamp lies between the first and last position timestamps of the trial. The result is converted to 0/1 and broadcast across the trial.

ii.
```python
return np.any((reward_timestamps >= trial_start_time) &
              (reward_timestamps <= trial_end_time))
...
output_arr[5, :] = int(was_rewarded)
```

iii. This implements the requested per-trial binary target without needing reward amplitude.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length mismatches are cropped to the shared minimum. Start/end list mismatches are silently truncated to the smaller count. Degenerate trials and sessions with fewer than two valid trials/cells are skipped. Tiny baselines are floored, NaN speed/dF/F samples are excluded from correlation, missing valid environment samples default to zero, unknown scene names default to zone A, and bad lick trials become all-zero lick labels.

ii.
```python
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
n_trials = min(len(trial_start_inds), len(teleport_inds))
abs_baseline[abs_baseline < 1e-10] = 1e-10
...
print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
```

iii. The agent frames these as defensive conversion choices and verified final format/statistics, though several fallbacks silently invent or discard data rather than failing loudly.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive work is loading large NWB fluorescence/neuropil/deconvolved arrays for all 152 sessions, per-trial maximin dF/F filtering over all selected cells, full-session interneuron correlations, and serializing the roughly 9.4-GB pickle. The notes report optimizing correlation reduced two-session runtime from 18.2 to 7.1 seconds and estimate/full-run conversion around minutes.

ii.
```python
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(...)
```

iii. The notes explicitly identify dF/F/interneuron work and large data I/O as performance concerns and document vectorizing Pearson correlation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer per-session loop is serial; plane loading/concatenation and especially the per-trial dF/F loop could potentially be batched or processed with boundary-aware vectorization. Trial output construction is also serial, though variable lengths make list construction natural. Interneuron correlation itself was already vectorized across cells.

ii.
```python
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
...
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(...)
```

iii. The trajectory reports a deliberate vectorized correlation optimization; it leaves variable-length trial extraction as a loop for clarity and compatibility with ragged output.

## 13-c. What processing does the code repeat multiple times?

i. Every trial repeats maximin dF/F filtering and later a second trial loop repeats boundary slicing for emitted arrays. Reward interval detection is performed once for malformed trials or normal trials and the result serves both current output and next input. Bin edges are recreated on every `discretize_position` call. All three neural representations (deconvolved, F, Fneu) are read even though only deconvolved data is emitted.

ii.
```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(...)
...
for i in range(n_trials):
    trial_neural = neural_data[:, s:e]
...
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
```

iii. The notes accept the dF/F pass as necessary for interneuron detection and the later pass as necessary to build ragged decoder trials; they do not discuss the repeatedly recreated position edges.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused `trial_num`, `scanning`, `rzone_cumul`, `reward_data`, and `plane_idx`; creates unused `plane_assignment`, `input_data`, and `trial_scalars`; computes several unused counts; and computes dF/F only to produce a cell mask while emitting stored deconvolved data. The module also imports unused `pearsonr`. Optional plotting consumes work only when requested.

ii.
```python
rzone_cumul = bts['reward_zone/data'][:]
trial_num = bts['trial number/data'][:]
scanning = bts['scanning/data'][:]
plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]
input_data = np.array([time_from_start], dtype=np.float32)
trial_scalars = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
```

iii. The documentation does not acknowledge most of this dead work. Some fields were evidently retained from exploration or intended robustness checks but never influence the saved dataset.
