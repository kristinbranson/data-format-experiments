# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script lists every `sub-*` directory, globs every NWB file in each, parses the session number, and reads each file directly with `h5py`. Full mode processes all 152 discovered sessions; sample mode selects two.

ii.
```python
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
with h5py.File(nwb_path, 'r') as f:
    F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
```

iii. The notes say the directory contains 11 subjects and 152 sessions and that direct NWB loading supplies raw Suite2p and aligned behavioral data. The full run recovered those counts.

## 1-b. How are the data split into subjects?

i. A subject is a `sub-<id>` directory; the prefix is removed and each retained session receives the index of that ID in encounter order.

ii.
```python
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
if subj not in subjects_seen: subjects_seen.append(subj)
subject_idx_list.append(subjects_seen.index(subj))
```

iii. The notes report the expected 11 mice and filename convention `sub-{id}_ses-{num}_behavior+ophys.nwb`.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; `ses-<number>` is parsed from its filename and each successfully processed file becomes one outer-list element.

ii.
```python
ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
all_sessions.append((subj, int(ses_str), nwb_file))
neural_all.append(result['neural'])
```

iii. The notes identify 152 NWBs/sessions and say the final output contains all 152.

## 1-d. How are the data split into trials?

i. Starts are samples where `trial_start == 1`; ends are samples where `teleport == 1`. Counts are silently reduced to the smaller count, paired in order, and slices exclude the teleport sample.

ii.
```python
tstart_idx = np.where(trial_start_arr == 1)[0]
teleport_idx = np.where(teleport_arr == 1)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))
start, stop = tstart_idx[t], teleport_idx[t]
neural = dff_valid[:, start:stop]
```

iii. The notes state that trial-start-to-teleport boundaries match the paper/reference pipeline.

## 1-e. How are trials filtered based on quality controls?

i. There is no substantive trial QC: only trials shorter than two samples are skipped. A whole session is skipped if fewer than two boundaries, fewer than two retained trials, or fewer than two valid cells exist. No erroneous-lick or 50-sample trial filter is applied.

ii.
```python
if n_trials < 2: return None
if n_tp < 2: continue
if len(neural_trials) < 2: return None
```

iii. The notes claim simply “Use trial_start to teleport boundaries” and explicitly retain all within-trial speeds; they do not justify the two-sample cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It derives neural data from raw `Fluorescence` (F), `Neuropil` (Fneu), and `iscell`, pooling planes; the stored `Deconvolved` series is deliberately unused.

ii.
```python
F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
Fneu_planes.append(f['processing']['ophys']['Neuropil'][pk]['data'][()].T)
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
```

iii. The notes correctly distinguish the NWB Suite2p deconvolution from the paper's custom pipeline.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated. The code subtracts `0.7*Fneu`, adds back trial mean neuropil, computes a per-trial maximin baseline (Gaussian 15, minimum 300, maximum 300), forms dF/F, smooths it with Gaussian sigma 2, and saves that dF/F. It does **not** perform the paper/reference OASIS deconvolution and does not implement session-dependent teleport-spanning baselines.

ii.
```python
f_ -= neu_coef * f_neu_
flow[:, start:stop] = minimum_filter1d(nansmooth(trial_data, 15, axis=1), 300, axis=-1)
flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], 300, axis=-1)
dff[valid] = (f_[valid] - flow[valid]) / np.abs(flow[valid])
dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
```

iii. The notes say dF/F follows the paper but explicitly decide “dF/F (not deconvolved),” despite also acknowledging that the paper applies optional/custom OASIS. Decoder results were used as a sanity check.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps `iscell[:,0] == 1`, then excludes cells whose Pearson correlation between dF/F and speed exceeds 0.5; sessions with fewer than two remaining cells are dropped.

ii.
```python
cell_mask = iscell[:, 0] == 1
is_interneuron = detect_interneurons(dff_cells, speed, threshold=0.5)
valid_cell_indices = np.where(cell_mask)[0][~is_interneuron]
```

iii. The notes cite manual `iscell` curation and the Methods' `corr(dF/F,speed)>0.5` putative-interneuron rule; the observed exclusion rate matched the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial arrays begin exactly at the detected trial-start index; no interpolation or extra offset is used.

ii.
```python
start = tstart_idx[t]
neural = dff_valid[:, start:stop].astype(np.float32)
```

iii. The notes call trial start the boundary/alignment event and validate trial lengths against raw boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Native behavior/imaging samples are retained at about 15.5 Hz (~64.5 ms), although metadata hard-codes 64.5 ms rather than deriving a global checked value.

ii.
```python
dt = np.mean(np.diff(timestamps))
'time_bin_size': 64.5,
'frame_rate_hz': 15.5,
```

iii. The notes report ~15.51 Hz and state that aligned frame samples are used directly.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses the raw `position` timestamps plus `trial_start` indices.

ii.
```python
timestamps = beh['position']['timestamps'][()]
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. The mapping plan identifies timestamps as the source and a raw-value sanity check was recorded.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from every timestamp in that trial.

ii. `time_from_start = timestamps[start:stop] - timestamps[start]`

iii. The notes describe exactly this transformation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with the same `[start:stop]` indices and therefore has the same number of columns; neural and behavior streams are first truncated to their common minimum length.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
neural = dff_valid[:, start:stop]
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. The notes describe NWB behavior as frame-aligned and validate a sample's timestamp and neural length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the raw behavioral `environment` time series.

ii. `environment = beh['environment']['data'][()]`

iii. The notes map this field to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, negative values are discarded and the integer median of remaining samples is broadcast across the trial; an empty trial defaults to 0.

ii.
```python
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]
if len(env_vals) > 0: env_per_trial[t] = int(np.median(env_vals))
inp[1, :] = env_per_trial[t]
```

iii. The notes describe environment as binary and per-trial, but do not specifically justify median aggregation/defaulting.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although raw `trial number` is loaded, the output is derived from the zero-based loop counter over detected trial boundaries.

ii.
```python
trial_number = beh['trial number']['data'][()]
for t in range(n_trials):
    inp[2, :] = t
```

iii. The notes call the desired value a 0-indexed trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is broadcast unchanged over every time point in a trial.

ii. `inp[2, :] = t`

iii. The mapping plan specifies a continuous, per-trial, zero-indexed value.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives outcome from `Reward.timestamps`, position timestamps, and each trial's start/teleport time interval.

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
```

iii. The notes identify reward timestamps as the source and report an omission rate matching the paper.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial is rewarded if any event timestamp lies in its inclusive interval. Trial 0 gets 0; later trials receive the prior trial's Boolean, broadcast over time.

ii.
```python
prev_outcome = float(rewarded_trials[t-1]) if t > 0 else 0.0
inp[3, :] = prev_outcome
```

iii. The notes follow the requested omitted=0/rewarded=1 definition.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and reward-zone coordinates inferred from a hard-coded subject/session scene table, not the raw `reward_zone` samples (which are loaded but unused).

ii.
```python
scene = SESSIONS_SCENES[subject_id][ses_num]
coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
trial_pos = position[start:stop]
```

iii. The notes say this mirrors `sessions_dict.py` and `get_reward_zones()`, with switches after 30 trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the zone is measured relative to its start, position after it relative to its end, and samples inside receive exactly zero.

ii.
```python
dist_to_rz = np.where(trial_pos < rz_start_cm, trial_pos - rz_start_cm,
    np.where(trial_pos > rz_end_cm, trial_pos - rz_end_cm, 0.0))
```

iii. The notes use the paper's A/B/C coordinates and describe the requested signed distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement seven requested categories, including a separate exact-zero category and inclusive upper bounds at +10/+50.

ii.
```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[distance == 0] = 3
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The notes state these bins reproduce the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[start:stop]` frame slice; the categorized result therefore has one value per neural column.

ii. `trial_pos = position[start:stop]`

iii. The notes treat behavior as aligned to imaging frames and validate a sample mapping.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavioral `position.data`.

ii. `position = beh['position']['data'][()]`

iii. The notes report the 0–451 cm observed range for the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is passed directly to categorical thresholding; no smoothing or normalization is applied.

ii. `pos_bins = discretize_position(trial_pos)`

iii. The notes map raw position to five 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Explicit masks create `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360` bins.

ii.
```python
bins[position < 90] = 0
bins[(position >= 90) & (position < 180)] = 1
bins[position >= 360] = 4
```

iii. The notes justify equal 90 cm divisions of the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The identical trial start/end frame indices are used.

ii. `trial_pos = position[start:stop]`

iii. The notes record a raw-position-to-bin spot check.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from behavioral `lick.data`.

ii. `lick = beh['lick']['data'][()]`

iii. The notes identify the NWB lick stream as the source.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Each sample greater than zero is converted to 1; all others become 0.

ii. `lick_binary = (trial_lick > 0).astype(np.int64)`

iii. The notes explain that the raw per-frame value may be cumulative/count-like and the target is binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick stream is truncated with behavior if needed and sliced with the same trial indices as neural data.

ii. `trial_lick = lick[start:stop]`

iii. The notes state behavior is aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from subject ID and session number through the hard-coded `SESSIONS_SCENES` metadata and scene-name parsing. Raw `reward_zone` is not used.

ii. `scene = SESSIONS_SCENES[subject_id][ses_num]`

iii. The notes say the mapping was transcribed from reference `sessions_dict.py`/`behavior.py`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed scenes map to A/B/C. Switch scenes use the pre-zone for trials 0–29 and post-zone from trial 30, then A/B/C maps to 0/1/2 and is broadcast.

ii.
```python
if trial_idx < change_trial: return REWARD_ZONE_DICT[pre_zone], pre_zone
return REWARD_ZONE_DICT[post_zone], post_zone
rz_label_int = {'A': 0, 'B': 1, 'C': 2}[rz_labels[t]]
out[4, :] = rz_label_int
```

iii. The notes cite the paper's “switch after 30 trials” and the reference zone coordinates.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward.timestamps` relative to `position.timestamps` and trial boundaries.

ii. `reward_ts = beh['Reward']['timestamps'][()]`

iii. The notes map reward timestamps to the binary outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp falls inclusively between its start and teleport timestamps, else 0; the scalar is broadcast across the trial.

ii.
```python
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
out[5, :] = int(rewarded_trials[t])
```

iii. The matching ~15.3% omission rate is cited as validation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length mismatches are truncated to the common minimum; starts/ends are truncated to the smaller count; NaN neural values become zero; negative environment samples are ignored; sessions/trials with too little usable data are skipped. Missing environment defaults to 0. No warning/assertion checks timestamp correspondence across behavior variables.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
n_trials = min(len(tstart_idx), len(teleport_idx))
neural = np.nan_to_num(neural, nan=0.0)
env_vals = env_vals[env_vals >= 0]
```

iii. The notes explicitly justify truncating observed off-by-one mismatches and call other long/short sessions real data, but do not justify zero-imputation or silently pairing mismatched boundaries.

## 13-a. What are the most time-consuming steps of the code?

i. Reading full NWB arrays, per-trial maximin filtering for every cell, interneuron correlations, holding/saving the ~9.3 GB pickle, and optional plotting dominate. The notes report dF/F taking roughly 0.8–8.4 seconds/session and ~800 seconds overall.

ii.
```python
F_planes.append(...['data'][()].T)
dff = compute_dff(...)
for i in range(n_cells):
    ... np.dot(...)
pickle.dump(data, f, protocol=4)
```

iii. Runtime measurements in the notes identify dF/F/full conversion as the main cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell correlation loop could use batched centered dot products; reward outcome intervals, environment aggregation, reward-zone construction, and portions of trial assembly could be vectorized. Trial-dependent filtering/slicing still naturally needs some iteration.

ii.
```python
for i in range(n_cells): ...
for t in range(n_trials): rewarded_trials[t] = ...
for t in range(n_trials): env_vals = environment[start:stop]
for t in range(n_trials): ... neural_trials.append(neural)
```

iii. The agent calls interneuron detection “vectorized implementation for speed,” but it only vectorizes arithmetic within each cell and supplies no efficiency analysis in its notes.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trials repeatedly: once inside each of two dF/F loops, then for reward outcomes, reward zones, environment, and final assembly. It also copies/slices arrays repeatedly and duplicates `out_tv` into `out` even though `out_tv` is temporary.

ii.
```python
for i in range(len(trial_start_idx)):  # baseline
for i in range(len(trial_start_idx)):  # smoothing
for t in range(n_trials):              # reward, zone, environment, assembly
```

iii. The notes do not discuss repeated processing; they emphasize correctness checks and total runtime.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `trial_number`, `scanning`, `autoreward`, and raw `reward_zone` but never uses them for output; computes `dt` but does not use it; constructs `out_trial` and `out_tv` only to copy/discard them; and in plotting computes an unused `pos`. Optional plots are diagnostic rather than decoder inputs.

ii.
```python
trial_number = beh['trial number']['data'][()]
scanning = beh['scanning']['data'][()]
dt = np.mean(np.diff(timestamps))
out_trial = np.array([rz_label_int, reward_outcome], dtype=np.int64)
pos = np.argmax(...)
```

iii. The notes say plotting supports sanity checks but do not acknowledge these unused variables/intermediates.
