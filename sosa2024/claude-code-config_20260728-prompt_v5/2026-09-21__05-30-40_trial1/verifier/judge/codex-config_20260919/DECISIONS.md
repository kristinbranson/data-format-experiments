# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `sub-*` directory, collects every NWB file beneath it, and processes each file as one session. It reads NWB datasets directly with `h5py` rather than through `pynwb`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
for subj in subjects:
    nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*.nwb')))
    all_nwb_files.extend(nwb_files)
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes report 11 subjects, 152 sessions, and 12,216 converted trials, and state that the directory structure and counts agree with the paper apart from one invalid trial.

## 1-b. How are the data split into subjects?

i. Subject identity comes from the parent `sub-*` directory. Unique IDs are sorted, and every retained session receives an index into that list.

ii.
```python
subject = os.path.basename(os.path.dirname(nwb_path))
...
unique_subjects = sorted(set(si['subject'] for si in session_infos))
subject_idx = np.array([unique_subjects.index(si['subject']) for si in session_infos])
```

iii. The notes justify this with the 11 subject directories matching the 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; files are sorted within subject and appended as separate entries in `neural`, `input`, and `output`.

ii.
```python
for idx, nwb_path in enumerate(all_nwb_files):
    neural_trials, input_trials, output_trials, session_info = process_session(nwb_path, ...)
    neural_all.append(neural_trials)
```

iii. The notes say every subject has 12–14 files and the full result has the expected 152 sessions.

## 1-d. How are the data split into trials?

i. Trial starts are all positive samples of `trial_start`; trial ends are all positive samples of `teleport`. The arrays are paired in order and sliced start-inclusive/end-exclusive after conversion through the code's one-based convention.

ii.
```python
trial_start_inds = np.where(trial_start_signal > 0)[0] + 1
teleport_inds = np.where(teleport_signal > 0)[0] + 1
...
s = trial_start_inds[i] - 1
e = teleport_inds[i] - 1
trial_neural = events_cells[:, s:e].copy()
```

iii. The notes identify a trial as `trial_start` through teleport and exclude teleport/ITI periods, following the paper.

## 1-e. How are trials filtered based on quality controls?

i. No minimum-duration quality filter is applied. Only nonpositive-length trials are skipped, and a whole session would be skipped if fewer than two trials remain. Start/end count mismatches are silently truncated to the shorter count.

ii.
```python
if len(trial_start_inds) != len(teleport_inds):
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
...
if e <= s:
    continue
```

iii. The notes say all valid trials were included and attribute the one-trial count discrepancy to an `e <= s` trial. They discuss lick sensor curation, but that changes lick labels rather than filtering trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values are derived from raw `Fluorescence` (F) and `Neuropil` (Fneu), pooled across imaging planes, rather than the NWB `Deconvolved` field.

ii.
```python
F_plane = ophys[f'Fluorescence/{plane}/data'][:]
Fneu_plane = ophys[f'Neuropil/{plane}/data'][:]
F_concat = np.concatenate(F_all, axis=1)
```

iii. The agent correctly noted that the stored deconvolution is Suite2p's raw result, whereas the paper recomputed dF/F and OASIS events.

## 2-b. How is the `neural` data processed?

i. F and Fneu are transposed; off-trial samples are made NaN; `0.7*Fneu` is subtracted; trial-mean neuropil is added back; a Gaussian-15, 300-sample minimum then maximum baseline is computed; dF/F is formed and Gaussian-smoothed with sigma 2; and OASIS deconvolution uses tau 0.7 and the per-plane rate. Unlike the reference, every session restricts baseline windows to laps and never uses session-specific `keep_teleports` processing.

ii.
```python
f_ -= NEU_COEF * f_neu_
f_[:, s:e] += NEU_COEF * np.nanmean(f_neu_[:, s:e], axis=1, keepdims=True)
flow[:, s:e] = nansmooth(f_[:, s:e], 15, axis=1)
flow[:, s:e] = minimum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
flow[:, s:e] = maximum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
spks[:, s:e] = oasis(trial_dff, 2000, TAU, frame_rate / n_planes)
```

iii. The notes cite the paper's custom dF/F pipeline and parameters. They do not identify the paper code's session-dependent rule allowing some baseline windows to span teleports.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with Suite2p `iscell > 0` are retained. The script deliberately does not remove putative interneurons whose dF/F correlates with speed above 0.5.

ii.
```python
cell_mask_all.append(iscell_plane > 0)
...
cell_indices = np.where(cell_mask)[0]
events_cells = events[cell_indices, :]
```

iii. The notes justify omission as small (~0.42%), computationally inconvenient, and partly redundant with manual curation. They acknowledge the resulting neuron count exceeds the paper's range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays are sliced beginning at the `trial_start` index, so each trial's first column is the alignment event and no further shifting occurs.

ii.
```python
s = trial_start_inds[i] - 1
e = teleport_inds[i] - 1
trial_neural = events_cells[:, s:e].copy()
```

iii. The notes describe temporal alignment as start of trial and record `off_start = 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one output bin, about 64.48 ms (15.5078125 Hz); there is no rebinning or resampling. Multi-plane scanner rate is divided by plane count.

ii.
```python
effective_rate = imaging_rate / n_planes if n_planes > 1 else imaging_rate
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. The notes state behavior and neural signals are frame-aligned at about 15.5 Hz and select the native imaging frame to match the reference.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the within-trial frame index and effective imaging rate. Position timestamps are loaded but not used for this input.

ii.
```python
behav_timestamps = bts['position/timestamps'][:]
...
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. The notes justify this by the common constant imaging/behavior frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based integer frame sequence is divided by the effective per-plane rate.

ii.
```python
n_tp = e - s
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. The agent treats the sampling interval as constant and therefore equivalent to subtracting the first behavioral timestamp.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly `e-s` samples and is stacked beside variables broadcast to that length, matching the neural slice `[:, s:e]`.

ii.
```python
trial_neural = events_cells[:, s:e].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
inputs = np.vstack([input_tv, ...])
```

iii. The notes state behavior and neural data are already aligned to imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the NWB behavioral `environment/data` series.

ii.
```python
environment = bts['environment/data'][:]
```

iii. The notes report that within scanning the raw values are 0=ENV1 and 1=ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative samples are removed, the trial median is cast to an integer, and missing trials inherit the previous trial (or remain zero for trial 0). The scalar is broadcast across time.

ii.
```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    env_per_trial[i] = int(np.median(valid_env))
elif i > 0:
    env_per_trial[i] = env_per_trial[i - 1]
```

iii. The notes say environment is constant within a trial, so a per-trial binary value is appropriate.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although raw `trial number/data` is loaded, the converted value is the zero-based loop index of the detected trial.

ii.
```python
trial_num = bts['trial number/data'][:]
...
trial_number = np.float32(i)
```

iii. The mapping table says NWB trial number, but the actual implementation uses the loop index; both represent within-session zero-based trial order.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is cast to float32 and repeated at every time point.

ii.
```python
np.full((1, n_tp), trial_number, dtype=np.float32)
```

iii. No additional transformation is documented.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Current-trial reward labels are derived from `Reward/timestamps` relative to position timestamps, with `autoreward/data` also able to force a rewarded label. Previous outcome uses the preceding label.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
autoreward = bts['autoreward/data'][:]
...
prev_outcome = np.float32(rewarded[i - 1])
```

iii. The notes say Reward timestamps provide reward delivery. They add autoreward as a safeguard, although the reference survey found it all zero.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any reward timestamp lies between its start and end timestamps or any autoreward sample is positive. Trial 0 is assigned zero; later trials copy the prior binary label and broadcast it.

ii.
```python
in_trial = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
rewarded[i] = int(np.any(in_trial))
...
prev_outcome = np.float32(0 if i == 0 else rewarded[i - 1])
```

iii. This follows the requested omitted=0/rewarded=1 interpretation.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral `position/data` and a per-trial zone inferred from positions at which `reward_zone/data > 0`. The mean active position is assigned to the nearest fixed zone center; absent activity inherits the prior zone, with A as the initial default.

ii.
```python
rz_positions = trial_pos[trial_rz > 0]
rz_center = np.mean(rz_positions)
dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
zone = min(dists, key=dists.get)
```

iii. The notes cite fixed zones A=80–130, B=200–250, C=320–370 and report near-equal zone frequencies. They did not use the reference solution's across-trial Viterbi inference.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus the near boundary is negative before the zone, zero inside its inclusive boundaries, and position minus the far boundary is positive after it; the result is then categorized.

ii.
```python
dist[before] = position[before] - zone_start
dist[inside] = 0.0
dist[after] = position[after] - zone_end
```

iii. The notes state this is signed distance to the nearest point in the reward zone, matching the task concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit Boolean masks implement `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
```

iii. The agent says these masks implement the decoder specification. Its exact +10 boundary differs from the reference's `np.digitize` behavior but follows the explicit `>0 ... +10` wording.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural arrays use the same `s:e` trial slice, yielding the same number of columns.

ii.
```python
trial_neural = events_cells[:, s:e].copy()
trial_pos = position[s:e]
dist_binned = discretize_distance_to_rz(dist)
```

iii. The notes state behavioral series are aligned to imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavioral `position/data`.

ii.
```python
position = bts['position/data'][:]
trial_pos = position[s:e]
```

iii. The notes identify position as centimeters along the 450 cm VR track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw trial slice is passed directly to categorical binning; there is no smoothing or resampling.

ii.
```python
pos_binned = discretize_position(trial_pos)
```

iii. The notes say the track is 450 cm and five equal spatial bins are required.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Values are placed into `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360` cm.

ii.
```python
bins[position < 90] = 0
bins[(position >= 90) & (position < 180)] = 1
bins[(position >= 270) & (position < 360)] = 3
bins[position >= 360] = 4
```

iii. These are five 90 cm bins spanning the 450 cm corridor, as requested.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same start-inclusive/end-exclusive indices slice position and neural activity.

ii.
```python
trial_neural = events_cells[:, s:e].copy()
trial_pos = position[s:e]
```

iii. The notes rely on the NWB's frame alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from behavioral `lick/data`.

ii.
```python
lick = bts['lick/data'][:]
```

iii. The notes describe it as a cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. For any trial where more than 35% of frames have count greater than 2, the script replaces the entire trial's lick signal with zero. It then binarizes remaining samples with `>0`.

ii.
```python
if n_frames > 0 and np.sum(trial_lick > 2) / n_frames > LICK_ERROR_THR:
    lick_corrected[s:e] = 0
lick_binary = (lick_corrected > 0).astype(np.int64)
```

iii. The notes claim this matches reference lick-sensor correction. The source method flags such trials as bad (often NaN); the human conversion instead simply binarizes raw lick values, so replacing errors with “no lick” is a substantive choice.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Corrected lick and neural data are sliced with the same trial indices.

ii.
```python
trial_neural = events_cells[:, s:e].copy()
trial_lick = lick_binary[s:e]
```

iii. The notes state all behavior is aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses `reward_zone/data` together with `position/data` to infer which fixed zone contains the mean active reward-zone position.

ii.
```python
zone_labels, zone_coords = determine_reward_zone_per_trial(
    position, rz_signal, trial_start_inds, teleport_inds)
```

iii. The notes say direct reward-zone state values were less useful than locating where that signal is active along the track.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The nearest-center label is encoded A=0, B=1, C=2 and repeated over all trial time points. Missing active samples inherit the previous zone, or default to A on the first trial.

ii.
```python
rz_loc = {'A': 0, 'B': 1, 'C': 2}[zone_label]
np.full((1, n_tp), rz_loc, dtype=np.int64)
```

iii. The agent validated approximately one-third occupancy per zone, but did not adopt the human reference's Viterbi smoothing across trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, behavior timestamps, and additionally `autoreward/data`.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
behav_timestamps = bts['position/timestamps'][:]
autoreward = bts['autoreward/data'][:]
```

iii. The notes identify Reward events as deliveries and report the expected ~85% reward rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is one if any reward timestamp falls from its start timestamp through its end timestamp, or if autoreward is positive in its index slice; otherwise zero. The scalar is broadcast over time.

ii.
```python
in_trial = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
if np.any(in_trial):
    rewarded[i] = 1
...
np.full((1, n_tp), reward_out, dtype=np.int64)
```

iii. The notes use omission-rate agreement (15.3%) as a sanity check.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Unequal start/end counts are truncated; nonpositive trials are skipped; missing zone observations inherit the prior label/default A; missing environment inherits the prior/default 0; neural NaNs inside emitted trials become zero; shape consistency is asserted; and sessions below two trials are dropped. The code does not reconcile neural-versus-behavior length mismatches or validate timestamp equality.

ii.
```python
min_len = min(len(trial_start_inds), len(teleport_inds))
...
trial_neural[np.isnan(trial_neural)] = 0
...
if len(neural_trials) < 2:
    continue
```

iii. The notes mention one invalid trial, report no neural NaNs in the result, and describe inherited/default values as fallbacks. They do not discuss the risks of silently pairing truncated boundary arrays.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant explicit step is full-session custom dF/F and OASIS deconvolution over all ROIs; loading large NWB arrays and serializing the 9.84 GB pickle are also costly. The script times deconvolution and total sessions.

ii.
```python
t_dff = time.time()
events, dff = compute_dff_and_deconvolve(...)
print(f"    dF/F + deconv: {time.time() - t_dff:.1f}s")
```

iii. The notes report 984 seconds for full conversion and identify full conversion of all 152 large sessions as the expensive operation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Separate per-trial loops infer zones, rewards, autoreward, environment, lick errors, and finally construct trial arrays. Reward membership, environment summaries, lick-error counts, and much of zone classification could be computed using boundary reductions or one consolidated loop. Correlations are not computed because the interneuron filter was omitted.

ii.
```python
for i in range(n_trials):  # zone inference
...
for i in range(n_trials):  # autoreward
...
for i in range(n_trials):  # environment
...
for i in range(n_trials):  # output construction
```

iii. The agent provided no efficiency justification for these repeated passes; variable trial lengths explain the final splitting loop but not all separate summary loops.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries are converted and sliced repeatedly in dF/F, zone, reward, autoreward, environment, lick, build, and plotting code. dF/F performs separate baseline and smoothing/deconvolution passes over the same trial windows.

ii.
```python
s = trial_start_inds[i] - 1
e = teleport_inds[i] - 1
```

iii. The notes focus on scientific validation and do not discuss this repetition.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `trial_num` and `reward_data` but never uses them, constructs unused `input_pt`, and computes dF/F/OASIS for every ROI before applying `iscell`. It also retains full `dff` only for optional plots and creates extensive optional plot intermediates.

ii.
```python
trial_num = bts['trial number/data'][:]
reward_data = bts['Reward/data'][:]
input_pt = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
events, dff = compute_dff_and_deconvolve(F_concat, Fneu_concat, ...)
events_cells = events[cell_indices, :]
```

iii. No justification is given. Applying `iscell` before expensive neural processing would preserve the final dataset while reducing work and memory.
