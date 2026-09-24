# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-sorts every matching NWB file under `data/sub-*`, opens each directly with `h5py`, reads behavioral and ophys arrays, and processes one file as one session. Full mode is the default; sample mode selects two files.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, ...)
f = h5py.File(nwb_path, 'r')
```

iii. The notes say the directory survey found 11 subjects, 152 sessions, and 12,216 trials, and full conversion reproduced those counts. The agent chose `h5py` because it could access the NWB datasets directly.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each file's NWB subject metadata. After conversion, unique IDs are naturally sorted and every session receives the corresponding integer `subject_idx`.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
```

iii. The notes report that this yielded the expected 11 mice and that the data directories follow `sub-{id}` naming.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; its session ID is read from metadata and its trial lists are appended once to the top-level session lists.

ii.
```python
sess_id = f['general']['session_id'][()].decode()
all_neural.append(result['neural_trials'])
all_input.append(result['input_trials'])
all_output.append(result['output_trials'])
```

iii. The agent documented 152 NWB files and 152 converted sessions, consistent with the file organization.

## 1-d. How are the data split into trials?

i. Trial starts are every positive `trial_start` sample and trial ends are every positive `teleport` sample. Counts are truncated to the smaller count if unequal. Each trial uses the half-open slice `[start:end)`.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
...
trial_neural = neural_all[:, t_start:t_end]
```

iii. The notes identify `trial_start_inds` and `teleport_inds` as the reference-code boundaries and report matching the dataset's trial count.

## 1-e. How are trials filtered based on quality controls?

i. Only trials shorter than two samples are dropped. Lick-sensor failures are not dropped; their lick samples become NaN and later their converted lick output becomes zero. Sessions with fewer than two retained trials are dropped.

ii.
```python
if n_tp < 2:
    continue
...
if len(result['neural_trials']) < 2:
    continue
```

iii. The agent says all timepoints were retained for decoding and describes lick error correction rather than trial exclusion. It reports 69 affected trials (0.56%).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from each plane in the NWB `processing/ophys/Deconvolved` group; multiple planes are concatenated across ROIs. The agent does not use raw `Fluorescence` or `Neuropil`.

ii.
```python
deconv_group = f['processing']['ophys']['Deconvolved']
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_data = np.concatenate(plane_data, axis=1)
```

iii. The notes assert that the NWB deconvolved events are the same as `sess.timeseries['events']` and therefore that recomputing dF/F is unnecessary.

## 2-b. How is the `neural` data processed?

i. Plane arrays are concatenated, filtered by `iscell`, transposed to neuron-by-time, converted per trial to `float32`, and otherwise left unchanged. No baseline correction, smoothing, or deconvolution is performed by the conversion.

ii.
```python
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The agent believed `Deconvolved` already represented the paper's OASIS-derived event signal. It also states that pooling planes matches the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose first `iscell` column equals one are retained. The code does not implement the paper's speed-correlation interneuron exclusion.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. The notes explicitly say interneuron filtering was skipped because the expected effect was small (~0.42%), while `iscell` includes manual curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced at the same frame indices from `trial_start` through just before `teleport`; the first retained column is treated as time zero.

ii.
```python
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The notes call the alignment “per-trial, from trial_start to teleport” and report exact neural spot checks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The agent estimates each session's bin width from the median difference of position timestamps and stores the across-session median, about 64.48 ms (15.51 Hz), in metadata.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
median_dt = np.median(all_dts)
'time_bin_size': median_dt * 1000
```

iii. The notes choose native frame rate, citing the paper's ~64.5 ms imaging frame and matching full-data statistics.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived indirectly from the `position` timestamps: their median spacing supplies `dt`, while the within-trial frame index supplies elapsed bins.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
dt = np.median(np.diff(pos_timestamps))
time_from_start = np.arange(n_tp) * dt
```

iii. The notes map “frame index × dt” to this input and report exact sample spot checks.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based integer sequence of trial samples is multiplied by the session's median timestamp interval.

ii.
```python
time_from_start = np.arange(n_tp) * dt
input_data[0, :] = time_from_start
```

iii. The agent used this as a simple native-rate elapsed-time representation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The sequence has exactly the same `n_tp = t_end - t_start` length as the neural slice, with sample zero aligned to the first neural sample.

ii.
```python
n_tp = t_end - t_start
trial_neural = neural_all[:, t_start:t_end]
input_data = np.zeros((4, n_tp), dtype=np.float32)
```

iii. The agent relies on shared frame indexing and reports exact trial spot checks; it performs no explicit neural/behavior timestamp validation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` data stream.

ii.
```python
env_data = beh['environment']['data'][:]
env_vals = env_data[t_start:t_end]
```

iii. The notes identify environment as a binary 0/1 variable and report ENV1/ENV2/mixed-session counts.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are removed, the median remaining value is converted to an integer, and that single value is repeated across the trial; an all-invalid trial defaults to zero.

ii.
```python
env_vals = env_vals[env_vals >= 0]
if len(env_vals) > 0:
    trial_env[t] = int(np.median(env_vals))
input_data[1, :] = float(trial_env[t])
```

iii. The agent treated environment as a per-trial binary decoder input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index over detected trial boundaries, not the NWB `trial number` stream.

ii.
```python
for t in range(n_trials):
    trial_num = float(t)
```

iii. The notes explicitly map trial index to a 0-indexed per-trial value.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The index is cast to float and repeated at every timepoint of that trial.

ii.
```python
input_data[2, :] = trial_num
```

iii. This follows the agent's interpretation of a continuous, per-trial input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` event timestamps mapped into the position timestamp index, then summarized as whether the preceding detected trial contains an event.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
...
prev_outcome = float(trial_rewarded[t - 1])
```

iii. The notes map previous trial reward to binary 0/1 and report spot checks against reward events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are insertion-mapped and clipped to the recording bounds. A trial is rewarded if any mapped index falls in `[start,end)`. The first trial is assigned zero; later trials copy the preceding trial's binary outcome across time.

ii.
```python
trial_rewarded[t] = int(np.any((reward_frame_inds >= t_start) &
                               (reward_frame_inds < t_end)))
prev_outcome = 0.0 if t == 0 else float(trial_rewarded[t - 1])
input_data[3, :] = prev_outcome
```

iii. The agent documents exact spot checks and uses zero for the undefined first-trial history.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral `position` plus a reward-zone label inferred from samples where behavioral `reward_zone > 0`. The minimum active position is tolerance-matched to fixed paper zones; missing labels inherit the nearest labeled trial (searching backward before forward at equal distance), with A as fallback.

ii.
```python
rz_active = trial_rzone > 0
rz_start_pos = trial_pos[rz_active].min()
rz_labels[t] = determine_reward_zone_label(rz_start_pos)
...
label = get_reward_zone_for_trial(rz_labels, t)
trial_rz_start[t] = REWARD_ZONES[label][0]
```

iii. The notes say zone inference from position was verified, and omission trials use neighboring trials because their zone stream may be absent.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The fixed zone is treated as `[start,start+50]`. Distance is position minus the start before it, zero inside it, and position minus the end after it; the result is then discretized.

ii.
```python
distance[before_mask] = position[before_mask] - rz_start
distance[in_mask] = 0.0
distance[after_mask] = position[after_mask] - rz_end
dist_bins = discretize_distance(distance)
```

iii. The agent says this matches the paper's position relative to reward-zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks create seven instruction-specified bins, with -50 in class 1, -10 in class 2, exactly zero in class 3, +10 in class 4, and +50 in class 5.

ii.
```python
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
```

iii. The docstring reproduces the requested seven category meanings.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same start/end indices and length as neural data; distance categories therefore align sample-for-sample.

ii.
```python
trial_neural = neural_all[:, t_start:t_end]
trial_pos = position[t_start:t_end]
output_data[0, :] = dist_bins
```

iii. The notes report an exact distance-output spot check.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the behavioral `position` data stream.

ii.
```python
position = beh['position']['data'][:]
trial_pos = position[t_start:t_end]
```

iii. The notes map position directly to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Each per-trial position sample is digitized against five equal 90-cm divisions of the 450-cm track, then clipped to classes 0–4.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The agent cites the 450-cm track and requested equal bins, and reports exact position-bin spot checks.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are 90, 180, 270, and 360 cm. `np.digitize` puts equality at each threshold in the higher class; out-of-range values are absorbed into the first or last class by clipping.

ii.
```python
bin_edges = np.linspace(0, 450.0, 6)
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, 4)
```

iii. This directly implements five equal-sized bins spanning 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity are sliced with identical frame boundaries and placed in arrays with identical `n_tp`.

ii.
```python
trial_neural = neural_all[:, t_start:t_end]
trial_pos = position[t_start:t_end]
output_data[1, :] = pos_bins
```

iii. The agent relies on the NWB streams' common indexing and reports exact spot checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavioral `lick` data stream.

ii.
```python
lick_data = beh['lick']['data'][:]
trial_licks = licks_processed[t_start:t_end]
```

iii. The notes identify this stream as cumulative lick counts and cite reference lick-error handling.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Per trial, if over 35% of values exceed 2, all lick values become NaN. Other values above one are clipped to one. Each retained trial is Gaussian-smoothed with sigma two, thresholded at 0.5, and NaNs become output zero.

ii.
```python
if frac_high > LICK_ERROR_THRESHOLD:
    licks[t_start:t_end] = np.nan
licks[licks > 1] = 1
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
lick_binary[np.isnan(smoothed_licks)] = 0
```

iii. The notes claim clipping, sigma-two smoothing, and the 0.35 error rule follow the paper repository; the decoder requirement motivates a binary final output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks are processed in full-session coordinates, then sliced with the same `[start:end)` indices as neural data. Smoothing is applied separately within each trial.

ii.
```python
trial_licks = licks_processed[t_start:t_end]
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
output_data[3, :] = lick_binary
```

iii. Shared frame slicing is the agent's alignment method.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred jointly from behavioral `reward_zone` activity and `position`, using the minimum position where the zone signal is active.

ii.
```python
trial_rzone = rzone[t_start:t_end]
trial_pos = pos[t_start:t_end]
rz_start_pos = trial_pos[trial_rzone > 0].min()
```

iii. The agent says the inferred labels were manually spot-checked and matched expected A/B/C positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. An observed start within 20 cm beyond either fixed-zone boundary is labeled A/B/C. Missing trials copy the nearest available label (backward wins ties), default to A if no label exists, map A/B/C to 0/1/2, and repeat the class across time.

ii.
```python
if start - 20 < rz_start_pos < end + 20:
    return label
...
label = get_reward_zone_for_trial(rz_labels, t)
trial_rz_label[t] = {'A': 0, 'B': 1, 'C': 2}.get(label, 0)
output_data[4, :] = rz_loc
```

iii. The notes justify neighbor inference for omission trials and report reward-zone label spot checks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the behavioral `Reward` event timestamps and position timestamps/trial boundaries.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
pos_timestamps = beh['position']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. The notes describe reward delivery as the binary trial outcome and report an 84.7% reward rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are converted to insertion indices and clipped. A trial is one if any mapped index is within `[start,end)`, otherwise zero; the value is repeated over all its samples.

ii.
```python
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
reward_in_trial = np.any((reward_frame_inds >= t_start) &
                         (reward_frame_inds < t_end))
output_data[5, :] = trial_rewarded[t]
```

iii. The agent reports exact outcome spot checks and matching aggregate reward rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Unequal start/end counts are silently paired after warning and truncation; trials under two frames are skipped; missing zone labels inherit a neighbor or A; all-invalid environment defaults to 0; reward indices are clipped; lick-error NaNs become no-lick; and sessions below two trials are skipped. There is no general neural/behavior length reconciliation or timestamp consistency assertion.

ii.
```python
n_trials = min(n_trials, len(teleport_inds))
...
return 'A'  # fallback
...
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
...
lick_binary[np.isnan(smoothed_licks)] = 0
```

iii. The notes emphasize successful full validation and edge-case checks, but do not justify the silent default classes beyond supporting complete decoder arrays.

## 13-a. What are the most time-consuming steps of the code?

i. The code times each session and the full conversion. Its dominant work is loading large `Deconvolved` plane arrays and behavioral arrays from 152 NWB/HDF5 files, concatenating/filtering them, materializing per-trial copies, and finally pickling about 9.4 GB. Optional plotting and decoder training are additional expensive operations, but not part of ordinary conversion.

ii.
```python
deconv_data = deconv_group[plane_keys[0]]['data'][:]
...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
...
pickle.dump(data, f, protocol=4)
```

iii. The notes estimate ~0.46 s per session and ~70 s total conversion; the full output size explains substantial serialization and memory cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops over trials separately compute rewards, environments, zone labels, zone starts, lick-error flags, and final trial arrays. The first four summary loops could be combined or partly vectorized with interval reductions; repeated nearest-label searches can be replaced by forward/backward fill. The final loop is reasonable because trials have variable lengths.

ii.
```python
for t in range(n_trials):  # rewards
    ...
for t in range(n_trials):  # environments
    ...
for t in range(n_trials):  # zone labels
    ...
for t in range(n_trials):  # output construction
    ...
```

iii. The agent did not discuss vectorization in its notes; it prioritized transparent per-trial processing and measured acceptable runtime.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trials repeatedly for lick QC, reward outcomes, environments, zone labeling, zone starts, final construction, and optional plotting. `get_reward_zone_for_trial` repeats an outward search twice per trial—once for class and once for start—and behavioral slicing is repeated in several passes.

ii.
```python
label = get_reward_zone_for_trial(rz_labels, t)  # label loop
...
label = get_reward_zone_for_trial(rz_labels, t)  # start loop
```

iii. The notes do not justify this repetition; the implementation favors simple separate stages.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads unused `scanning`, stores `rz_starts` only for optional plots, computes several metadata/print summaries, and in plot mode computes figures and distributions not consumed by the dataset. It also calculates `in_mask` before assigning zeros to an array already initialized to zero.

ii.
```python
scanning = beh['scanning']['data'][:]
...
in_mask = (position >= rz_start) & (position <= rz_end)
distance[in_mask] = 0.0
```

iii. The optional diagnostics were intended for sanity checking; `scanning` has no documented downstream use.
