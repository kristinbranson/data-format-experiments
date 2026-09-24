# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent lists every `sub-*` directory and every `.nwb` file below it, then reads each file directly with `h5py`. Unknown subjects, sessions absent from its hard-coded scene table, sessions that raise exceptions, and sessions with fewer than two valid trials are skipped.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
with h5py.File(filepath, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
```

iii. The trajectory says the agent explored the NWB layout, mapped `sub-mN` to `GCAMPN`, and concluded that the NWB files contained the required behavior, neural, ROI, and reward streams. It later cited 11 mice and 152 sessions as matching the paper.

## 1-b. How are the data split into subjects?

i. Each `sub-*` directory is treated as a subject. A fixed mapping translates it to the paper's GCAMP name for session metadata; the saved subject name removes `sub-` (for example, `m11`).

ii.
```python
for subj_dir_name in subjects:
    gcamp_name = SUBJECT_MAP[subj_id]
    mouse_name = subj_id.replace('sub-', '')
    subject_names.append(mouse_name)
```

iii. The agent explicitly chose `sub-mN -> GCAMPN` after examining the data and paper code.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The experimental day is parsed from `ses-NN` in its filename and used to look up hard-coded scene metadata.

ii.
```python
for sess_file in session_files:
    ses_part = sess_file.split('_')[1]
    exp_day = int(ses_part.split('-')[1])
    scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. The trajectory states that session number was interpreted as experimental day and scene metadata was needed to determine reward zones.

## 1-d. How are the data split into trials?

i. Trial starts are all positive samples in `trial_start`; trial ends are all positive samples in `teleport`. The two lists are truncated to their common minimum and each trial is sliced `[start:end)`.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
teleport_idx = np.where(nwb_data['teleport'] > 0)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))
...
trial_n = neural[start:end, :].T
```

iii. The agent said trials were marked by `trial_start` and `teleport` events and that streams were already frame-aligned.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped only when its end is not after its start or it has fewer than two samples. A session is dropped if fewer than two trials remain. Lick-sensor error trials are retained, with their licks replaced by zero.

ii.
```python
if end <= start:
    continue
if n_timepoints < 2:
    continue
...
if len(trial_neural) < 2:
    return None
```

iii. The trajectory does not justify the two-sample cutoff. It only notes the target requirement of at least two trials per session and describes lick-error correction as paper-derived.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Saved neural data comes from the NWB `processing/ophys/Deconvolved` arrays. `Fluorescence` is separately loaded only to identify putative interneurons; neuropil is not loaded.

ii.
```python
deconv_keys = list(ophys['Deconvolved'].keys())
deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
deconvolved = np.concatenate(deconv_list, axis=1)
```

iii. The agent reasoned that NWB deconvolved calcium activity corresponded to the OASIS-deconvolved signal in the paper.

## 2-b. How is the `neural` data processed?

i. Plane arrays are sorted and concatenated, curated ROIs and the inferred interneuron mask are applied, and trial slices are transposed to neuron-by-time `float32`. It does not perform neuropil subtraction, baseline estimation, dF/F smoothing, or OASIS deconvolution itself.

ii.
```python
deconvolved = np.concatenate(deconv_list, axis=1)
deconvolved = nwb_data['deconvolved'][:, cell_mask]
neural = deconvolved[:, neuron_mask]
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The agent justified this by treating the stored deconvolution as the paper's processed events and noted that planes should be pooled.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must have `iscell[:, 0] == 1`. Putative interneurons are then excluded when raw stored fluorescence has Pearson correlation greater than 0.5 with speed over samples where position is positive.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
valid_mask = nwb_data['position'] > 0
is_interneuron = identify_interneurons(fluorescence, nwb_data['speed'], valid_mask)
neuron_mask = ~is_interneuron
```

iii. The agent cited Suite2P curation and the Methods' `r > 0.5` speed-correlation exclusion, while acknowledging in code comments that fluorescence was being used as a dF/F proxy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is implicit: neural samples are sliced starting at the `trial_start` index and ending before `teleport`.

ii.
```python
start = tstart_idx[i]
end = teleport_idx[i]
trial_n = neural[start:end, :].T
```

iii. The agent stated that NWB behavior and imaging samples were already aligned and selected trial onset as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Trial samples are retained at the stored rate; time is computed using `1 / imaging_rate`, while metadata is fixed to `1000 / 15.5078125 = ~64.48 ms`.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
...
'time_bin_size': 1000.0 / 15.5078125
```

iii. The agent concluded the recordings were approximately 15.5 Hz and chose the native ~64.5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Although position timestamps are loaded, this input is actually synthesized from the trial sample count and imaging rate.

ii.
```python
timestamps = bts['position/timestamps'][:]
...
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The agent relied on its conclusion that behavioral and neural samples were frame-aligned at a uniform imaging rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based integer sample index is multiplied by the reciprocal imaging rate and cast to `float32`.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. No separate justification was recorded beyond using native frame timing and aligning to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created with exactly the number of samples in the same `[start:end)` neural slice, so sample zero corresponds to trial start.

ii.
```python
n_timepoints = end - start
trial_n = neural[start:end, :].T
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The trajectory says NWB streams were already aligned to imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The code loads the raw `environment` series but does not use it. Environment is inferred from the hard-coded scene string in `SESSIONS_INFO`.

ii.
```python
environment = bts['environment/data'][:]
...
env_before, env_after = parse_scene_environment(scene)
```

iii. The agent chose scene/session metadata to map environment and specifically noted that m17 and m18 began in ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene text is parsed as ENV1=0 and ENV2=1. For a scene containing both, the value switches after trial index 29; the per-trial value is repeated across time.

ii.
```python
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    env_per_trial[min(SWITCH_TRIAL, n_trials):] = env_after
input_arr[1, :] = env_per_trial[i]
```

iii. The agent used the paper statement that switch sessions change after 30 trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index, not the loaded NWB `trial number` time series.

ii.
```python
for i in range(n_trials):
    input_arr[2, :] = float(i)
```

iii. The trajectory identifies trial segmentation from start/teleport events; it does not separately explain rejecting the stored trial-number values.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is converted to float and repeated across every sample of the trial.

ii.
```python
input_arr[2, :] = float(i)
```

iii. The agent documented this as a 0-indexed within-session trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from the separate `Reward/timestamps` array and position timestamps used as trial boundary times.

ii.
```python
reward_ts = bts['Reward/timestamps'][:]
start_t = timestamps[tstart_idx[i]]
end_t = timestamps[teleport_idx[i]]
```

iii. The agent noted that rewards had separate timestamps and decided a trial was rewarded if any reward timestamp fell inside it.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current-trial reward flags are computed by interval tests; the array is shifted by one trial, with the first value set to zero, and repeated in time.

ii.
```python
reward_per_trial[i] = int(np.any((reward_ts >= start_t) & (reward_ts <= end_t)))
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. This directly implements omitted=0, rewarded=1, with no previous outcome for the first trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone labels inferred from the hard-coded session scene and a fixed trial-30 switch. The loaded `reward_zone` series is unused.

ii.
```python
rz_labels = parse_scene_reward_zones(scene, n_trials)
pos = nwb_data['position'][start:end]
rz_start, rz_end = REWARD_ZONES[rz_labels[i]]
```

iii. The agent considered inferring zones from the NWB signal but ultimately chose `sessions_dict` scene information and the paper's trial-30 switch rule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the active zone, zero inside its inclusive boundaries, and positive after it, measured to the nearest zone edge; it is then discretized.

ii.
```python
dist[before] = position[before] - rz_start
dist[inside] = 0.0
dist[after] = position[after] - rz_end
dist_disc = discretize_distance_to_reward(dist)
```

iii. The agent described the zones as A=80–130, B=200–250, and C=320–370 cm and followed the requested signed-distance interpretation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Boolean masks implement the seven requested ranges, including an exact-zero class.

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

iii. The agent intended to implement the bins exactly as specified.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[start:end)` indices, so the output has one value per neural sample.

ii.
```python
trial_n = neural[start:end, :].T
pos = nwb_data['position'][start:end]
```

iii. The agent relied on the NWB streams being frame-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = bts['position/data'][:]
pos = nwb_data['position'][start:end]
```

iii. The agent identified position as the direct VR corridor coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to 0–450 cm and then floor-divided into 90 cm bins, with indices clipped to 0–4.

ii.
```python
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
out = np.clip(np.floor(position / 90).astype(int), 0, 4)
```

iii. The 450 cm track and five equal bins motivated 90 cm bins; clipping handles samples slightly outside the nominal track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Categories are floor(position/90), clipped to `[0, 4]`; thus 90, 180, 270, and 360 begin the next class.

ii.
```python
bin_size = TRACK_LENGTH / n_bins
out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. The agent followed the requested five equal-width bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural data are sliced by identical trial boundaries.

ii.
```python
trial_n = neural[start:end, :].T
pos = nwb_data['position'][start:end]
```

iii. The agent regarded the streams as natively aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the raw behavioral `lick/data` stream.

ii.
```python
lick = bts['lick/data'][:]
lck = licks_corrected[start:end]
```

iii. The agent identified this as the frame-level lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Per trial, if more than 30% of raw lick samples exceed 2, every lick value in that trial is set to NaN; those NaNs are subsequently changed to zero. All remaining positive values are binarized to one.

ii.
```python
if np.sum(trial_licks > 2) / len(trial_licks) > threshold:
    licks[start:end] = np.nan
...
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. The agent cited the paper's lick-sensor error rule and chose zero because categorical decoder outputs cannot contain NaN.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Corrected lick data and neural data are sliced with the same indices.

ii.
```python
trial_n = neural[start:end, :].T
lck = licks_corrected[start:end]
```

iii. The agent relied on native frame alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the hard-coded `SESSIONS_INFO` scene string and trial index, not the loaded raw `reward_zone` series.

ii.
```python
scene = SESSIONS_INFO[gcamp_name][exp_day]
rz_labels = parse_scene_reward_zones(scene, n_trials)
```

iii. The agent selected session scene metadata after examining the paper repository's session dictionary.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene suffixes give a constant A/B/C label, while switch scenes are parsed into before/after labels and switched at trial 30. Labels map to A=0, B=1, C=2 and are repeated over time.

ii.
```python
rz_labels[:ct] = zone_before
rz_labels[ct:] = zone_after
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
output_arr[4, :] = rz_loc
```

iii. The paper's scene names and fixed 30-trial pre-switch block were the stated basis.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` and position timestamps at trial boundaries; reward magnitudes and `autoreward` are not used.

ii.
```python
reward_ts = bts['Reward/timestamps'][:]
timestamps = bts['position/timestamps'][:]
```

iii. The agent observed that reward events use a separate timestamp stream.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is one if any reward timestamp lies inclusively between its start and teleport timestamps, otherwise zero; the scalar is repeated at every trial sample.

ii.
```python
if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
    reward_per_trial[i] = 1
output_arr[5, :] = reward_per_trial[i]
```

iii. This was chosen as the direct binary rewarded/omitted definition.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavioral arrays with unequal lengths are truncated to the shorter length. Invalid or sub-two-sample trials and sub-two-trial sessions are dropped. Lick-error trials are zero-filled. Unknown metadata and any session raising an exception are skipped with a warning. No imputation is performed for other missing values.

ii.
```python
min_len = min(n_behavior, n_neural)
position = position[:min_len]
deconvolved = deconvolved[:min_len]
...
except Exception as e:
    print(f"  Error: {e}")
    continue
```

iii. The trajectory specifically found a one-sample m18 session mismatch and added truncation. It considered successful verification and aggregate statistics sufficient evidence for the remaining handling.

## 13-a. What are the most time-consuming steps of the code?

i. The likely dominant work is reading full neural/behavior arrays from 152 NWB files, concatenating large plane arrays, computing a correlation separately for every curated neuron, constructing thousands of trial arrays, serializing the large pickle, and the later decoder training. The agent did not profile conversion timings.

ii.
```python
for subj_dir_name in subjects:
    for sess_file in session_files:
        nwb_data = load_nwb_session(filepath)
        result = process_session(nwb_data, scene, exp_day)
```

iii. The trajectory reports sample conversion, full conversion, verification, and decoder training, but gives no measured per-stage timings.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron Pearson-correlation loop can be replaced with column-wise centered covariance/std operations. Reward outcome can be assigned by binning reward timestamps into trial intervals instead of scanning all reward timestamps for every trial. Trial slicing itself remains naturally loop-based because trials have variable length.

ii.
```python
for c in range(n_neurons):
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
...
for i in range(n_trials):
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
```

iii. The agent did not discuss vectorization; it focused on correctness and decoder validation.

## 13-c. What processing does the code repeat multiple times?

i. Each session is first loaded during sample conversion and then reloaded during full conversion. Within full conversion, trial loops repeatedly allocate input/output arrays and repeat constants across time. Scene parsing and per-trial reward interval scans also repeat work.

ii.
```python
input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
output_arr = np.zeros((6, n_timepoints), dtype=np.int64)
```

iii. The trajectory explicitly ran a sample conversion before the full conversion for validation; it did not identify repeated computation as a concern.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and truncates `trial_num`, raw `environment`, raw `reward_zone`, `autoreward`, reward event values, and `plane_idx` without using them in converted variables. It creates an immediately overwritten placeholder `input_arr` and an unused `input_per_trial`. It also stores several diagnostic fields only to print or place in metadata.

ii.
```python
trial_num = bts['trial number/data'][:]
environment = bts['environment/data'][:]
reward_zone = bts['reward_zone/data'][:]
...
input_arr = np.array([time_from_start[0] if False else 0])
input_per_trial = np.array([...])
```

iii. The trajectory supplies no justification for these discarded values; they appear to be remnants of exploration and an abandoned mixed scalar/time-varying representation.
