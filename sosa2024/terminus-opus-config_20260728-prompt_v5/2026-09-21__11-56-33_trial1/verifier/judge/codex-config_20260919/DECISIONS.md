# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent lists every `sub-*` directory under `/app/data`, sorts every `*.nwb` file in each directory, and processes each file with `pynwb.NWBHDF5IO`. Full mode uses all 152 files; sample mode deliberately selects two.

ii.
```python
subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
for sub in subjects:
    files = sorted(glob.glob(os.path.join(data_dir, sub, '*.nwb')))
    all_nwb_files.extend(files)
with NWBHDF5IO(nwb_file, 'r') as io:
    nwb = io.read()
```

iii. The notes report 11 subjects and 152 sessions and justify NWB loading as covering the complete one-level subject/session layout.

## 1-b. How are the data split into subjects?

i. Subject directories identify candidate subjects, while the saved subject ID comes from `nwb.subject.subject_id`. A first-seen unique-subject list is built, and each session receives its index into that list.

ii.
```python
subject_id = nwb.subject.subject_id
if subj not in unique_subjects:
    unique_subjects.append(subj)
all_subject_idx.append(unique_subjects.index(subj))
```

iii. The notes say the directory naming and NWB subject metadata agree and yield the paper's 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session. Its `session_id` and scene are retained in metadata; sessions returning `None` would be skipped.

ii.
```python
for si, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The agent inferred this from one file per subject/day and validated that all 152 files became 152 sessions.

## 1-d. How are the data split into trials?

i. Starts are frames where `trial_start == 1`. For each start, the end is the first later frame where `teleport == 1`; the end frame is excluded. If none exists, the final data frame is used.

ii.
```python
trial_start_inds = np.where(tstart == 1)[0]
teleport_inds = np.where(teleport == 1)[0]
later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
trial_ends[i] = later_teleports[0] if len(later_teleports) else len(position) - 1
trial_neural = dff[:, start:end]
```

iii. Exploration showed that trial start through teleport is the active corridor traversal and the following gray/teleport interval is an ITI.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than two detected trials are rejected. Individual trials are retained whenever the clipped interval has at least two frames; after segmentation, sessions with fewer than two retained trials are rejected. There is no 50-frame trial cutoff.

ii.
```python
if n_trials < 2:
    return None
...
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The notes state that the paper has no explicit trial exclusion beyond ordinary data quality and that even very long trials were considered legitimate. They do not justify choosing two frames as the minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is derived from each plane's raw `Fluorescence` (`F`) and `Neuropil` (`Fneu`) series, restricted using `iscell` and `planeIdx`, then pooled across planes. The stored NWB `Deconvolved` field is not used.

ii.
```python
F_plane = nwb.processing['ophys']['Fluorescence'][plane_name].data[:]
Fneu_plane = nwb.processing['ophys']['Neuropil'][plane_name].data[:]
F_list.append(F_plane[:, plane_iscell].T.astype(np.float64))
Fneu_list.append(Fneu_plane[:, plane_iscell].T.astype(np.float64))
```

iii. The agent observed that NWB `Deconvolved` values were on a raw-fluorescence scale and therefore chose to reconstruct the signal from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The agent subtracts `0.7*Fneu`, adds back the per-trial mean neuropil, estimates a per-trial maximin baseline (Gaussian sigma 15, 20-second minimum then maximum filters), computes `(F-baseline)/abs(baseline)`, smooths each trial with a sigma-2 Gaussian, and saves this dF/F. It does not perform OASIS deconvolution.

ii.
```python
f_ = F - neu_coef * Fneu
f_[:, start:stop] += neu_coef * np.nanmean(Fneu[:, start:stop], axis=1, keepdims=True)
flow[:, start:stop] = nansmooth(f_[:, start:stop], 15, axis=1)
flow[:, start:stop] = ndimage.minimum_filter1d(flow[:, start:stop], window_size, axis=1)
flow[:, start:stop] = ndimage.maximum_filter1d(flow[:, start:stop], window_size, axis=1)
dff[valid_mask] = (f_[valid_mask] - flow[valid_mask]) / np.abs(flow[valid_mask])
dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
```

iii. The notes explicitly choose dF/F because it is reproducible and information-preserving, despite acknowledging that the paper's GLM and place-cell analyses use deconvolved calcium events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs first pass manual Suite2p curation (`iscell[:,0] == 1`). Then cells whose dF/F has Pearson correlation greater than 0.5 with running speed are removed as putative interneurons.

ii.
```python
plane_iscell = iscell[plane_mask, 0] == 1
...
corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
if not np.isnan(corr) and corr > 0.5:
    interneuron_mask[c] = False
```

iii. Both filters and the 0.5 threshold are taken from the Methods; planes are pooled as in the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural frames are sliced from the `trial_start` frame to, but not including, the teleport frame. Thus time zero is trial start and trial lengths remain variable.

ii.
```python
start = trial_start_inds[i]
end = trial_ends[i]
trial_neural = dff[:, start:end].astype(np.float32)
```

iii. The notes say neural and behavioral streams are already frame-aligned, so common indices provide trial-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Native imaging/behavior frames are retained, and metadata declares `1000/15.5078125 = 64.48 ms` per bin.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,
'frame_rate': 15.5078125,
```

iii. The notes report a common effective sampling rate of about 15.5 Hz and preserve it to avoid unnecessary resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses timestamps attached to the raw `position` behavioral time series and the detected trial-start index.

ii.
```python
timestamps = bts.time_series['position'].timestamps[:].astype(np.float64)
time_from_start = timestamps[start:end] - timestamps[start]
```

iii. The agent considered the behavioral streams frame-aligned and used position timestamps as the shared clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial and the result is cast to float32.

ii.
```python
time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
```

iii. This directly implements elapsed seconds from the alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start:end]` indices and clipped end are used for timestamps and neural data, and the resulting vector is placed in the first input row.

ii.
```python
end = min(end, dff.shape[1], len(position))
trial_neural = dff[:, start:end]
input_data[0, :] = timestamps[start:end] - timestamps[start]
```

iii. The notes say common frame indexing makes explicit interpolation unnecessary.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the scene string parsed from the NWB `identifier`, not from the raw per-frame `environment` time series.

ii.
```python
scene = nwb.identifier.split('/')[-1]
env_per_trial = get_env_per_trial(scene, n_trials)
```

iii. The agent found scene names encode Env1/Env2 and switches, and used the known switch after trial 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene-name text is mapped to 0 for Env1 and 1 for Env2. Cross-environment sessions are split at trial index 30; the scalar trial value is broadcast across frames.

ii.
```python
env_labels[:change_trial] = 0
env_labels[change_trial:] = 1
...
input_data[1, :] = env_val
```

iii. The notes cite the experimental design's trial-30 switch and report checks against scene names.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based Python loop index over detected `trial_start` events, not from the stored `trial number` values (although that series is loaded).

ii.
```python
trial_num = bts.time_series['trial number'].data[:]
for i in range(n_trials):
    trial_number = float(i)
```

iii. Trial exploration showed stored trial numbering changes near teleport, so start-event order was judged clearer.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is cast to float and broadcast across all frames of its trial.

ii.
```python
input_data[2, :] = trial_number
```

iii. The agent defines the target as a continuous, zero-indexed within-session trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It comes from the timestamps of the raw `Reward` time series, together with trial start/end timestamps.

ii.
```python
reward_timestamps = bts.time_series['Reward'].timestamps[:]
rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. The notes identify `Reward` as the authoritative event stream and validate an omission rate near 15%.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any reward timestamp lies inclusively between its start and end times. The reward vector is shifted by one trial; the first trial defaults to zero. Values are broadcast over trial frames.

ii.
```python
mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
rewarded[i] = int(np.any(mask))
prev_reward = np.zeros(n_trials, dtype=int)
prev_reward[1:] = rewarded[:-1]
```

iii. This directly implements omitted=0/rewarded=1 for the preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It combines raw per-frame `position` with reward-zone coordinates inferred from the scene string in `nwb.identifier` and the fixed A/B/C ranges.

ii.
```python
rz_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_coords[i,0], rz_coords[i,1])
```

iii. The scene encodes location and switches, and the coordinate ranges come from the paper/reference behavior code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the zone is expressed relative to its start (negative), position after the zone relative to its end (positive), and every position inside the zone is zero.

ii.
```python
distance[before_zone] = position[before_zone] - rz_start
distance[in_zone] = 0.0
distance[after_zone] = position[after_zone] - rz_end
```

iii. This is the signed distance to the nearest location in the zone and matches the requested semantics.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks produce seven classes with boundaries -50, -10, exactly 0, +10, and +50 cm.

ii.
```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The agent states these are the instruction's exact categories and spot-checked them.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the identical clipped `[start:end]` slice, so each distance class corresponds to the same frame.

ii.
```python
trial_neural = dff[:, start:end]
trial_pos = position[start:end]
output_data[0, :] = dist_bins
```

iii. The agent relies on shared frame alignment in the NWB streams.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw `position` behavioral series.

ii.
```python
position = bts.time_series['position'].data[:].astype(np.float64)
trial_pos = position[start:end]
```

iii. Position is recorded in centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is not smoothed or resampled; it is only discretized.

ii.
```python
pos_bins = discretize_position(trial_pos)
output_data[1, :] = pos_bins
```

iii. The notes say the raw values already represent the desired corridor coordinate.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is mapped into five 90-cm bins: `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`.

ii.
```python
bins[position < 90] = 0
bins[(position >= 90) & (position < 180)] = 1
bins[(position >= 180) & (position < 270)] = 2
bins[(position >= 270) & (position < 360)] = 3
bins[position >= 360] = 4
```

iii. Five equal divisions of the 450-cm track give 90-cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same `[start:end]` frame slice is applied to position and neural data.

ii.
```python
trial_neural = dff[:, start:end]
trial_pos = position[start:end]
```

iii. Shared NWB frame indices provide direct alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw per-frame behavioral `lick` series.

ii.
```python
lick = bts.time_series['lick'].data[:].astype(np.float64)
trial_lick = lick[start:end]
```

iii. Exploration found that raw values may exceed one, so they are event counts rather than an already binary flag.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every positive value becomes 1 and all other values become 0.

ii.
```python
lick_binary = (trial_lick > 0).astype(np.int64)
```

iii. This implements the requested no/yes output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays are sliced using the identical trial frame boundaries.

ii.
```python
trial_neural = dff[:, start:end]
trial_lick = lick[start:end]
```

iii. The agent says all behavioral streams are already aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene suffix in `nwb.identifier`, rather than the raw `reward_zone` and `position` series.

ii.
```python
scene = parse_scene_name(identifier)
rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
```

iii. The identifier explicitly names LocationA/B/C and any transition, which the agent considered more direct than the noisy behavioral field.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene text is parsed into A/B/C. Switch sessions change at trial 30. Labels are encoded A=0, B=1, C=2 and broadcast over time.

ii.
```python
rz_labels[:ct] = label1
rz_labels[ct:] = label2
rz_loc = {'A': 0, 'B': 1, 'C': 2}.get(rz_label, 0)
output_data[4, :] = rz_loc
```

iii. The fixed ranges and trial-30 switch come from the experimental design and reference code.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the event timestamps of the raw `Reward` behavioral time series and trial boundary timestamps.

ii.
```python
reward_timestamps = reward_ts_obj.timestamps[:]
rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. The notes treat delivered reward events, rather than autoreward or reward-zone occupancy, as the outcome source.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, any reward timestamp within the inclusive start/end interval yields 1; otherwise it yields 0. The scalar is broadcast across the trial.

ii.
```python
mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
if np.any(mask):
    rewarded[i] = 1
output_data[5, :] = reward_outcome
```

iii. This implements a binary per-trial delivery outcome, and the resulting 15.3% omission rate matches the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Trial ends are clipped to the shortest available neural/position length; trials shorter than two frames and sessions with fewer than two valid trials are skipped. Missing final teleports fall back to the last position frame. Neural NaNs are replaced with zero. Unknown reward-zone scenes cause session rejection, while unknown individual labels default to A. No interpolation or global timestamp consistency assertions are performed.

ii.
```python
trial_ends[i] = len(position) - 1
end = min(end, dff.shape[1], len(position))
if n_frames < 2:
    continue
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
if rz_labels is None:
    return None
```

iii. The notes mention one-frame neural/behavior mismatches and present clipping and NaN replacement as defensive handling. They report that all sessions passed.

## 13-a. What are the most time-consuming steps of the code?

i. Loading full NWB fluorescence arrays and computing per-trial dF/F baselines dominate. The measured full conversion took about 428 seconds; pickle serialization of the roughly 9.4-GB duplicated arrays is also substantial.

ii.
```python
F_plane = ...data[:]
Fneu_plane = ...data[:]
dff = compute_dff(F, Fneu, ...)
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes provide per-session timings of up to about 6 seconds for dF/F and describe NWB loading as largely unavoidable.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell interneuron-correlation loop could be vectorized. Trial-end matching repeatedly filters the full teleport array and could use `searchsorted`. Reward assignment loops over trials, as do baseline computation and output construction; some full-session transforms could be done before splitting.

ii.
```python
for c in range(n_cells):
    corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
for i in range(n_trials):
    later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
```

iii. The notes specifically identify the per-cell correlation check and say discretization was already vectorized; variable-length trials make complete vectorization less convenient.

## 13-c. What processing does the code repeat multiple times?

i. Each trial's start/end is traversed repeatedly: once to add neuropil and compute baseline, again to smooth dF/F, again during reward classification, and again to construct trial arrays. Scene parsing and reward-zone/environment arrays also encode related scene/switch information separately. Per-trial scalars are broadcast into large time-series matrices.

ii.
```python
for i in range(n_trials):  # baseline
    ...
for i in range(n_trials):  # dFF smoothing
    ...
for i in range(n_trials):  # rewards
    ...
for i in range(n_trials):  # segmentation/output
    ...
```

iii. The agent's notes do not discuss these repeated passes in detail; they emphasize that conversion remained fast enough and that vectorized discretizers were used.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `trial number` but never uses its values. It creates `input_time_varying` and `input_per_trial` and then discards both in favor of a newly allocated broadcast matrix. It computes/returns diagnostic `env_per_trial`, `rz_labels`, and `rewarded` fields that are not copied into the final dataset. It also casts full F/Fneu arrays to float64 and later converts trial neural data to float32.

ii.
```python
trial_num = bts.time_series['trial number'].data[:].astype(np.float64)
input_time_varying = time_from_start.reshape(1, -1)
input_per_trial = np.array([env_val, trial_number, prev_outcome], dtype=np.float32)
result = {..., 'env_per_trial': env_per_trial, 'rz_labels': rz_labels, 'rewarded': rewarded}
```

iii. The agent did not document these as unnecessary; they appear to be development artifacts and diagnostics.
