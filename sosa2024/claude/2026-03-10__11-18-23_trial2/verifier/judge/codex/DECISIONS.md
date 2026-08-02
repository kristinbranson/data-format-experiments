# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `data/` directory for `sub-*` subject folders, then scans each subject folder for `*.nwb` files. Each NWB file is treated as one session and opened directly with `h5py`, not `pynwb`. Within each file it loads the ophys arrays, behavioral time series, and reward timestamps needed for conversion.

ii.
```python
def find_nwb_files(data_dir='data'):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    sessions = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
        for fpath in files:
            sessions.append({
                'subject': subj.replace('sub-', ''),
                'filepath': fpath,
                'filename': os.path.basename(fpath),
            })
    return sessions

with h5py.File(filepath, 'r') as f:
    deconv_group = f['processing/ophys/Deconvolved']
    position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset is organized as `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb` and that this structure covers all 11 subjects and 152 sessions. The trajectory and notes also state the agent intentionally used NWB contents directly rather than reconstructing the higher-level paper pipeline.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `sub-*` directories under `data/`. The stored `subject_id` field from each NWB is also read and later used when building `subjects` and `subject_idx`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
subject_id = f['general/subject/subject_id'][()].decode()
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes justify this by listing the subject directories and matching them to the paper's 11 mice.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The script keeps one entry per file in `sessions_info`, processes each file once, and appends one session-level element to `neural`, `input`, and `output`.

ii.
```python
files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
for fpath in files:
    sessions.append({
        'subject': subj.replace('sub-', ''),
        'filepath': fpath,
        'filename': os.path.basename(fpath),
    })
...
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
    all_neural.append(result['neural'])
```

iii. The notes state there are 152 session files total and treat each NWB file as one session.

## 1-d. How are the data split into trials?

i. Trials are defined by indices where `trial_start > 0` and `teleport > 0`. The script pairs each trial start with the first later teleport if the counts do not already match, then slices `[start:end)` per trial.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
...
if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)
...
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes say trial boundaries come from `trial_start` and `teleport`. The agent's trajectory repeatedly describes trial alignment as "start of trial to teleport."

## 1-e. How are trials filtered based on quality controls?

i. The AI only discards sessions with fewer than 2 trials and trial segments with fewer than 2 frames. It does not apply the reference script's `< 50` timepoint trial filter.

ii.
```python
if n_trials < 2:
    print(f"  WARNING: Only {n_trials} trials, skipping session")
    return None
...
n_tp = ei - si
if n_tp < 2:
    continue
```

iii. There is no strong explicit justification in the notes beyond wanting "cleaner output" and keeping sessions usable. The notes do not mention the human reference's `min_ntimepoints=50` rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrix is derived primarily from the NWB `processing/ophys/Deconvolved/<plane>/data` arrays. The script also loads `Fluorescence`, `Neuropil`, `iscell`, and `planeIdx` to filter cells before selecting the final columns.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
...
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
neural_all = deconv[:, final_cell_mask].T
```

iii. The notes explicitly say the NWB files already contain "the ALREADY PROCESSED deconvolved events" and that these should be used directly.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, crops neural and behavior to a common length if needed, applies `iscell`, computes a simple neuropil-corrected `dF/F` proxy from fluorescence and neuropil, uses that proxy to identify putative interneurons via speed correlation, removes those cells, transposes to `(neurons, time)`, and replaces NaNs with 0 per trial.

ii.
```python
deconv = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(flu_list, axis=1)
neuropil_data = np.concatenate(neu_list, axis=1)
...
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
...
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
final_cell_mask = cell_mask_concat & ~interneuron_mask
neural_all = deconv[:, final_cell_mask].T
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The notes justify using NWB deconvolved events directly, but also claim the paper excluded interneurons and that a "vectorized z-scored dot product for speed-dFF correlation" implements that exclusion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered first by `iscell[:, 0] == 1`, then by a custom interneuron exclusion rule: compute a simplified `dF/F`, correlate it with speed on valid frames, and drop cells with correlation `> 0.5`.

ii.
```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
...
valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
...
corrs = (speed_z @ dff_z) / len(speed_z)
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True
...
final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. The notes say `iscell` comes from Suite2p curation and that the paper removed putative interneurons; they therefore chose to add an extra speed-correlation filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each trial by slicing the neural session matrix from each `trial_start` index to the paired `teleport` index. No additional temporal shift is applied.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes repeatedly state the conversion is "Temporally align based on start of the trial" and that the trial boundaries already provide that alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are kept at the imaging frame rate, with nominal bin size `1000 / 15.5078125 = 64.48 ms`. No temporal rebinning is applied.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
frame_time = 1.0 / imaging_rate
...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
...
'time_bin_size': time_bin_ms,
```

iii. The notes say all behavior time series are already aligned at about 15.5 Hz and the NWB data should be used directly.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI does not derive this from NWB timestamps. It derives it from frame count and imaging rate: `np.arange(n_tp) * (1.0 / imaging_rate)`.

ii.
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The notes explicitly map this variable as "Frame index * time_bin_size, in seconds."

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, it creates a uniformly spaced vector from 0 using the frame duration implied by the imaging rate. It does not subtract the actual first behavioral timestamp for that trial.

ii.
```python
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input_tv = time_from_start.reshape(1, -1)
```

iii. The only explicit justification is the mapping note that time should be frame index times bin size.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector length is the same `n_tp` as the neural slice for that trial, so each frame gets one time value.

ii.
```python
n_tp = ei - si
trial_neural = neural_all[:, si:ei].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input = np.vstack([
    trial_input_tv,
    np.full((1, n_tp), env_type, dtype=np.float32),
    ...
])
```

iii. The notes and code both assume imaging and behavior are already frame-aligned, so equal-length per-trial slices are sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although the script loads the NWB `environment` series, the converted environment label is actually derived from the session `identifier` string via scene parsing. The per-trial environment comes from `get_reward_zone_labels(scene, n_trials)`.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_env = env_per_trial.copy()
```

iii. The notes justify this as more reliable for switch sessions, especially cross-environment switches.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses scene names such as `Env1_LocationB_to_A` or `Env1_A_to_Env2_B`, converts `Env1/Env2` to `0/1`, and assigns the first 30 trials to the "from" environment and remaining trials to the "to" environment for switch sessions. It then repeats the resulting scalar across all timepoints in the trial.

ii.
```python
if '_to_' in scene:
    ...
    labels[:change_trial] = from_zone
    labels[change_trial:] = to_zone
    env_per_trial[:change_trial] = from_env
    env_per_trial[change_trial:] = to_env
...
trial_input = np.vstack([
    trial_input_tv,
    np.full((1, n_tp), env_type, dtype=np.float32),
    ...
])
```

iii. The trajectory and notes say this was chosen because scene parsing was considered "more reliable for cross-env switches."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `t` after trials are segmented from `trial_start` and `teleport`.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. The notes describe trial number simply as an integer per trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond using the sequential within-session trial index and repeating it across the full trial.

ii.
```python
trial_num = np.float32(t)
...
np.full((1, n_tp), trial_num, dtype=np.float32),
```

iii. No deeper justification is given beyond matching the decoder requirement for per-trial trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward/timestamps` array aligned onto behavior frame timestamps, then summarized per trial.

ii.
```python
behav_timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. The notes describe reward as event-based and say it was converted to a frame-aligned signal by matching reward timestamps to behavior timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first computes `trial_rewarded[t] = 1` if any aligned reward frame falls between the current trial's start and end. It then creates `prev_outcome` by shifting that vector by one trial, with the first trial forced to 0, and repeats that value across time within each trial.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. The notes map this variable as a binary lagged per-trial outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance is derived from trial position samples and reward-zone coordinates inferred from the parsed scene name, not from the NWB `reward_zone` series. For each trial, the script uses `rz_coords[t]` plus `position[si:ei]`.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_pos = position[si:ei]
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes justify this by saying reward-zone labels can be reliably parsed from the scene string, including switch sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the signed distance is 0 inside the zone, negative before the zone, and positive after the zone, measured relative to the nearest zone edge.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    dist[before] = position[before] - rz_start
    dist[after] = position[after] - rz_end
    dist[inside] = 0.0
    return dist
```

iii. The notes say this follows the reward-relative position concept in the paper and uses the canonical A/B/C reward-zone coordinates.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI manually thresholds signed distance into seven bins matching the user instruction.

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

iii. The notes map this output as "7 bins based on signed distance."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Both the neural slice and the position slice use the same `[si:ei)` indices for a trial, so the distance sequence is frame-aligned with neural data.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_bins = discretize_distance(signed_dist)
```

iii. The notes say all behavior time series are already aligned to imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the NWB `position` behavioral time series.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = position[si:ei]
```

iii. The notes list `position` as one of the frame-aligned behavior series available in each NWB.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The only processing is per-trial slicing and discretization. The raw position values are otherwise used directly.

ii.
```python
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. The notes map absolute position as a discretized version of corridor position.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is binned into five equal 90 cm intervals over a nominal 0 to 450 cm track via `floor(position / 90)` clipped to `[0, 4]`.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The notes justify this as "5 equal bins (90cm each)" based on the 450 cm track length.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by using the same trial slice indices for both position and neural activity.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. The notes assume behavior and imaging are already frame-aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the NWB `lick` behavioral time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
trial_lick = lick_binary[si:ei]
```

iii. The notes identify lick as one of the directly available behavior streams.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first binarizes all positive lick values to 1. It then applies a stuck-sensor heuristic: if more than 30% of a trial's frames have raw lick count `> 2`, it sets the entire trial's lick output to 0 rather than NaN. Finally it binarizes again when writing `trial_output`.

ii.
```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1
...
if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
...
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes cite the paper's lick-sensor error discussion and say the agent wanted "cleaner output," but changed the correction from setting bad trials to NaN to setting them to 0.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned with neural data by slicing the same trial frame interval for both.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes state behavior streams are already synchronized to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session `identifier` string by scene parsing, not from the raw `reward_zone` behavior series.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The notes say scene parsing handles single-zone, within-environment switch, and cross-environment switch sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers the zone label per trial from the parsed scene string, assumes the switch happens at trial 30 for switch sessions, converts zone labels `A/B/C` to integer classes `0/1/2`, and repeats that class across the trial.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    ...
    labels[:change_trial] = from_zone
    labels[change_trial:] = to_zone
...
rz_loc = zone_map.get(rz_labels[t], 0)
...
np.full((1, n_tp), rz_loc, dtype=np.int64),
```

iii. The notes justify this by saying the scene name fully specifies reward-zone identity and switches.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, aligned to behavior-frame timestamps, then summarized per trial.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
rew_outcome = int(trial_rewarded[t])
```

iii. The notes emphasize that the NWB reward field is sparse and must be frame-aligned first.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are projected onto the nearest behavior frame. A trial is marked rewarded if any reward frame falls between that trial's start and teleport indices. The result is then repeated across all timepoints in the trial output matrix.

ii.
```python
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
if np.any(reward_frames[si:ei] > 0):
    trial_rewarded[t] = 1
...
np.full((1, n_tp), rew_outcome, dtype=np.int64),
```

iii. The notes say this matches their decision to convert sparse reward events into a frame-aligned per-trial binary outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles mismatched neural and behavior lengths by truncating all streams to the shorter length, tries to repair start/end count mismatches by pairing each trial start with the next later teleport, replaces NaNs in neural data with 0, zeroes out likely stuck-lick trials, and skips sessions with fewer than 2 trials or trial segments shorter than 2 frames.

ii.
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    position = position[:min_len]
...
if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
...
if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
```

iii. The notes justify truncation for off-by-one mismatches and cite the paper's lick-sensor issue, but several other repairs are ad hoc and not strongly justified in the notes.

## 13-a. What are the most time-consuming steps of the code?

i. The heaviest steps are loading full NWB arrays for each session, especially `Deconvolved`, `Fluorescence`, and `Neuropil`; computing the simplified `dF/F` and neuron-speed correlations for interneuron exclusion; iterating over all trials to build trialwise arrays; and finally pickling the very large output dataset.

ii.
```python
with h5py.File(filepath, 'r') as f:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
    ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
for t in range(n_trials):
    ...
    neural_trials.append(...)
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. This is inferred from the code structure and from the notes' emphasis on large NWB sessions and a 9.4 GB output pickle.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The list-comprehension over ROIs for `iscell` mapping, the reward timestamp loop using `np.argmin` per reward event, the per-trial loops used separately for reward labeling, lick correction, and trial extraction, and the small loop over accepted cells when applying the interneuron threshold could all be vectorized or combined further.

ii.
```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
...
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
...
for t in range(n_trials):
    ...
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True
```

iii. The notes explicitly celebrate vectorization for the speed-correlation step, which implies these remaining loops were left unoptimized.

## 13-c. What processing does the code repeat multiple times?

i. The code traverses trial boundaries multiple times: once for trial reward outcome, once for lick correction, and again to build trial inputs/outputs. It also loads `environment` data but then ignores it in favor of scene parsing, and it computes `trial_input_pt` but never uses it.

ii.
```python
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    ...
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)
```

iii. This is mostly an inference from the code; the notes do not explicitly discuss repeated traversal of the same per-trial intervals.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads fluorescence and neuropil and computes a simplified `dF/F` solely to remove putative interneurons, even though the final saved neural data are still the deconvolved events. It also loads `environment` but discards it, and constructs `trial_input_pt` but never uses it.

ii.
```python
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)
```

iii. The notes justify the interneuron filter scientifically, but from the standpoint of the saved dataset these intermediate arrays and one temporary variable are discarded.
