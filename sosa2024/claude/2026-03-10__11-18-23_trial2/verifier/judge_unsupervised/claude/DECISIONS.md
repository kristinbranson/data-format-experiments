# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over subject directories (`data/sub-*`), finds all `.nwb` files sorted by name, and processes each NWB file individually using `h5py`. Each NWB file represents one session. Data is loaded from the HDF5 groups: `processing/ophys/` for neural data and `processing/behavior/BehavioralTimeSeries/` for behavioral variables. Results are accumulated into lists across sessions.

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
```

iii. The agent identified that data is organized as NWB files per session, grouped by subject directory. All 152 NWB files across 11 subjects are discovered and processed sequentially.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from directory names (e.g., `sub-m11` -> `m11`). A unique sorted list of subject IDs is built, and `subject_idx` maps each session to its subject index.

ii.
```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. Subject identity is parsed from the directory structure, consistent with the NWB organization (DANDI:001361). The agent confirmed 11 unique subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The `process_session()` function handles a single NWB file, returning lists of per-trial data. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
    if result is None:
        continue
    if result['n_trials'] < 2:
        continue
    all_neural.append(result['neural'])
    ...
```

iii. The agent identified that each NWB file corresponds to one session. 152 sessions were processed.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by the `trial_start` signal (frame indices where `trial_start > 0`) and the `teleport` signal (frame indices where `teleport > 0`). Each trial spans from a `trial_start` index to the corresponding `teleport` index. Teleport events are matched to trial starts by finding the first teleport after each trial start.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
n_trials = len(trial_start_inds)
# Match teleport to trial start
if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
```

iii. The agent used the NWB's `trial_start` and `teleport` behavioral signals, consistent with the reference code's trial boundary detection. Trials with fewer than 2 timepoints are skipped.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints (`n_tp < 2`) are excluded. There is no additional trial-level quality filtering (e.g., based on speed, running behavior, or other criteria).

ii.
```python
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    n_tp = ei - si
    if n_tp < 2:
        continue
```

iii. The agent applied a minimal quality filter (requiring at least 2 timepoints). The reference code's decoder analysis (`get_timeseries_data`) applies a speed threshold (speed < 2 cm/s) for place cell analyses, but the agent correctly noted that speed is a decoder output in this task, so that filter was not applied to include all trial data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `Deconvolved` data in the NWB files (`processing/ophys/Deconvolved/{plane}/data`). For multi-plane sessions (subjects m17, m18), data from all planes is concatenated along the neuron axis.

ii.
```python
deconv_list = []
for plane in planes:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    deconv_list.append(d)
deconv = np.concatenate(deconv_list, axis=1)
```

iii. The agent determined that NWB files contain already-processed deconvolved calcium events (output of the full dF/F pipeline: neuropil subtraction, maximin baseline, smoothing, OASIS deconvolution). Using these directly avoids reprocessing.

## 2-b. How is the `neural` data processed?

i. The deconvolved events are used directly without additional processing. The only transformations are: (1) filtering to accepted cells (iscell + interneuron exclusion), (2) transposing to (n_neurons, n_timepoints), (3) converting to float32, and (4) replacing NaN values with 0.

ii.
```python
neural_all = deconv[:, final_cell_mask].T  # (n_neurons, n_timepoints)
neural_all = neural_all.astype(np.float32)
# Per trial:
trial_neural = neural_all[:, si:ei].copy()
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The agent's CONVERSION_NOTES state that NWB `Deconvolved` data is already fully processed, so no additional dF/F or deconvolution computation is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering is applied: (1) Suite2p's `iscell` classification (iscell[:,0] == 1), and (2) interneuron exclusion based on speed-dF/F correlation > 0.5. The interneuron filter computes a simplified dF/F using neuropil-corrected fluorescence with a median baseline, then correlates with speed using vectorized z-scored dot product.

ii.
```python
# iscell filter
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

# Interneuron exclusion
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
f_median[f_median == 0] = 1
dff_simple = (f_corrected - f_median) / np.abs(f_median)

valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
# ... vectorized correlation ...
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True

final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. The agent followed the reference code's two-step filtering. The interneuron exclusion uses a simplified dF/F (median baseline) rather than the full maximin baseline from the reference preprocessing pipeline. The 0.7 neuropil coefficient and 0.5 correlation threshold match the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. Each trial's neural data begins at the frame where `trial_start > 0` and ends at the corresponding `teleport` frame. Time from trial start is computed as frame index times frame duration.

ii.
```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
```

iii. The instructions specify "Temporally align based on start of the trial." The agent aligns to the first frame of each trial (trial_start signal), consistent with both the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~15.5078125 Hz, ~64.48 ms/frame) is used directly. No temporal rebinning is applied.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL  # ~64.48 ms
```

iii. The agent uses the native imaging rate as the time bin. The reference code operates at the same rate. No rebinning is needed since all data streams (neural, behavioral) are already sampled at the imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the imaging frame rate and the frame index within each trial. No raw variable is read; it is computed from the trial structure and the known imaging rate.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The agent computes time from trial start by multiplying the frame index by the frame duration (1/imaging_rate), yielding a linearly increasing time in seconds starting at 0.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Simple multiplication: frame_index * (1/imaging_rate). The imaging rate is read from the NWB file's `general/optophysiology/ImagingPlane/imaging_rate` field.

ii.
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Straightforward computation. The imaging rate is ~15.5 Hz, so each frame is ~64.5 ms.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Perfectly aligned by construction. Both neural data and time_from_start use the same frame indices (from trial_start to teleport), so they share the same number of timepoints.

ii.
```python
# Neural: neural_all[:, si:ei]  -> n_tp frames
# Time:   np.arange(n_tp) * frame_time  -> n_tp frames
trial_input_tv = time_from_start.reshape(1, -1)  # (1, n_tp)
```

iii. Since time is derived from the frame count, it is inherently aligned with the neural data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the session's scene name parsed from the NWB `identifier` field (e.g., `Env1_LocationB_to_A`), NOT from the NWB `environment` behavioral timeseries.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
# ...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
trial_env = env_per_trial.copy()
```

iii. The agent chose to parse environment from the scene name for reliability, particularly for cross-environment switch sessions. The NWB `environment` field is available but not used for this purpose.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to extract environment numbers. For single-environment sessions (e.g., `Env1_LocationA`), all trials get the same environment. For switch sessions (e.g., `Env1_B_to_Env2_C`), trials 0-29 get the first environment and trials 30+ get the second. Environment is encoded as binary: Env1=0, Env2=1.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        from_zone, from_env = _parse_zone_and_env(from_parts)
        to_zone, to_env = _parse_zone_and_env(to_parts)
        env_per_trial[:change_trial] = from_env
        env_per_trial[change_trial:] = to_env
    else:
        zone, env = _parse_zone_and_env(parts)
        env_per_trial[:] = env
```

iii. The agent hardcodes the switch trial at 30, consistent with the paper's description that reward zone switches occur after trial 30. The per-trial value is broadcast to all timepoints.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `t` (0-based) during trial iteration, NOT from the NWB `trial number` behavioral timeseries.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. The agent uses the sequential trial index rather than the NWB field. The NWB `trial number` field contains per-frame values with -1 for inter-trial periods. The loop index `t` effectively provides the same 0-based trial numbering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The 0-based trial index is cast to float32 and broadcast to all timepoints within the trial.

ii.
```python
trial_num = np.float32(t)
np.full((1, n_tp), trial_num, dtype=np.float32)
```

iii. Minimal processing - just type conversion and broadcasting.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward outcome of the previous trial, which itself is computed from the sparse `Reward/timestamps` in the NWB file matched to behavioral frame timestamps.

ii.
```python
# Reward per trial
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0

trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1

# Previous outcome
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. The agent computes reward outcome per trial by mapping sparse reward event timestamps to imaging frames, then checks if any reward event falls within the trial. Previous outcome is the lagged reward outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial t > 0, the previous trial outcome is `trial_rewarded[t-1]`. For trial 0, the previous outcome is set to 0 (no reward / omission). The value is broadcast to all timepoints.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
# Trial 0: uses default 0 (omission)
```

iii. The agent treats the first trial's previous outcome as 0 (omission), a reasonable default since there is no prior trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from (1) the `position` behavioral timeseries (animal position on the track in cm) and (2) the reward zone coordinates (start, end) determined from the scene name.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
# ...
trial_pos = position[si:ei]
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Position is directly available in the NWB. Reward zone coordinates are parsed from the session's scene name using a lookup table matching the reference code (A=[80,130], B=[200,250], C=[320,370]).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. A signed distance is computed: negative if the animal is before the zone (position < zone_start), positive if after (position > zone_end), and zero if inside the zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    dist[before] = position[before] - rz_start  # negative
    dist[after] = position[after] - rz_end       # positive
    dist[inside] = 0.0
    return dist
```

iii. This implements signed distance to the nearest edge of the reward zone, consistent with the concept of "distance to any location in the reward zone" from the instructions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 bins matching the instruction specification exactly.

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

iii. Bins match the instructions: 0: < -50cm, 1: -50 to -10cm, 2: -10 to <0cm, 3: 0cm, 4: >0 to +10cm, 5: +10 to +50cm, 6: >+50cm.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Aligned by using the same frame indices (si:ei) for both neural and behavioral data within each trial. Position data comes from the same frame-aligned behavioral timeseries.

ii.
```python
trial_pos = position[si:ei]
# same si:ei as trial_neural = neural_all[:, si:ei]
```

iii. Neural and behavioral data share the same frame-level alignment in the NWB files, so slicing with the same indices ensures temporal alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral timeseries in the NWB file.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. Position is directly available in the NWB as a continuous variable in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw position is discretized into 5 equal-sized bins based on the 450 cm track length (each bin = 90 cm).

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. `floor(position / 90)` maps 0-89.99 -> 0, 90-179.99 -> 1, etc. Clipping to [0, 4] handles edge cases.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal bins: 0: 0-90cm, 1: 90-180cm, 2: 180-270cm, 3: 270-360cm, 4: 360-450cm.

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. Matches the instruction for "5 equal-sized bins" over the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-level alignment as all behavioral variables -- using the same trial boundary indices (si:ei).

ii.
```python
trial_pos = position[si:ei]
# Same frame indices as neural data
```

iii. Inherent alignment from the NWB frame structure.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral timeseries in the NWB file.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. The NWB lick field contains per-frame lick data (cumulative count per frame).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two processing steps: (1) Binarization: any lick count > 0 is set to 1. (2) Lick sensor error correction: if >30% of frames in a trial have lick count > 2, the entire trial's lick signal is set to 0.

ii.
```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1
# Sensor error correction per trial
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0  # Set to 0 instead of NaN
```

iii. The agent follows the reference code's `correct_lick_sensor_error()` function. The reference uses a threshold of 30-50% for identifying stuck lick sensors and sets values to NaN; the agent uses 30% and sets to 0 instead of NaN for cleaner output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-level alignment as other behavioral variables.

ii.
```python
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. Inherent alignment from the NWB frame structure.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session's scene name (parsed from the NWB `identifier` field), not from the NWB `reward_zone` behavioral timeseries.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The agent parses zone labels (A/B/C) from scene names and maps them to integers (0/1/2). The reward zone coordinates dictionary matches the reference code: A=[80,130], B=[200,250], C=[320,370].

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing extracts the reward zone letter for each trial. For switch sessions, trials 0-29 get the "from" zone and trials 30+ get the "to" zone. The letter is mapped to an integer: A=0, B=1, C=2. The value is constant per trial and broadcast to all timepoints.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
    else:
        labels[:] = zone
    return labels, coords, env_per_trial
```

iii. The switch point at trial 30 is hardcoded, consistent with the paper's experimental protocol.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward/timestamps` and `Reward/data` fields in the NWB file, combined with the `position/timestamps` behavioral timeseries for frame alignment.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
behav_timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
```

iii. Reward events are stored as sparse timestamps in the NWB (not frame-aligned), requiring alignment to imaging frames.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to the nearest imaging frame via `argmin(|behav_timestamps - reward_timestamp|)`. A binary frame-level reward signal is created. Per trial, if any frame within the trial boundaries has a reward event, the trial is marked as rewarded (1), otherwise not rewarded (0). The per-trial value is broadcast to all timepoints.

ii.
```python
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0

trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
```

iii. The agent correctly handles the sparse reward format. The resulting ~84% reward rate matches the paper's ~85% reward rate (with ~15% omission).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Neural/behavior length mismatch**: When the number of neural timepoints differs from behavioral timepoints (common in multi-plane sessions), both are truncated to the shorter length.
- **NaN in neural data**: Replaced with 0.0 via `np.nan_to_num`.
- **Lick sensor errors**: Trials with >30% frames having lick count > 2 have their lick signal set to 0.
- **Teleport/trial count mismatch**: Teleport events are matched to trial starts when counts don't match.
- **Short trials**: Trials with fewer than 2 timepoints are skipped.

ii.
```python
# Length alignment
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    position = position[:min_len]
    ...

# NaN handling
trial_neural = np.nan_to_num(trial_neural, nan=0.0)

# Lick correction
if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
```

iii. The agent documented these issues in CONVERSION_NOTES and handled them with reasonable defaults.

## 13-a. What are the most time-consuming steps of the code?

i. Based on the conversion output (~762.5s total for 152 sessions, ~5s/session average), the main bottleneck is reading NWB files with h5py (loading large arrays from disk). Sessions with more neurons take longer (e.g., m12 sessions with ~1300 neurons take ~4-5s vs m11 sessions with ~200 neurons taking ~0.6-1s). The interneuron correlation computation is vectorized and relatively fast.

ii.
```python
# NWB data loading (I/O bound)
deconv_list.append(f[f'processing/ophys/Deconvolved/{plane}/data'][:])
flu_list.append(f[f'processing/ophys/Fluorescence/{plane}/data'][:])
neu_list.append(f[f'processing/ophys/Neuropil/{plane}/data'][:])
```

iii. The agent noted total processing time of ~762.5 seconds. NWB I/O dominates.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- **Reward timestamp to frame mapping** (line 346-348): iterates over reward timestamps doing argmin per event.
- **Reward per trial** (line 352-356): iterates over trials checking for reward frames.
- **iscell mapping** (line 282): list comprehension building cell mask.

ii.
```python
# Could be vectorized with np.searchsorted:
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0

# Could use np.add.reduceat or vectorized slicing:
for t in range(n_trials):
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
```

iii. These loops are over small arrays (typically ~80 trials, ~70 reward events) so the performance impact is minimal.

## 13-c. What processing does the code repeat multiple times?

i. The code loads both `Fluorescence` and `Neuropil` data AND `Deconvolved` data for every session. The Fluorescence and Neuropil data are only used for interneuron exclusion (computing the simplified dF/F for speed correlation), while the Deconvolved data is used as the neural signal. Loading all three is a significant I/O overhead.

ii.
```python
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
```

iii. The Fluorescence and Neuropil data loading could potentially be avoided if interneuron exclusion used the deconvolved data instead, though this would change the filtering behavior.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `speed` as an output variable (discretized into 5 bins), which is included in the output data. However, examining the instruction's decoder outputs, speed IS listed as a required output, so this is not unnecessary. The code does load fluorescence and neuropil data solely for interneuron exclusion, which requires computing a simplified dF/F -- this intermediate dF/F computation is discarded after filtering. The full dF/F is never used as neural data.

ii.
```python
# These are loaded only for interneuron exclusion:
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
# dff_simple is used only for correlation, then discarded
```

iii. The fluorescence-based dF/F computation is intermediate processing used only for cell filtering. It is not part of the final data output.
