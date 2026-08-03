# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by listing `data/` for `sub-*` directories and globbing `*.nwb` files within each. It loads each NWB file using `h5py` (not `pynwb`), directly accessing HDF5 groups/datasets. All 152 sessions across 11 subjects are processed.

ii.
```python
def find_nwb_files(data_dir='data'):
    """Find all NWB files organized by subject."""
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

```python
with h5py.File(filepath, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    ...
```

iii. The AI noted 11 subjects and 152 sessions matching the paper. Using `h5py` directly instead of `pynwb` was a deliberate choice for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from subdirectory names (`sub-{id}`) and from the NWB file's `general/subject/subject_id` field. Unique subjects are collected across all processed sessions and sorted.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The AI confirmed 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All `.nwb` files within each subject directory are processed as separate sessions.

ii.
```python
files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
for fpath in files:
    sessions.append({...})
```

iii. The AI found 152 total sessions (matching the paper's count of 10*14 + 1*12).

## 1-d. How are the data split into trials?

i. Trial boundaries are found using `trial_start` (positive values mark trial starts) and `teleport` (positive values mark trial ends). The AI finds all teleport-positive frames and matches each to the nearest following trial start.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
n_trials = len(trial_start_inds)

if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)
```

iii. The AI identified trial_start and teleport as trial boundary markers from the behavior data.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Sessions with fewer than 2 valid trials are also skipped.

ii.
```python
if n_tp < 2:
    continue
...
if result['n_trials'] < 2:
    print(f"  Skipping: fewer than 2 valid trials")
    continue
```

iii. The AI used a minimal threshold of 2 timepoints for trial validity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field in the `processing/ophys` group. Data from multiple planes is concatenated.

ii.
```python
deconv_group = f['processing/ophys/Deconvolved']
planes = sorted(deconv_group.keys())
for plane in planes:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    deconv_list.append(d)
deconv = np.concatenate(deconv_list, axis=1)
```

iii. The AI noted the paper trains decoders on deconvolved events and that NWB files contain already-processed deconvolved data.

## 2-b. How is the `neural` data processed?

i. In addition to combining multi-plane data, the AI applies two filtering steps: (1) `iscell` filter, and (2) interneuron exclusion based on speed-dFF correlation > 0.5. The interneuron exclusion computes neuropil-corrected dF/F (`F - 0.7*Fneu`), then correlates with speed, excluding cells with correlation > 0.5. Data is also cast to float32.

ii.
```python
# Step 1: iscell filter
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

# Step 2: Interneuron exclusion
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
f_median[f_median == 0] = 1
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
corrs = (speed_z @ dff_z) / len(speed_z)
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True

final_cell_mask = cell_mask_concat & ~interneuron_mask
neural_all = deconv[:, final_cell_mask].T
neural_all = neural_all.astype(np.float32)
```

iii. The AI cited reference code mentioning interneuron exclusion based on speed-dFF correlation and neuropil subtraction coefficient of 0.7. The CONVERSION_NOTES.md mentions "Suite2p iscell + interneuron exclusion (speed-dFF corr > 0.5)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered based on both `iscell` (Suite2p cell classification) and interneuron exclusion (speed-dFF correlation > 0.5). NaN values in neural data are replaced with 0.

ii.
```python
final_cell_mask = cell_mask_concat & ~interneuron_mask
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The AI noted the paper describes quality metrics and interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced from trial_start_ind to teleport_ind for each trial. Since alignment is to trial start and the data is already co-registered between neural and behavior, no additional temporal alignment is needed.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
```

iii. The AI verified that behavior and neural data share the same frame indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is computed as `1000.0 / IMAGING_RATE_NOMINAL` where `IMAGING_RATE_NOMINAL = 15.5078125 Hz`, giving ~64.48 ms per frame. No rebinning is applied.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
```

iii. The AI uses a hardcoded imaging rate constant rather than reading it from each session's data.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the imaging rate constant. Time is computed as frame index multiplied by frame duration (`1/imaging_rate`).

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The AI used the imaging rate from the NWB file's `general/optophysiology/ImagingPlane/imaging_rate` field.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed by multiplying frame indices (0, 1, 2, ...) by the frame duration. This gives evenly-spaced time values starting at 0.

ii.
```python
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. This approach assumes perfectly regular frame timing, rather than using recorded timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Since both are derived from the same frame indices, they are inherently aligned.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Both neural and time-from-start use the same `si:ei` frame range.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from parsing the NWB file's `identifier` field (scene name). The scene name encodes the environment (Env1/Env2). For cross-environment switches, the environment changes at trial 30.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_env = env_per_trial.copy()
```

```python
def _parse_zone_and_env(parts):
    for p in parts:
        if p.startswith('Env'):
            env_num = int(p.replace('Env', ''))
            env = env_num - 1  # 0-indexed
    return zone, env
```

iii. The AI chose to parse scene names rather than reading the `environment` behavior time series, citing it as "more reliable for cross-env switches."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene name parsing extracts the environment number. For switch sessions (containing `_to_`), the environment changes at trial 30 (hardcoded `change_trial=30`).

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        from_zone, from_env = _parse_zone_and_env(from_parts)
        to_zone, to_env = _parse_zone_and_env(to_parts)
        env_per_trial[:change_trial] = from_env
        env_per_trial[change_trial:] = to_env
    else:
        env_per_trial[:] = env
```

iii. The AI noted the paper describes switches occurring at trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-based loop index over trials within each session.

ii.
```python
for t in range(n_trials):
    trial_num = np.float32(t)
    ...
    np.full((1, n_tp), trial_num, dtype=np.float32),
```

iii. The AI uses sequential trial indexing within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial, broadcast to shape (1, n_tp).

ii.
```python
trial_num = np.float32(t)
np.full((1, n_tp), trial_num, dtype=np.float32),
```

iii. Sequential numbering within session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the behavior data. Reward event timestamps are matched to behavior frame timestamps using `np.argmin(np.abs(...))` to create a frame-aligned binary reward signal.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. The AI noted reward events have separate sparse timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward event occurred in the previous trial. Trial 0 gets 0 (no previous trial).

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1

prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. The AI follows the instruction specification of binary previous trial outcome (omitted=0, rewarded=1).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone coordinates. The reward zone for each trial is determined by parsing the NWB `identifier` (scene name) rather than from the `reward_zone` behavior variable.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI determined reward zones from the session scene name, with zones A=[80,130], B=[200,250], C=[320,370] matching the reference code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if position is before the zone start, positive if after the zone end, 0 if inside the zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[after] = position[after] - rz_end       # positive
    return dist
```

iii. This matches the signed distance concept from the paper.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean conditions:

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

iii. The bin boundaries match the instruction specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices (si:ei), so no additional alignment is needed.

ii.
```python
trial_pos = position[si:ei]
trial_neural = neural_all[:, si:ei].copy()
```

iii. Same frame indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = position[si:ei]
```

iii. Direct use of position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond discretization.

ii.
```python
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The AI chose 90 cm bins based on a track length of 450 cm (450/5 = 90).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
trial_pos = position[si:ei]
```

iii. Same frame indexing as neural.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Direct use of lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are binarized (>0 -> 1). Additionally, the AI applies lick sensor error correction: if >30% of frames in a trial have lick count > 2, the entire trial's lick is set to 0.

ii.
```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1

for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0
```

iii. The AI cited the reference code's `correct_lick_sensor_error()` function from behavior.py.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. Same frame indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from parsing the NWB file's `identifier` (scene name). The scene name encodes the reward zone location (A, B, or C) and whether it switches mid-session.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The AI parsed scene names like `Env1_LocationB_to_A` to extract reward zone labels, with switches at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing determines the zone label per trial. For switch sessions, the zone changes at trial 30. Zone labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
    else:
        labels[:] = zone
```

iii. The AI handled single-zone, within-env switch, and cross-env switch session formats.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the behavior data.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Sparse reward event timestamps are matched to behavior frame times.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event occurred within the trial time range [trial_start, teleport]. Binary per-trial value: 0=no reward, 1=reward.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
rew_outcome = int(trial_rewarded[t])
```

iii. Binary reward outcome per trial, matching the instruction specification.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Truncated to the shorter array length.
- **Short trials**: Trials with < 2 timepoints skipped.
- **Teleport/trial count mismatch**: Teleports are matched to trial starts by finding the first teleport after each trial start.
- **NaN in neural data**: Replaced with 0.
- **Lick sensor errors**: Trials with >30% high-count frames are zeroed out.

ii.
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    ...

trial_neural = np.nan_to_num(trial_neural, nan=0.0)

if n_tp < 2:
    continue
```

iii. Documented in CONVERSION_NOTES.md as edge cases found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with `h5py` and reading large arrays (neural data, fluorescence, neuropil)
2. Computing interneuron exclusion (dFF correlation with speed for all cells)
3. Matching reward timestamps frame-by-frame using `np.argmin`
4. Saving the large pickle file

ii. N/A

iii. Total processing time reported as 762.5 seconds for all 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The reward timestamp matching loop iterates over each reward event using `np.argmin(np.abs(behav_timestamps - rt))` which is O(n_rewards * n_timepoints). This could be vectorized with `np.searchsorted`. The per-trial loop for trial segmentation could potentially be partially vectorized.

ii.
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. The AI did not note this inefficiency.

## 13-c. What processing does the code repeat multiple times?

i. The code loads fluorescence and neuropil data in addition to deconvolved data, even though only the deconvolved data is used for the final neural output. The fluorescence and neuropil are only used for the interneuron exclusion step.

ii.
```python
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
```

iii. These extra data loads are needed for the interneuron exclusion step but are not needed if that step is removed.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron exclusion step loads fluorescence and neuropil data, computes dF/F, and correlates with speed. This processing is extra relative to the reference solution. The lick sensor error correction is also additional processing not in the reference.

ii.
```python
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
```

iii. The interneuron exclusion adds significant I/O and computation time per session.
