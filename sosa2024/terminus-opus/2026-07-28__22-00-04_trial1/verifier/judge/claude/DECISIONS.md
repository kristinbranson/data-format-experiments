# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb')` to find all NWB files. Files are loaded using `h5py` (not `pynwb`). Each NWB file is processed as a session.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
f = h5py.File(nwb_path, 'r')
```

iii. The AI chose h5py for direct HDF5 access rather than pynwb. The glob pattern matches all NWB files in the data directory structure.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file metadata (`general/subject/subject_id`). Unique subjects are collected across all processed sessions.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
```

iii. Subject IDs are read directly from NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files are found via glob and processed individually.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, ...)
```

iii. One file per session is the standard NWB organization.

## 1-d. How are the data split into trials?

i. Trial starts are identified where `trial_start` > 0. Trial ends are identified where `teleport` > 0. The AI uses `np.where(teleport_markers > 0)[0]` for teleport indices.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
```

iii. The AI identified trial_start and teleport as the key markers. However, the teleport detection differs from the reference which finds the *onset* of teleport (transition from <=0 to >0).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
if n_tp < 2:
    continue
...
if len(result['neural_trials']) < 2:
    print(f"  SKIPPING: fewer than 2 trials")
    continue
```

iii. Minimal filtering. The reference uses a threshold of 50 timepoints instead.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field in `processing/ophys`.

ii.
```python
deconv_group = f['processing']['ophys']['Deconvolved']
plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
if len(plane_keys) == 1:
    deconv_data = deconv_group[plane_keys[0]]['data'][:]
else:
    plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
    deconv_data = np.concatenate(plane_data, axis=1)
```

iii. The paper trains decoders on deconvolved calcium events.

## 2-b. How is the `neural` data processed?

i. Multi-plane data is concatenated along the ROI axis. Cells are filtered by `iscell`. Data is cast to float32. No additional processing (no normalization, no smoothing).

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T  # (n_neurons, n_timepoints)
...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The AI reads a single iscell array from `ImageSegmentation/PlaneSegmentation`. For multi-plane sessions, this differs from the reference which reads iscell per plane from the ROI table.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Filtered by `iscell[:, 0] == 1` only.

ii.
```python
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. Uses the NWB-stored iscell flag which includes Suite2p + manual curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced from `trial_start_inds[t]` to `teleport_inds[t]`, which aligns to trial start.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. Since alignment is to trial start, no additional temporal shifting is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native frame rate (~64.5 ms, ~15.5 Hz) is preserved. No temporal rebinning is applied.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
...
'time_bin_size': median_dt * 1000,  # in ms
```

iii. The data is kept at the stored frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index and the median timestep `dt` computed from `position` timestamps.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
...
time_from_start = np.arange(n_tp) * dt
```

iii. Uses computed dt rather than actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `np.arange(n_tp) * dt` - multiply frame index by median dt. This differs from the reference which uses actual timestamps minus the first timestamp.

ii.
```python
time_from_start = np.arange(n_tp) * dt
```

iii. Assumes uniform frame spacing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same frame indices (t_start to t_end), so they are inherently aligned.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
...
time_from_start = np.arange(n_tp) * dt
```

iii. Same indexing ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the `environment` behavioral time series.

ii.
```python
env_data = beh['environment']['data'][:]
...
trial_env[t] = int(np.median(env_vals))
```

iii. The environment variable is read from the NWB file.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, environment values are extracted, values < 0 are excluded, and the median of remaining values is taken. The value is then broadcast across all timepoints.

ii.
```python
env_vals = env_data[t_start:t_end]
env_vals = env_vals[env_vals >= 0]  # exclude -1
if len(env_vals) > 0:
    trial_env[t] = int(np.median(env_vals))
```

iii. Takes median to handle any within-trial variation. Reference simply takes the raw values without median aggregation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `t` over trials within the session.

ii.
```python
trial_num = float(t)
```

iii. Sequential index starting from 0.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing, just the loop counter. Broadcast as a constant across all timepoints.

ii.
```python
input_data[2, :] = trial_num
```

iii. Matches the reference approach.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward` timestamps. Reward timestamps are converted to frame indices using `searchsorted`.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
```

iii. Uses reward event timestamps to determine which trials had rewards.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward frame index falls within the trial's time range. For trial 0, set to 0. For subsequent trials, use the previous trial's reward status.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
...
if t == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(trial_rewarded[t - 1])
```

iii. Consistent with the reference approach.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and the reward zone start position. Reward zone is determined from the `reward_zone` behavioral time series using `determine_reward_zone_label()` which matches positions to known zone ranges with tolerance.

ii.
```python
rz_starts, rz_labels = get_reward_zone_start_per_trial(
    position, rzone, trial_start_inds, teleport_inds)
...
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None
```

iii. Uses a simple tolerance-based matching rather than the Viterbi algorithm used in the reference.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to nearest edge of reward zone `[rz_start, rz_start + 50]`. Zero when inside the zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start):
    rz_end = rz_start + 50.0
    distance = np.zeros_like(position)
    before_mask = position < rz_start
    distance[before_mask] = position[before_mask] - rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    distance[in_mask] = 0.0
    after_mask = position > rz_end
    distance[after_mask] = position[after_mask] - rz_end
    return distance
```

iii. Same logic as reference. However, a key difference: the AI uses the actual observed rz_start position from the data rather than the fixed reward_zone_dict boundaries used in the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean masking matching the instruction specification.

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

iii. Matches the instruction bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
trial_pos = position[t_start:t_end]
...
distance = compute_distance_to_reward_zone(trial_pos, rz_start)
```

iii. Same indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. From the `position` behavioral time series.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[t_start:t_end]
```

iii. Direct position data from NWB.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extraction and discretization.

ii.
```python
pos_bins = discretize_position(trial_pos, n_bins=5)
```

iii. Raw position values are discretized directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins using `np.linspace(0, 450, 6)` = [0, 90, 180, 270, 360, 450].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The AI uses bins [0-90, 90-180, 180-270, 270-360, 360-450]. The reference uses [-inf, 50, 150, 250, 350, inf] giving bins [-inf to 50, 50-150, 150-250, 250-350, 350-inf]. These are very different bin edges.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_pos = position[t_start:t_end]
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behavioral time series.

ii.
```python
lick_data = beh['lick']['data'][:]
```

iii. Direct lick data from NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick error correction (if >35% of samples have count >2, set trial to NaN), clips to [0,1], applies Gaussian smoothing with sigma=2, then thresholds at >0.5 for binary output.

ii.
```python
def process_licks(lick_data, trial_start_inds, teleport_inds):
    licks = np.copy(lick_data).astype(np.float64)
    for t in range(len(trial_start_inds)):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        trial_licks = licks[t_start:t_end]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > LICK_ERROR_THRESHOLD:
                licks[t_start:t_end] = np.nan
    licks[licks > 1] = 1
    return licks
...
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
```

iii. The AI follows the reference code's lick processing pipeline. The reference solution simply does `(licks_curr > 0).astype(int)` without error correction or smoothing.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_licks = licks_processed[t_start:t_end]
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavioral time series and `position`. For each trial, the minimum position where reward_zone > 0 is found, and matched to known zone ranges (A/B/C) with tolerance.

ii.
```python
def get_reward_zone_start_per_trial(pos, rzone, trial_start_inds, teleport_inds):
    for t in range(n_trials):
        trial_rzone = rzone[t_start:t_end]
        trial_pos = pos[t_start:t_end]
        rz_active = trial_rzone > 0
        if np.any(rz_active):
            rz_start_pos = trial_pos[rz_active].min()
            rz_labels[t] = determine_reward_zone_label(rz_start_pos)
```

iii. Simpler approach than the reference's Viterbi algorithm.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, find the minimum position where reward_zone > 0. Match to nearest zone within 20cm tolerance. For omission trials (no active reward zone), infer from nearest neighboring trial.

ii.
```python
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None

def get_reward_zone_for_trial(rz_labels, trial_idx):
    if rz_labels[trial_idx] is not None:
        return rz_labels[trial_idx]
    for offset in range(1, len(rz_labels)):
        if trial_idx - offset >= 0 and rz_labels[trial_idx - offset] is not None:
            return rz_labels[trial_idx - offset]
        ...
```

iii. The reference uses Viterbi algorithm for more robust assignment. The AI's approach is simpler but may be less robust for noisy data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. Uses reward event timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward frame index falls within [trial_start, teleport). Binary: 1 if rewarded, 0 otherwise.

ii.
```python
reward_in_trial = np.any(
    (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
)
trial_rewarded[t] = 1 if reward_in_trial else 0
...
output_data[5, :] = reward_out
```

iii. Consistent with reference approach.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Trial start/teleport count mismatch**: Truncate to the minimum count.
- **Short trials**: Skip trials with < 2 timepoints.
- **Few trials per session**: Skip sessions with < 2 valid trials.
- **Lick sensor errors**: Trials with >35% high-count lick samples set to NaN.
- **Missing reward zone**: Infer from nearest neighboring trial.

ii.
```python
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
...
if n_tp < 2:
    continue
...
if len(result['neural_trials']) < 2:
    continue
```

iii. The reference handles neural/behavior length mismatches by cropping, which the AI does not explicitly do. The AI has more aggressive lick processing from the reference code.

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files via h5py and reading large neural data arrays. The AI reports total conversion time and per-session timing.

ii.
```python
t0 = time.time()
f = h5py.File(nwb_path, 'r')
...
elapsed = time.time() - t0
print(f"  ... {elapsed:.1f}s")
```

iii. I/O bound on reading large HDF5 files.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` (lines 300-380) iterates over trials sequentially. Some operations like reward zone determination and environment extraction could be vectorized. The lick error correction loop could also be vectorized.

ii. N/A

iii. Variable-length trials make full vectorization difficult.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file only once (no separate survey step), so there is no redundant file loading. However, reward zone label determination involves iterating through trials twice (once for rz_starts/labels, once for assigning labels to omission trials).

ii. N/A

iii. The AI's approach is more efficient than the reference which has a separate survey step that loads all files twice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies lick error correction and Gaussian smoothing from the reference code. The reference solution does not apply these processing steps, using raw lick binarization instead. The smoothing changes the lick values and may not be appropriate for the decoder task.

ii.
```python
licks_processed = process_licks(lick_data, trial_start_inds, teleport_inds)
...
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
```

iii. The lick smoothing is borrowed from the reference analysis code but may not be needed for the decoder format.
