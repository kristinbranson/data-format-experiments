# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files using a glob pattern `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`. Each file is opened with `h5py` (not `pynwb`) and the relevant groups are read directly by HDF5 path. Behavioral data comes from `processing/behavior/BehavioralTimeSeries/`, neural data from `processing/ophys/Deconvolved/`.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
f = h5py.File(nwb_path, 'r')
beh = f['processing']['behavior']['BehavioralTimeSeries']
position = beh['position']['data'][:]
...
deconv_group = f['processing']['ophys']['Deconvolved']
```

iii. The AI's CONVERSION_NOTES document that NWB files are organized as `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb` and that they found 152 NWB files matching. Using h5py directly rather than pynwb is a valid choice for reading HDF5 data.

## 1-b. How are the data split into subjects?

i. Subject ID is extracted from within the NWB file metadata (`general/subject/subject_id`). Unique subjects are collected across all sessions.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
```

iii. The AI notes 11 subjects matching the paper. This approach derives subject IDs from the NWB metadata rather than directory names, but produces the same result.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from within the NWB file.

ii.
```python
sess_id = f['general']['session_id'][()].decode()
```

iii. The AI notes 152 total sessions matching the data.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by `trial_start` (frames where the marker is > 0) and `teleport` (frames where the marker is > 0). All frames where teleport > 0 are used, not just the rising edge.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
n_trials = len(trial_start_inds)

if n_trials != len(teleport_inds):
    print(f"  WARNING: trial_start ({n_trials}) != teleport ({len(teleport_inds)}) count")
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
```

iii. The AI's CONVERSION_NOTES note that trial boundaries come from trial_start and teleport markers. The approach uses all positive teleport frames rather than detecting the rising edge, which is a less robust method but appears to produce the correct number of trials (12,216).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Lick error correction is applied (trials with >35% of samples having lick count > 2 get licks set to NaN, but the trial is NOT removed).

ii.
```python
if n_tp < 2:
    continue
```

iii. The AI's CONVERSION_NOTES mention a minimum of 2 timepoints. No explicit discussion of why this threshold was chosen.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's pre-computed `Deconvolved` data, stored at `processing/ophys/Deconvolved/plane{N}/data`. This is suite2p's own deconvolution of raw fluorescence, NOT the paper's own dF/F and deconvolution pipeline.

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

iii. The AI's CONVERSION_NOTES state: "The NWB files contain pre-computed deconvolved events, so dF/F computation is not needed." However, the Step 1 notes also identify the `dff` function from `preprocessing.py` as a key function. The AI chose to use the pre-computed data instead of replicating the paper's processing.

## 2-b. How is the `neural` data processed?

i. The neural data is filtered by `iscell` (cell classification from suite2p/manual curation) and transposed. No additional processing (no dF/F computation, no neuropil subtraction, no baseline correction, no smoothing, no deconvolution) is applied since the AI uses the pre-computed deconvolved data.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T  # (n_neurons, n_timepoints)
```

iii. The AI's CONVERSION_NOTES state they use the deconvolved events directly. The AI notes that "dF/F computation is not needed" because the NWB already contains deconvolved data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `iscell` filter is applied. Putative interneurons (cells with dF/F-speed correlation > 0.5) are NOT removed.

ii.
```python
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. The AI's CONVERSION_NOTES (Step 5) state: "interneuron filtering skipped (~0.42% effect)." The AI acknowledges this filter exists in the paper but chose to skip it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced using trial start and teleport indices, which aligns it to the start of the trial. No additional temporal shifting is needed.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The instructions specify alignment to "start of the trial," which is achieved by slicing from `trial_start_inds[t]` to `teleport_inds[t]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native frame rate (~15.5 Hz, ~64.5 ms time bin) is used. No temporal rebinning is applied.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
...
'time_bin_size': median_dt * 1000,  # in ms
```

iii. The AI's CONVERSION_NOTES confirm the frame rate of 15.51 Hz and time bin of 64.48 ms, matching the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within the trial and the median time step `dt` (computed from position timestamps).

ii.
```python
dt = np.median(np.diff(pos_timestamps))
...
time_from_start = np.arange(n_tp) * dt
```

iii. The AI does not use the actual timestamps for each frame, but instead reconstructs time from a constant dt. Since the frame rate is constant, this should produce very similar results.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Each timepoint within a trial is assigned `frame_index * dt`, where dt is the median inter-frame interval.

ii.
```python
time_from_start = np.arange(n_tp) * dt
```

iii. This produces time from trial start in seconds, starting at 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data share the same frame indices within each trial, so they are inherently aligned.

ii. Same indexing: `t_start:t_end` for both neural and inputs.

iii. The AI verifies consistent frame rates across behavioral and neural data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env_data = beh['environment']['data'][:]
...
trial_env[t] = int(np.median(env_vals))
```

iii. The environment variable records the environment type (0 or 1) for each timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the median of the environment values (excluding -1) is taken and cast to int. This is a per-trial value broadcast to all timepoints.

ii.
```python
env_vals = env_data[t_start:t_end]
env_vals = env_vals[env_vals >= 0]  # exclude -1
if len(env_vals) > 0:
    trial_env[t] = int(np.median(env_vals))
```

iii. The AI filters out -1 values and uses the median, which is a robust way to get the per-trial environment type.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index (loop counter `t`).

ii.
```python
trial_num = float(t)
```

iii. Sequential 0-indexed trial number within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the loop index is used directly. The value is constant across all timepoints within a trial.

ii.
```python
input_data[2, :] = trial_num
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps. Reward timestamps are mapped to frame indices using `searchsorted`, and a per-trial binary reward array is computed.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
...
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
```

iii. The AI uses the Reward timestamps from the NWB file to determine whether each trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For the first trial, previous outcome is 0. For subsequent trials, it is the reward outcome of the previous trial.

ii.
```python
if t == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(trial_rewarded[t - 1])
```

iii. Binary (0 = omitted, 1 = rewarded) from the previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. The reward zone is determined by a simple nearest-zone matching: the minimum position where `reward_zone > 0` is compared against known zone ranges with a ±20 cm tolerance.

ii.
```python
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None
...
rz_start_pos = trial_pos[rz_active].min()
rz_labels[t] = determine_reward_zone_label(rz_start_pos)
```

iii. The AI identifies reward zone boundaries from the paper (A=[80,130], B=[200,250], C=[320,370]) and uses a tolerance-based matching to assign each trial's reward zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from the animal's position to the nearest edge of the reward zone (defined as `[rz_start, rz_start + 50]`). Distance is negative before the zone, 0 inside, and positive after.

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

iii. This follows the paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit conditional assignments.

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

iii. Matches the 7 bins specified in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices within each trial, so no additional alignment is needed.

ii. Same `t_start:t_end` slicing for position and neural data.

iii. Inherently aligned through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[t_start:t_end]
```

iii. The `position` variable records the animal's position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins spanning the 450 cm track using `np.digitize` with edges from `np.linspace(0, 450, 6)`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. Produces 5 bins of 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Same as 8-b: 5 equal bins using `np.digitize` with edges [90, 180, 270, 360, 450], clipped to [0, 4].

ii. See 8-b.

iii. The bin edges at 90, 180, 270, 360 match the 450/5 = 90 cm per bin specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii. Same `t_start:t_end` slicing.

iii. Inherently aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_data = beh['lick']['data'][:]
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick error correction (if >35% of trial samples have lick count > 2, set to NaN), clips to [0,1], then applies Gaussian smoothing with sigma=2, and finally thresholds at > 0.5 to produce binary output.

ii.
```python
def process_licks(lick_data, trial_start_inds, teleport_inds):
    licks = np.copy(lick_data).astype(np.float64)
    for t in range(len(trial_start_inds)):
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

iii. The AI applies the paper's lick error correction and smoothing pipeline, which is referenced in the code's glmUtils.py. However, the instruction simply asks for binary lick (0 = no, 1 = yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii. Same `t_start:t_end` slicing.

iii. Inherently aligned.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series. For each trial, the minimum position where `reward_zone > 0` is found, and a nearest-zone matching with ±20 cm tolerance assigns the zone label (A, B, or C).

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
    return rz_starts, rz_labels
```

iii. The AI uses the position where the reward zone is active to determine which of the three reward zones (A, B, C) is being used.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the reward zone label is determined by nearest-zone matching of the minimum position where `reward_zone > 0`. For omission trials (no reward zone activation), the label is inferred from the nearest neighboring trial that has a label. The label is mapped to 0=A, 1=B, 2=C.

ii.
```python
def get_reward_zone_for_trial(rz_labels, trial_idx):
    if rz_labels[trial_idx] is not None:
        return rz_labels[trial_idx]
    for offset in range(1, len(rz_labels)):
        if trial_idx - offset >= 0 and rz_labels[trial_idx - offset] is not None:
            return rz_labels[trial_idx - offset]
        if trial_idx + offset < len(rz_labels) and rz_labels[trial_idx + offset] is not None:
            return rz_labels[trial_idx + offset]
    return 'A'  # fallback
```

iii. The AI infers missing labels from neighbors, while the reference uses a more sophisticated Viterbi algorithm.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps. Reward timestamps are mapped to frame indices, and a binary per-trial reward flag is computed.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
...
reward_in_trial = np.any(
    (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
)
trial_rewarded[t] = 1 if reward_in_trial else 0
```

iii. The Reward time series records reward delivery events with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices via `searchsorted`. For each trial, the output is 1 if any reward event fell within the trial boundaries, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
...
trial_rewarded[t] = 1 if reward_in_trial else 0
...
output_data[5, :] = reward_out
```

iii. Binary per-trial reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Trial start/teleport count mismatch**: If counts differ, truncate to the minimum.
- **Lick sensor errors**: Trials with >35% high lick counts get licks set to NaN (but the trial is kept).
- **Missing reward zone data**: For omission trials where `reward_zone` is never active, the label is inferred from the nearest neighboring trial.
- **Very short trials**: Trials with < 2 timepoints are skipped.

ii.
```python
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
...
if frac_high > LICK_ERROR_THRESHOLD:
    licks[t_start:t_end] = np.nan
...
if n_tp < 2:
    continue
```

iii. These are defensive checks for data quality issues discovered during exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py and reading large arrays (I/O bound)
2. Processing all 152 sessions sequentially
3. Saving the pickle file

ii. N/A

iii. The AI reports total processing time of ~70s for all sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `process_session` iterate over each trial sequentially for reward zone detection, lick processing, reward computation, and environment extraction. Some of these (e.g., reward zone start detection) could potentially be vectorized.

ii. N/A

iii. The variable trial lengths make full vectorization awkward.

## 13-c. What processing does the code repeat multiple times?

i. No significant repeated processing. The AI processes each NWB file only once (unlike the reference which has a separate survey step that loads all files first).

ii. N/A

iii. The AI's approach is more efficient in this regard.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies lick smoothing (Gaussian with sigma=2) and thresholding at 0.5, which adds processing complexity beyond simple binarization. The lick error correction per the paper's code is also applied, though for a simple binary lick output it may be unnecessary.

ii.
```python
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
```

iii. The downstream decoder uses binary lick values; the smoothing may not add value.
