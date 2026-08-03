# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb')` to find all NWB files. It loads them using `h5py.File()` directly (not pynwb). Each NWB file is processed in `process_session()` which extracts neural, behavioral, and metadata fields from the HDF5 structure.

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

iii. The AI identified all NWB files via glob pattern matching. Using h5py instead of pynwb is a valid choice that gives direct access to the same underlying HDF5 data.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the session results after processing. The subject ID is read from each NWB file's metadata (`general/subject/subject_id`). Unique subjects are collected and sorted.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
```

iii. Subject IDs come directly from the NWB metadata rather than directory names. Both approaches yield the same subject set.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from the NWB metadata (`general/session_id`).

ii.
```python
sess_id = f['general']['session_id'][()].decode()
```

iii. One file = one session, consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trials are split using `trial_start_markers > 0` for trial starts and `teleport_markers > 0` for trial ends.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
n_trials = len(trial_start_inds)
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
```

iii. The AI uses `np.where(teleport_markers > 0)` which finds ALL timepoints where teleport is positive, rather than detecting rising edges. The reference uses `(teleport[1:] > 0) & (teleport[:-1] <= 0)` to find only the onset of each teleport event. If teleport stays >0 for multiple frames, the AI's approach would yield incorrect trial boundaries. However, the AI reports correct trial counts (12,216), suggesting this may work in practice if teleport is only 1 frame long.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are excluded.

ii.
```python
n_tp = t_end - t_start
if n_tp < 2:
    continue
```

iii. The AI uses a minimal threshold of 2 timepoints. The reference uses `min_ntimepoints=50` (though the reference has a bug where it checks `idx.sum()` which is the sum of index values, not the count, so it effectively never filters).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` data in the ophys processing module.

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

iii. Matches the paper's use of deconvolved calcium events.

## 2-b. How is the `neural` data processed?

i. For multi-plane animals, data from multiple planes is concatenated along the ROI axis. Data is cast to float32. Cells are filtered by the iscell flag.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T  # (n_neurons, n_timepoints)
```

iii. The AI reads iscell from a single `PlaneSegmentation` table. For multi-plane animals, the reference reads iscell separately for each plane's ROI table, filtering per-plane before concatenation. The AI's approach assumes a single iscell array covers all planes, which may produce incorrect filtering for multi-plane animals (m17, m18).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Filtered by `iscell[:, 0] == 1` from the ImageSegmentation PlaneSegmentation table.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. The iscell flag reflects Suite2p classification plus manual curation. The AI reads it from a single location rather than per-plane. No additional quality filtering (e.g., interneuron exclusion) is applied, consistent with the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced from `t_start` to `t_end` (trial start to teleport), so it is aligned to trial start. No additional temporal shifting is needed.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. Since alignment is to trial start, simply slicing from trial start index is correct.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native frame rate (~15.5 Hz, ~64.5 ms bins) is preserved. The time bin size is computed from the median timestamp difference.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
...
'time_bin_size': median_dt * 1000,  # in ms
```

iii. Consistent with the reference approach of using the native imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame rate (dt computed from position timestamps), not from the actual timestamps themselves.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
...
time_from_start = np.arange(n_tp) * dt
```

iii. The AI reconstructs time from `arange * dt` rather than using actual timestamps subtracted from trial start. The reference uses `timestamps_curr - timestamps_curr[0]`. These should be very similar if the frame rate is constant, but the reference approach is more accurate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as `np.arange(n_tp) * dt` where `dt` is the median inter-frame interval.

ii.
```python
time_from_start = np.arange(n_tp) * dt
input_data[0, :] = time_from_start
```

iii. Simple reconstruction from frame count and dt. The reference subtracts the first timestamp of each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices, so time_from_start at index i corresponds to neural data at index i.

ii.
```python
trial_neural = neural_all[:, t_start:t_end]
time_from_start = np.arange(n_tp) * dt
```

iii. Both are indexed by the same frame range [t_start:t_end], ensuring alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env_data = beh['environment']['data'][:]
...
trial_env[t] = int(np.median(env_vals))
```

iii. The environment variable is binary (0 or 1) per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, environment values within the trial are extracted, negative values (-1) are excluded, and the median of remaining values is taken. The result is cast to int.

ii.
```python
env_vals = env_data[t_start:t_end]
env_vals = env_vals[env_vals >= 0]
if len(env_vals) > 0:
    trial_env[t] = int(np.median(env_vals))
```

iii. The reference simply uses the raw environment values directly within the trial slice (`vr_environment[idx].astype(int)`). The AI adds extra processing (filtering negatives, taking median) which is more defensive but unnecessary if the data is clean within trial boundaries.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index from the loop counter.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = float(t)
    input_data[2, :] = trial_num
```

iii. Sequential index starting from 0.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond converting the loop index to float. The value is constant across all timepoints within a trial.

ii.
```python
trial_num = float(t)
input_data[2, :] = trial_num
```

iii. Same as the reference approach.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward timestamps are mapped to frame indices via `searchsorted`.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
```

iii. Same source variable as the reference.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A per-trial reward binary is computed first. For each trial, it checks if any reward frame index falls within [t_start, t_end). Previous trial outcome for trial t is `trial_rewarded[t-1]`; for trial 0, it is 0.

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

iii. Functionally equivalent to the reference approach. The reference creates an isreward array at every timepoint and checks with `np.any(isreward[prev_trial_idx])`, while the AI precomputes per-trial reward status.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the reward zone label for the current trial. Reward zone labels are inferred from `reward_zone` and `position` data using tolerance-based matching and nearest-neighbor interpolation for missing trials.

ii.
```python
rz_starts, rz_labels = get_reward_zone_start_per_trial(position, rzone, trial_start_inds, teleport_inds)
...
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None
```

iii. The reference uses a Viterbi algorithm for reward zone assignment, which is more robust. The AI's tolerance-based approach with nearest-neighbor fallback is simpler but could misassign edge cases.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to reward zone boundaries [rz_start, rz_start+50]. Negative before zone, 0 inside, positive after. Reward zone start comes from `REWARD_ZONES[label][0]`.

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

iii. Functionally equivalent to the reference's `compute_distance_to_reward_zone`. Both use the same zone boundaries since the AI ultimately maps to `REWARD_ZONES[label][0]` which matches `reward_zone_dict`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic matching the specification.

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

iii. The reference uses `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The AI's explicit conditional approach is functionally equivalent, with minor boundary differences: the AI uses `distance == 0` for bin 3 and `distance > 0` for bin 4, while the reference uses `1e-6` as the boundary between bins 3 and 4. Both should produce the same results for practical purposes since distance == 0 is the "in zone" category.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices [t_start:t_end] are used for both neural and position data, ensuring alignment.

ii.
```python
trial_pos = position[t_start:t_end]
trial_neural = neural_all[:, t_start:t_end]
```

iii. Consistent indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[t_start:t_end]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins using `np.linspace(0, TRACK_LENGTH, 6)` = [0, 90, 180, 270, 360, 450].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The AI uses bin edges [0, 90, 180, 270, 360, 450]. The reference uses `[-inf, 50, 150, 250, 350, inf]`. These are substantially different bin boundaries, producing different position bin assignments.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal-sized bins spanning [0, 450]: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)  # [0, 90, 180, 270, 360, 450]
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The reference uses bins [-inf to 50), [50-150), [150-250), [250-350), [350-inf), which are 100cm-wide bins centered differently. The AI's 90cm-wide bins are different. The instructions say "5 equal-sized bins" and the AI interpreted "equal-sized" as equal divisions of [0, 450] while the reference used different boundaries.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_pos = position[t_start:t_end]
```

iii. Consistent indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_data = beh['lick']['data'][:]
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies extensive processing: (1) lick sensor error correction (if >35% of samples have cumulative lick count >2, set trial licks to NaN), (2) clip to [0,1], (3) Gaussian smoothing with sigma=2, (4) threshold at >0.5 for binary output. NaN values are set to 0.

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

iii. The reference uses simple binarization: `(licks_curr > 0).astype(int)`. The AI adds lick error correction, clipping, smoothing, and a higher threshold (0.5 vs 0), following the reference paper's analysis code more closely but deviating from the reference conversion code. This will produce different lick outputs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_licks = licks_processed[t_start:t_end]
```

iii. Consistent indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavior time series. The minimum position where `reward_zone > 0` is computed per trial, then matched to known zone boundaries with 20cm tolerance. Missing trials use nearest-neighbor interpolation.

ii.
```python
def get_reward_zone_start_per_trial(pos, rzone, trial_start_inds, teleport_inds):
    for t in range(n_trials):
        rz_active = trial_rzone > 0
        if np.any(rz_active):
            rz_start_pos = trial_pos[rz_active].min()
            rz_labels[t] = determine_reward_zone_label(rz_start_pos)
...
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
```

iii. The reference uses a Viterbi algorithm to assign reward zone labels, which is more robust for noisy data and handles transitions between zones better.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per-trial: find min position where reward_zone > 0, match to A/B/C via tolerance, use nearest neighbor for omission trials. Encode as 0=A, 1=B, 2=C.

ii.
```python
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
trial_rz_label[t] = rz_label_map.get(label, 0)
output_data[4, :] = rz_loc
```

iii. The reference uses Viterbi decoding which provides temporal smoothing and is more robust. The AI's approach may produce different assignments for edge cases.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. Same source as the reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are converted to frame indices. For each trial, check if any reward frame falls within [t_start, t_end). Binary output: 1 if rewarded, 0 otherwise.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
output_data[5, :] = reward_out
```

iii. Functionally equivalent to the reference approach.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Trial start / teleport count mismatch**: Truncate to minimum count.
- **Short trials**: Skip trials with < 2 timepoints.
- **Missing reward zone**: Use nearest-neighbor interpolation from neighboring trials.
- **Lick sensor errors**: Trials with >35% high lick counts have licks set to NaN, then smoothed.
- **Reward frame index clipping**: Clipped to valid range.

ii.
```python
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
...
if n_tp < 2:
    continue
...
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
```

iii. The reference handles neural/behavior length mismatch by cropping to minimum, and uses the Viterbi algorithm for missing reward zone data. The AI's handling is generally reasonable but differs in specifics.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py and reading large arrays
2. Processing each session sequentially
3. Saving the pickle file

ii. N/A

iii. The AI code processes sessions in a single loop without a separate survey step, so it only loads each file once (unlike the reference which loads twice - once for survey, once for conversion).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial reward computation loops over all trials with `searchsorted` checks. The lick error detection loops over trials. The reward zone label computation loops over trials with nearest-neighbor search. These could potentially be vectorized.

ii. N/A

iii. The per-trial loops are the natural structure for variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The AI code does NOT repeat file loading (unlike the reference which has a separate survey step that loads all files first). However, reward zone start positions are computed separately from reward zone labels, involving redundant iteration.

ii. N/A

iii. The AI is more efficient than the reference in this regard by avoiding a separate survey pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The lick error correction and smoothing pipeline (nansmooth with sigma=2, clipping, threshold at 0.5) is more complex than needed for the binary lick output. The reference simply binarizes at >0. The speed is also processed with `np.abs()` which may not match the raw data semantics.

ii.
```python
licks_processed = process_licks(lick_data, trial_start_inds, teleport_inds)
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
...
speed_bins = discretize_speed(np.abs(trial_speed))
```

iii. The extra lick processing comes from the reference analysis code but is not needed for the decoder format conversion. Taking absolute value of speed is an extra step not in the reference.
