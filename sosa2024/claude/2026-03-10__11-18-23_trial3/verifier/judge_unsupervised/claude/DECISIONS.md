# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from the `data/` directory using h5py. It scans for subject directories matching the pattern `sub-*`, then finds all `.nwb` files within each. Each NWB file is opened with `h5py.File()` and behavioral timeseries, neural data (Fluorescence, Neuropil, Deconvolved), and metadata are extracted. The function `get_all_nwb_files()` builds a list of file info dicts, then `build_dataset()` iterates over them calling `process_session()` for each.

ii.
```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj.replace('sub-', ''),
                'filepath': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file,
            })
    return all_files
```

```python
with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    position = behav['position']['data'][:]
    speed = behav['speed']['data'][:]
    lick = behav['lick']['data'][:]
    # ... etc
    ophys = f['processing']['ophys']
    deconv_data = ophys['Deconvolved']['plane0']['data'][:]
```

iii. The AI identified that NWB files are organized by subject in `data/sub-{id}/` directories. Each NWB file contains one session's worth of behavioral and neural data. This approach correctly loads all available data from the standardized NWB format.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the directory name prefix `sub-{id}`. The subject ID is extracted by stripping the `sub-` prefix. A unique subjects list is built during dataset construction, and each session is assigned a `subject_idx` into this list.

ii.
```python
subj = session_info['subject']
if subj not in subjects_list:
    subjects_list.append(subj)
subject_idx_list.append(subjects_list.index(subj))
```

iii. This follows the NWB file organization where each subject has its own directory. The resulting subjects list contains 11 mice (m3, m4, m7, m11-m15, m17-m19), matching the paper's description of 11 switch-condition mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from the NWB metadata (`general/session_id`). Each file is processed independently by `process_session()` and contributes one entry to the lists in the output data structure.

ii.
```python
session_id = f['general']['session_id'][()].decode()
# ...
for i, file_info in enumerate(nwb_files):
    neural_trials, input_trials, output_trials, session_info = process_session(
        file_info['filepath'], show_processing=show_processing, session_idx=i)
    all_neural.append(neural_trials)
```

iii. The AI correctly identified that each NWB file = 1 session. The resulting 152 sessions match the expected count (11 mice x 14 days, minus 2 for m11 which started imaging on day 3).

## 1-d. How are the data split into trials?

i. Trial boundaries are found using the `trial_start` and `teleport` binary signals in the NWB behavioral timeseries. Frame indices where these signals are >0 mark trial starts and ends respectively. Each trial spans from its start index to its teleport index.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]

valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
```

iii. The reference code uses `sess.trial_start_inds` and `sess.teleport_inds` with `start-1:stop-1` slicing (1-indexed to 0-indexed conversion). The NWB signals are 0-indexed, so the agent's `np.where(...)[0]` approach gives equivalent indices. Trials are sliced as `[start:end]` which includes the start frame and excludes the teleport frame, consistent with the reference's `[start-1:stop-1]` convention.

## 1-e. How are trials filtered based on quality controls?

i. Two quality filters are applied: (1) trials with fewer than 5 timepoints are skipped, and (2) lick error trials have their lick data set to NaN (but the trial is still included). Sessions with fewer than 2 valid trials are skipped entirely. There is no speed-based trial filtering.

ii.
```python
if n_timepoints < 5:
    continue  # Skip very short trials

# Lick error correction per trial
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_lick = lick[start:end]
    if len(trial_lick) > 0:
        frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
        if frac_bad > LICK_ERROR_FRACTION:  # 0.35
            lick_binary[start:end] = np.nan
```

iii. The lick error correction matches the reference code's approach (>35% of samples with cumulative lick >2 triggers NaN for that trial's licks). The <5 timepoint filter is a reasonable edge case handler. The reference code also uses a speed threshold (<2 cm/s) to mask samples, but the AI chose not to exclude those timepoints from the data entirely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the pre-computed Deconvolved events stored in the NWB files at `processing/ophys/Deconvolved/plane{N}/data`. These are OASIS-deconvolved calcium signals.

ii.
```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]  # (n_samples, n_rois)
# For multi-plane:
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][:])
deconv_data = np.concatenate(deconv_parts, axis=1)
```

iii. The AI correctly identified that NWB files contain pre-computed deconvolved events (equivalent to `sess.timeseries['events']` in the reference code). The reference pipeline goes: raw F -> neuropil subtraction -> dF/F -> OASIS deconvolution, but since the NWB already has deconvolved data, using it directly is correct.

## 2-b. How is the `neural` data processed?

i. Processing steps: (1) Filter by `iscell` mask (Suite2p curation), (2) exclude putative interneurons identified by dF/F-speed correlation >0.5, (3) for multi-plane animals, concatenate neurons across planes, (4) extract per-trial segments between trial_start and teleport indices, (5) replace NaN values with 0, (6) transpose to (n_neurons, n_timepoints) format.

ii.
```python
# iscell filtering
iscell = seg['iscell'][:, 0].astype(bool)
# Interneuron exclusion
is_interneuron = identify_interneurons(dff, speed, iscell)
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
# Select neurons
neural_all = deconv_data[:, final_neuron_mask]
# Per-trial extraction
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The AI computed dF/F from raw Fluorescence and Neuropil data specifically for interneuron detection (not for the final neural output). The dF/F computation follows the reference: neuropil subtraction (coef=0.7), maximin baseline (300-frame window), Gaussian smooth (sigma=2). The NaN-to-0 replacement follows the reference pattern (`X[np.isnan(X)] = 0`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neuron-level filters: (1) Suite2p `iscell` classification removes non-cell ROIs, (2) putative interneurons with Pearson correlation of dF/F with speed >0.5 are excluded.

ii.
```python
INTERNEURON_CORR_THRESHOLD = 0.5

def identify_interneurons(dff, speed, iscell_mask):
    # Vectorized Pearson correlation of dFF with speed
    r = cov_XY / (std_X * std_Y)
    is_interneuron = r > INTERNEURON_CORR_THRESHOLD
    return is_interneuron
```

iii. Both filters match the reference paper: iscell from Suite2p manual curation, and interneuron exclusion at r>0.5 (paper: "correlation of >0.5", excluding "0.42 +/- 0.85% of cells"). The AI's converted data shows 0.28% interneuron exclusion, within the paper's reported range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. Each trial's neural data begins at the frame where `trial_start` signal is >0 and ends at the frame where `teleport` signal is >0. Time from trial start is computed as frame index * frame period.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()  # start = trial_starts[t]
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The instructions specify "Temporally align based on start of the trial", which the AI implements correctly. The first timepoint of each trial corresponds to t=0 (trial onset).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one imaging frame: ~64.5 ms (1/15.5078125 Hz). No temporal rebinning is applied; data is kept at native imaging frame rate.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
# metadata:
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame (~64.48 ms)
```

iii. The reference paper states imaging rate of ~15.5 Hz. No rebinning is needed since all behavioral and neural timeseries in the NWB are already at this common frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This variable is not derived from any raw data variable. It is computed from the frame index within each trial multiplied by the frame period (1/imaging_rate).

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input[0, :] = time_from_start
```

iii. Since each trial starts at frame index 0 and the imaging rate is constant, the time from trial start is simply the frame count times the frame period. This is a straightforward computation.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The number of timepoints per trial is computed as `end - start` (teleport index minus trial start index). An array `[0, 1, 2, ..., n_timepoints-1]` is multiplied by the frame period (~0.0645 s) to get time in seconds.

ii.
```python
n_timepoints = end - start
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. No complex processing needed. The time variable linearly increases from 0 at the start of each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time from trial start is inherently aligned with neural data because both share the same frame indexing. Frame 0 of the trial = time 0, frame 1 = FRAME_PERIOD seconds, etc. Both neural data and time are extracted over the same `[start:end]` range.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()  # n_timepoints
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD  # same n_timepoints
```

iii. The alignment is exact by construction since both use the same trial boundaries and frame indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the NWB `environment` behavioral timeseries, which contains values -1 (pre-TTL), 0 (ENV1), or 1 (ENV2).

ii.
```python
environment = behav['environment']['data'][:]
```

iii. The NWB environment variable directly encodes environment type as 0 or 1, matching the binary ENV1/ENV2 requirement.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the environment values within the trial window are extracted. Invalid values (<0, i.e., pre-TTL) are filtered out, and the median of remaining values is taken as the trial's environment type.

ii.
```python
trial_env = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    env_vals = environment[start:end]
    valid_env = env_vals[env_vals >= 0]
    if len(valid_env) > 0:
        trial_env[t] = int(np.median(valid_env))
    else:
        trial_env[t] = 0
```

iii. Since environment is constant within a trial, taking the median is a robust way to extract the per-trial value. The filtering of negative values handles pre-TTL artifacts.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the loop counter `t` iterating over the detected trials, NOT from the NWB `trial number` behavioral timeseries. Although the NWB `trial number` variable is loaded, it is not used for the input.

ii.
```python
trial_number = behav['trial number']['data'][:]  # loaded but not used for input
# ...
trial_num = float(t)  # loop counter used instead
trial_input[2, :] = trial_num
```

iii. The loop counter `t` represents the 0-indexed trial within the session. The NWB `trial number` field contains VR-system trial numbers which should be similar but may differ slightly. The AI chose to use the sequential index rather than the VR trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the 0-indexed loop counter `t` converted to float. It is broadcast across all timepoints in the trial (same value for every frame).

ii.
```python
trial_num = float(t)
trial_input[2, :] = trial_num  # broadcast per-trial value
```

iii. No processing beyond type conversion. The value ranges from 0 to n_trials-1 for each session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the NWB Reward event data. Reward timestamps are mapped to frame indices using `np.searchsorted`, then each trial is checked for whether any reward events fall within its boundaries. The previous trial's reward status becomes the current trial's "previous trial outcome".

ii.
```python
reward_data = behav['Reward']['data'][:]
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]

reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)

trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```

iii. The reward detection uses timestamp-to-frame mapping since rewards are event-based (not frame-rate sampled). This correctly identifies which trials received rewards.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial t, the previous trial outcome is `trial_rewarded[t-1]` (binary: 0=omitted, 1=rewarded). For the first trial (t=0), the value defaults to 0 (unknown/no previous).

ii.
```python
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. This is a straightforward one-trial lag of the reward outcome. The first trial defaulting to 0 is a reasonable convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from two sources: (1) the NWB `position` timeseries (animal's position on track in cm), and (2) the reward zone boundaries, which are inferred from the NWB `reward_zone` signal and position data.

ii.
```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The signed distance is computed from the animal's current position relative to the known reward zone start/end coordinates.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance is computed: negative if before the zone (position < zone start), zero if within the zone (position between start and end), positive if past the zone (position > zone end). The reward zone for each trial is identified by examining where `reward_zone > 0` in the NWB signal and mapping the mean position to the closest known zone (A: 80-130, B: 200-250, C: 320-370).

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start  # negative
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end  # positive
    return distance
```

iii. The signed distance convention (negative before, zero in zone, positive after) correctly captures the animal's spatial relationship to the reward zone. The reward zone identification from position data is a robust approach when scene metadata is unavailable in NWB format.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is discretized into 7 bins matching the instruction specification:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 cm to < 0 cm
- 3: 0 cm (in reward zone)
- 4: >0 cm to +10 cm
- 5: +10 to +50 cm
- 6: > +50 cm

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # in reward zone
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bins match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance to reward zone is computed from position data at the same frame indices as the neural data (`[start:end]` for each trial). Both share the same temporal alignment (each index corresponds to the same imaging frame).

ii.
```python
trial_pos = position[start:end]  # same start:end as neural
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. Since behavioral and neural data in NWB are already aligned to the imaging frame rate, no additional alignment is needed.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived directly from the NWB `position` behavioral timeseries, which records the animal's position on the 450 cm virtual track in centimeters.

ii.
```python
position = behav['position']['data'][:]
trial_pos = position[start:end]
```

iii. The position variable in NWB is already in cm on the track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw position (0-450 cm) is discretized into 5 equal-sized bins of 90 cm each using floor division.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. Five bins of 90 cm each covering the 450 cm track: [0,90), [90,180), [180,270), [270,360), [360,450]. The clip ensures edge values stay within valid bin range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal bins:
- 0: 0-90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: 360-450 cm

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. Matches the instruction "Absolute position in corridor, discretized into 5 equal-sized bins".

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is extracted from the same frame indices as neural data (`[start:end]`), ensuring 1:1 temporal alignment.

ii.
```python
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
trial_output[1, :] = pos_bins
```

iii. Both share the same NWB frame-rate sampling and trial boundaries.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the NWB `lick` behavioral timeseries, which contains cumulative lick counts per frame.

ii.
```python
lick = behav['lick']['data'][:]
```

iii. The lick variable in NWB records cumulative lick sensor readings at the imaging frame rate.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two processing steps: (1) Binarization: cumulative lick counts are clipped to [0, 1] using `np.clip`, so 0 stays 0 and any count >=1 becomes 1. (2) Error correction: for trials where >35% of samples have cumulative count >2, all lick values are set to NaN. NaN lick values are then replaced with 0 in the final output.

ii.
```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)

# Error correction
for t in range(n_trials):
    trial_lick = lick[start:end]
    frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
    if frac_bad > LICK_ERROR_FRACTION:  # 0.35
        lick_binary[start:end] = np.nan

# In per-trial output construction:
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The reference code does `licks[licks > 1] = 1` which is functionally equivalent to `np.clip(lick, 0, 1)` for non-negative integer counts. The error correction threshold of 35% matches the reference code's `0.35`. The reference applies error correction before binarization, while the agent applies them in parallel (error check on original data, binarization separately), but the result is equivalent.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data is extracted from the same frame indices as neural data, ensuring temporal alignment.

ii.
```python
trial_lick = lick_binary[start:end]
trial_output[3, :] = lick_vals
```

iii. Same frame-rate alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from two NWB variables: (1) `reward_zone` signal (>0 when animal is in the reward zone) and (2) `position` (to determine which physical zone the animal is in). The mean position during reward_zone > 0 periods is compared to known zone centers.

ii.
```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]
    in_rz = rz_trial > 0
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone_name
    return best_zone
```

iii. The reference code uses scene strings from `sessions_dict.py` to map to reward zone coordinates via `get_reward_zones()`. Since scene metadata is not available in NWB files, the agent infers the zone from position data. This produces correct results (balanced A/B/C distribution at ~33% each).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial: (1) Find timepoints where reward_zone > 0, (2) compute mean position at those timepoints, (3) map to closest zone center (A: 105cm, B: 225cm, C: 345cm). If no reward zone entry detected, carry forward the last known zone. Zones are encoded as integers: A=0, B=1, C=2. The value is broadcast across all timepoints in the trial.

ii.
```python
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
trial_output[4, :] = rz_loc  # broadcast per-trial
```

iii. The carry-forward logic for trials without reward zone entry handles edge cases. The 3-category encoding matches the instruction specification (0=A, 1=B, 2=C).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the NWB `Reward` event data, which contains timestamps and data values for reward delivery events.

ii.
```python
reward_data = behav['Reward']['data'][:]
reward_timestamps = behav['Reward']['timestamps'][:]
```

iii. Reward events are stored separately from the frame-rate behavioral timeseries in NWB format, requiring timestamp-to-frame mapping.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are converted to frame indices using `np.searchsorted`. For each trial, a binary flag indicates whether any reward event occurred within the trial's frame window. The value is broadcast across all timepoints.

ii.
```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
reward_out = int(trial_rewarded[t])
trial_output[5, :] = reward_out  # broadcast per-trial
```

iii. The reference code checks both `sess.vr_data['reward']` and `sess.vr_data['rzone']` to determine reward. The agent's approach of checking for reward events within trial boundaries is a reasonable equivalent using NWB event data.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Behavioral-neural length mismatch**: Truncated to common length (occurs for multi-plane animals differing by 1 frame)
- **Very short trials** (<5 frames): Skipped entirely
- **Lick sensor errors** (>35% samples with count >2): Lick data set to NaN, then replaced with 0 in output
- **NaN in neural data**: Replaced with 0 (following reference pattern)
- **Missing reward zone entry**: Carry forward last known zone; fill backward for leading unknowns
- **Missing environment values** (pre-TTL = -1): Filtered out, median of valid values used
- **Mismatched trial_starts/teleports count**: Truncated to minimum; invalid pairs (teleport before start) filtered

ii.
```python
# Length mismatch
n_samples = min(n_behav_samples, n_neural_samples)
position = position[:n_samples]
deconv_data = deconv_data[:n_samples, :]

# NaN neural
trial_neural[np.isnan(trial_neural)] = 0

# NaN lick -> 0
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)

# Reward zone carry-forward
if zone is not None:
    last_known_zone = zone
trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. The data handling is robust and addresses known issues in the NWB data. Each edge case is handled in a documented manner.

## 13-a. What are the most time-consuming steps of the code?

i. Based on the per-session timing reported: (1) dF/F computation (for interneuron detection) is the most expensive step, (2) NWB file loading (h5py I/O) is second, (3) interneuron detection (vectorized correlation). Total conversion time for 152 sessions is ~879 seconds (~14.6 minutes).

ii.
```python
t1 = time.time()
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
t_dff = time.time() - t1

t1 = time.time()
is_interneuron = identify_interneurons(dff, speed, iscell)
t_int = time.time() - t1
```

iii. The dF/F computation involves per-trial loops with Gaussian filtering and min/max filtering operations across potentially thousands of neurons and tens of thousands of timepoints.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- **Lick error correction**: The per-trial loop checking `frac_bad` could use vectorized operations over pre-segmented trial data
- **Environment per trial**: Similar per-trial loop could use fancy indexing
- **Previous trial outcome**: The loop `for t in range(1, n_trials)` could be replaced with array shifting
- **Reward detection per trial**: The per-trial reward check loop could use searchsorted with trial boundaries
- **dF/F computation**: The per-trial loop in `compute_dff()` with Gaussian filtering could potentially be done in a single pass with masked operations

ii.
```python
# Current loop-based lick error:
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_lick = lick[start:end]
    frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
    if frac_bad > LICK_ERROR_FRACTION:
        lick_binary[start:end] = np.nan

# Current loop-based environment:
for t in range(n_trials):
    env_vals = environment[start:end]
    valid_env = env_vals[env_vals >= 0]
    trial_env[t] = int(np.median(valid_env))
```

iii. The interneuron detection was already vectorized (mentioned in CONVERSION_NOTES as improving from 4.3s to 0.4s). The remaining loops are over trials (~80 per session) so the performance impact is modest.

## 13-c. What processing does the code repeat multiple times?

i. The dF/F computation is done for every session even though the deconvolved data is already in NWB. The dF/F is only needed for interneuron detection (speed-dFF correlation), but involves full neuropil subtraction, baseline computation, and Gaussian smoothing for all neurons and timepoints. Additionally, the trial boundary extraction (finding start/teleport indices) and the per-trial slicing pattern is repeated for each variable independently rather than being done once.

ii.
```python
# dF/F computed for ALL neurons, but only used for interneuron check
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
# Then interneuron check only uses iscell-filtered subset
dff_cells = dff[:, iscell_mask]
```

iii. The dF/F computation for all ROIs (not just iscell) is wasteful since only iscell-filtered neurons need the speed correlation check.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several forms of unnecessary processing:
1. **dF/F for non-cell ROIs**: dF/F is computed for ALL ROIs including non-cells, but only iscell-filtered neurons are used for interneuron detection
2. **Loading fluorescence/neuropil data**: This data is loaded for every session only for interneuron detection; the actual neural output uses pre-computed deconvolved data
3. **Per-trial output broadcasting**: Per-trial values (environment, trial number, previous outcome, reward zone, reward outcome) are broadcast to all timepoints in `(n_vars, n_timepoints)` format even though they are constant per trial; downstream analysis could handle them as scalars

ii.
```python
# Loading F and Fneu only for interneuron detection
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
# These large arrays are only used for compute_dff -> identify_interneurons
```

iii. Loading raw fluorescence data constitutes significant I/O overhead. For the final output format, per-trial variables being broadcast to full timeseries increases the output file size (9.4 GB) unnecessarily, though this matches the target format specification.
