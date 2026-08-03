# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all `sub-*` directories in the `data` directory, then collects all `.nwb` files within each. It uses `h5py` (not `pynwb`) to directly access HDF5 groups in each NWB file, loading behavioral time series from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/Deconvolved`. All sessions are processed unless `--sample` is used (which picks 2 hardcoded sessions).

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
            all_files.append({...})
    return all_files
```
```python
with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    position = behav['position']['data'][:]
    ...
    ophys = f['processing']['ophys']
    deconv_data = ophys['Deconvolved']['plane0']['data'][:]
```

iii. The AI documented in CONVERSION_NOTES.md that it found 11 subjects with 152 total sessions, matching the paper. It chose h5py for direct HDF5 access rather than pynwb.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `sub-*` directories. The subject ID is extracted from the directory name by removing the `sub-` prefix. Subject IDs are also read from the NWB file metadata for session tracking.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
subject_id = f['general']['subject']['subject_id'][()].decode()
```

iii. The 11 subjects match the paper's count for the switch condition.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are collected per subject directory and sorted alphabetically.

ii.
```python
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

iii. The AI noted 152 total sessions (14 per subject, 12 for m11), matching the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by finding all indices where `trial_start_signal > 0` and all indices where `teleport_signal > 0`. The number of trials is set to the minimum of these two counts, and they are truncated to match. A validity check ensures each teleport comes after its corresponding trial start.

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

iii. The AI notes that trial boundaries match the expected ~80.5 trials/session from the paper. Unlike the reference which detects teleport rising edges, the AI takes all frames where teleport > 0. This works if teleport is a single-frame impulse signal.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. Additionally, lick error correction is applied: trials where >35% of samples have cumulative lick count > 2 have their lick data set to NaN (the trial is still included, just with zeroed lick).

ii.
```python
if n_timepoints < 5:
    continue  # Skip very short trials
```
```python
frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
    lick_error_trials.append(t)
```

iii. The AI found ~69 lick error trials across all sessions (paper reports 81 including fixed mice). The 5-timepoint threshold is much lower than the reference's 50.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field in the NWB ophys processing module. The AI also loads `Fluorescence` and `Neuropil` data for computing dF/F used in interneuron detection.

ii.
```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
```

iii. The AI correctly identified that the NWB files contain pre-computed deconvolved events and uses those for the final neural data, consistent with the paper's decoder approach.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps beyond loading: (1) combines neurons across planes for multi-plane sessions, (2) filters by `iscell` mask, (3) excludes putative interneurons identified by correlation of dF/F with speed > 0.5. NaN values in neural data are replaced with 0.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
neural_all = deconv_data[:, final_neuron_mask]
...
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The AI documented that the paper describes interneuron exclusion (Pearson r > 0.5 with speed), and implemented it by computing dF/F from Fluorescence/Neuropil data, then correlating with speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` from Suite2p manual curation, (2) interneuron exclusion based on dF/F-speed correlation > 0.5.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_interneuron = identify_interneurons(dff, speed, iscell)
final_neuron_mask[iscell_indices[non_interneuron]] = True
```

iii. The AI noted that the paper reports 0.42 +/- 0.85% interneuron exclusion rate, and the converted data shows 0.28%, within the expected range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data from trial_start index to teleport index. No additional temporal offset is applied.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()  # (n_neurons, n_timepoints)
```

iii. The instructions specify alignment to start of trial, which is achieved by the trial boundary slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is kept at the native imaging rate (~15.5 Hz, ~64.5 ms per frame). No rebinning is applied. The time bin size is computed from a hardcoded constant `IMAGING_RATE = 15.5078125`.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
...
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
```

iii. The AI hardcoded the imaging rate rather than reading it from each NWB file dynamically, though it does read it for logging. The reference computes the bin size from `nplanes/plane_data.rate*1000`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within the trial multiplied by the frame period (1/IMAGING_RATE). Does NOT use the NWB timestamps directly.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The AI used computed time rather than NWB timestamps. The reference uses actual behavior timestamps and subtracts the first timestamp.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Simply multiply frame index (0, 1, 2, ...) by the frame period constant. No timestamp subtraction.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. This assumes perfectly uniform frame spacing, which is approximately true for the scanning microscope.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same frame indices (start:end from trial boundaries), so alignment is inherent.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. Neural and behavioral data share the same indexing since they are sampled at the same rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = behav['environment']['data'][:]
```

iii. The environment variable is 0 for ENV1 and 1 for ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI takes the median of valid (>= 0) environment values within the trial. This handles the -1 pre-TTL values.

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
```

iii. The reference simply indexes environment at the trial indices and casts to int. The AI's median approach is more defensive but functionally equivalent since environment is constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over trials (0-indexed within session).

ii.
```python
trial_num = float(t)
...
trial_input[2, :] = trial_num
```

iii. The AI uses the sequential trial index within the session, same as the reference.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
trial_input[2, :] = trial_num
```

iii. Simple sequential numbering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` event timestamps. Reward timestamps are mapped to frame indices using `np.searchsorted`, and a per-trial `trial_rewarded` boolean array is computed.

ii.
```python
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

iii. The AI's approach maps reward event timestamps to frame indices and checks which trials contain reward events, consistent with the reference.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial t, the previous trial outcome is `trial_rewarded[t-1]`. For the first trial, it defaults to 0.

ii.
```python
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. This matches the instructions: omitted = 0, rewarded = 1.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for each trial. Reward zone identification uses `reward_zone` signal and `position` to determine which zone (A, B, or C) is active, by computing the mean position where reward_zone > 0 and matching to the nearest zone center.

ii.
```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]
    in_rz = rz_trial > 0
    if not np.any(in_rz):
        return None
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    best_zone = None
    best_dist = np.inf
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone_name
    return best_zone
```

iii. The AI uses a simpler nearest-center approach compared to the reference's Viterbi algorithm. For trials with no reward zone entry, the AI forward-fills from the last known zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before zone, 0 inside zone, positive after zone. Then discretized into 7 bins matching the instructions.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end
    return distance
```

iii. This matches the expected signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using explicit conditional assignment into 7 bins. Bin 3 corresponds to exactly 0 (in reward zone).

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

iii. The bins match the instructions. The reference uses `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]` which achieves a similar result, though the boundary handling differs slightly (reference uses 1e-6 to separate exactly 0 from small positives).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Both use the same frame indices from trial boundaries, so alignment is inherent.

ii.
```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. Same indexing as neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
```

iii. The position variable records the animal's position in cm on the VR track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360-450) by dividing by 90 and flooring, then clipping to [0, 4].

ii.
```python
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)  # [0, 90, 180, 270, 360, 450]
...
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The AI uses 90 cm bins spanning 0-450 cm. The reference uses bins [-inf, 50, 150, 250, 350, inf] which are 100 cm wide but offset differently (centered differently on the track).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The instructions say "5 equal-sized bins". The AI interprets this as 5 equal 90cm bins covering 0-450cm. The reference uses different bin edges.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices from trial boundaries.

ii.
```python
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
```

iii. Same indexing as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = behav['lick']['data'][:]
```

iii. The lick variable contains cumulative lick counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps: (1) Clip lick values to [0, 1] for binarization. (2) Apply lick error correction: if >35% of a trial's samples have cumulative lick > 2, set lick data for that trial to NaN, which is later replaced with 0.

ii.
```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)

for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_lick = lick[start:end]
    if len(trial_lick) > 0:
        frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
        if frac_bad > LICK_ERROR_FRACTION:
            lick_binary[start:end] = np.nan
            lick_error_trials.append(t)
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The AI implements lick error correction from the reference code (`get_timeseries_data` in `glmUtils.py`). The reference solution (human) simply uses `(lick > 0).astype(int)` without error correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices from trial boundaries.

ii.
```python
trial_lick = lick_binary[start:end]
```

iii. Same indexing as neural data.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series. The position where reward_zone > 0 is used to determine which zone (A, B, or C) the animal entered.

ii.
```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    ...
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        ...
```

iii. Uses nearest zone center matching. Forward-fills for trials where the animal doesn't enter the reward zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, compute mean position where reward_zone > 0, match to nearest zone center (A=105, B=225, C=345). For trials with no reward zone entry, forward-fill from last known zone. Leading None values are backward-filled. Mapped to indices: A=0, B=1, C=2.

ii.
```python
last_known_zone = None
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
    if zone is not None:
        last_known_zone = zone
    trial_rz_label.append(zone if zone is not None else last_known_zone)

if trial_rz_label[0] is None:
    for t in range(n_trials):
        if trial_rz_label[t] is not None:
            for tt in range(t):
                trial_rz_label[tt] = trial_rz_label[t]
            break

rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. The reference uses a Viterbi algorithm with Gaussian emission model and transition probabilities to make more robust assignments. The AI's approach is simpler but may be less robust to noisy position data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` event timestamps.

ii.
```python
reward_timestamps = behav['Reward']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
```

iii. Reward events have their own timestamps separate from the behavior sampling rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices. For each trial, check if any reward frame index falls within [start, end). Binary output: 1 if rewarded, 0 otherwise.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
...
trial_output[5, :] = reward_out
```

iii. This matches the reference approach. The AI noted ~15.3% omission rate, consistent with the paper's ~15%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Truncated to common length.
- **Short trials**: Trials < 5 timepoints skipped.
- **Missing reward zone data**: Forward-fill from last known zone, backward-fill for leading unknowns.
- **NaN in neural data**: Replaced with 0.
- **Lick errors**: Trials with >35% bad lick samples have lick set to NaN then 0.
- **Multi-plane data**: Neurons concatenated across planes.

ii.
```python
n_samples = min(n_behav_samples, n_neural_samples)
...
if n_timepoints < 5:
    continue
...
trial_neural[np.isnan(trial_neural)] = 0
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The AI handles multiple edge cases defensively. The reference also handles neural/behavior mismatch and short trials (with a higher threshold of 50).

## 13-a. What are the most time-consuming steps of the code?

i. The AI identified and timed:
1. Loading NWB files (I/O bound)
2. dF/F computation (for interneuron detection)
3. Interneuron detection (correlation computation)
4. Full conversion took ~879s (~14.6 min) for 152 sessions.

ii. N/A (timing was printed during execution)

iii. The dF/F computation and interneuron detection are additional computational costs not present in the reference.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial sequentially for constructing inputs and outputs. The lick error correction loop could potentially be vectorized. The reward zone identification loop iterates per trial.

ii. N/A

iii. The AI vectorized the interneuron detection (matrix correlation instead of per-neuron loop), reducing time from 4.3s to 0.4s per session.

## 13-c. What processing does the code repeat multiple times?

i. The AI's code processes each NWB file only once (no separate survey step), so there is minimal repeated processing. However, the reward zone identification and distance computation are done inline per trial.

ii. N/A

iii. Unlike the reference which has a separate survey step that loads files twice, the AI processes everything in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F from raw Fluorescence and Neuropil data solely for interneuron detection. This is computationally expensive (maximin baseline, Gaussian smoothing per trial per neuron). The dF/F itself is never used as neural data -- only the deconvolved data is. The reference solution does not compute dF/F at all.

ii.
```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
```

iii. The dF/F computation is the most expensive step after data loading, and is only used to identify the ~0.3% of neurons that are interneurons.
