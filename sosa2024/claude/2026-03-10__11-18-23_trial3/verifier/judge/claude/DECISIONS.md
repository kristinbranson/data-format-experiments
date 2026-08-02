# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of `data/` that start with `sub-`. Files are found using `os.listdir`, sorted, and each `.nwb` file within each subject directory is included. The data are loaded using `h5py` (not `pynwb`), directly accessing the HDF5 structure of the NWB files.

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

Loading with h5py:
```python
with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    position = behav['position']['data'][:]
    ...
    ophys = f['processing']['ophys']
    deconv_data = ophys['Deconvolved']['plane0']['data'][:]
```

iii. The AI explored the NWB file structure in early trajectory steps and found it could use h5py directly to access all needed data. The CONVERSION_NOTES confirm 11 subjects and 152 total sessions were found, matching the paper.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` that start with `sub-`. The subject ID is extracted by stripping the `sub-` prefix.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```

iii. The number of subject directories (11) matches the number in the paper. Subject IDs are also confirmed from the HDF5 metadata field `general/subject/subject_id`.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by listing all `.nwb` files within each subject directory.

ii.
```python
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

iii. File names follow the pattern `sub-{id}_ses-{nn}_behavior+ophys.nwb`. The CONVERSION_NOTES confirm 152 total sessions (14 per subject, 12 for m11).

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using `trial_start_signal` (for starts) and `teleport_signal` (for ends). The code finds all indices where these signals are > 0 using `np.where`, then pairs starts with teleports and filters for validity (teleport > corresponding start).

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

iii. The AI found that `trial_start` and `teleport` signals mark trial boundaries. The CONVERSION_NOTES report ~80.4 +/- 6.1 trials per session, matching the paper's 80.5 +/- 7.4. However, this approach takes all frames where teleport > 0, not just the onset (rising edge), which could be problematic if the teleport signal stays high for multiple frames.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. Additionally, lick error correction identifies trials where >35% of samples have cumulative lick count > 2, though these trials are not removed -- only their lick data is set to NaN.

ii.
```python
if n_timepoints < 5:
    continue  # Skip very short trials
```

Lick error correction:
```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_lick = lick[start:end]
    if len(trial_lick) > 0:
        frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
        if frac_bad > LICK_ERROR_FRACTION:
            lick_binary[start:end] = np.nan
            lick_error_trials.append(t)
```

iii. The CONVERSION_NOTES mention that 69 lick error trials were found (close to the paper's 81 for all mice). The minimum trial length threshold of 5 is relatively permissive.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field in the ophys processing module of the NWB files.

ii.
```python
ophys = f['processing']['ophys']
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
```

For multi-plane sessions:
```python
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][:])
deconv_data = np.concatenate(deconv_parts, axis=1)
```

iii. The paper describes using deconvolved calcium events (OASIS algorithm from Suite2p) for decoding analyses. The NWB files contain pre-computed deconvolved data.

## 2-b. How is the `neural` data processed?

i. Multiple processing steps: (1) neurons from multiple planes are concatenated, (2) neurons are filtered by `iscell` classification, (3) putative interneurons are excluded based on dF/F-speed correlation, (4) NaN values in deconvolved data are replaced with 0.

ii.
```python
# iscell filtering
iscell = seg['iscell'][:, 0].astype(bool)

# Interneuron exclusion
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)

# Final mask
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
neural_all = deconv_data[:, final_neuron_mask]

# NaN replacement
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The CONVERSION_NOTES explain that the paper describes interneuron exclusion (Pearson r > 0.5 with speed). The AI computed dF/F from Fluorescence and Neuropil data specifically to enable this filtering step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Suite2p `iscell` classification, and (2) interneuron exclusion based on correlation between dF/F and speed exceeding 0.5.

ii.
```python
def identify_interneurons(dff, speed, iscell_mask):
    dff_cells = dff[:, iscell_mask]
    # Vectorized Pearson correlation
    X = dff_cells[valid, :]
    Y = speed[valid]
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean()
    cov_XY = (X_centered * Y_centered[:, np.newaxis]).sum(axis=0) / (n - 1)
    std_X = np.sqrt((X_centered ** 2).sum(axis=0) / (n - 1))
    std_Y = np.sqrt((Y_centered ** 2).sum() / (n - 1))
    r = cov_XY / (std_X * std_Y)
    is_interneuron = r > INTERNEURON_CORR_THRESHOLD
    return is_interneuron
```

iii. The paper states interneurons (0.42 +/- 0.85% of cells) were excluded based on speed correlation > 0.5. The CONVERSION_NOTES report 0.28% excluded, within the expected range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data are aligned to trial start by indexing from `trial_starts[t]` to `teleports[t]` for each trial. No additional temporal shifting is needed since alignment to trial start simply means extracting data from the trial start index onward.

ii.
```python
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_neural = neural_all[start:end, :].T.copy()
```

iii. The instructions specify alignment to "start of the trial", which corresponds to the first imaging frame of each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5 Hz (~64.5 ms per frame). No temporal rebinning is applied.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
...
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
```

iii. The paper reports imaging at ~15.5 Hz. Multi-plane animals image at 31 Hz total (15.5 Hz per plane), so the effective per-plane rate is the same.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is computed from the frame index and a constant frame period (1/15.5078125 Hz), rather than from actual timestamps.

ii.
```python
FRAME_PERIOD = 1.0 / IMAGING_RATE
...
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The AI used the known imaging rate to compute time rather than extracting timestamps from the NWB file. Since the imaging rate is constant, this should produce equivalent results.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Multiply the zero-based frame index within the trial by the frame period in seconds.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. This produces a time vector starting at 0 at the beginning of each trial, incrementing by ~64.5 ms per frame.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices within each trial (both extracted from `start:end`), so they are inherently aligned. The time vector has the same number of timepoints as the neural data.

ii.
```python
n_timepoints = end - start
trial_neural = neural_all[start:end, :].T.copy()
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. Both use the same `start:end` indexing, guaranteeing alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB file.

ii.
```python
environment = behav['environment']['data'][:]
...
trial_env = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    env_vals = environment[start:end]
    valid_env = env_vals[env_vals >= 0]
    if len(valid_env) > 0:
        trial_env[t] = int(np.median(valid_env))
```

iii. Environment is 0 (ENV1) or 1 (ENV2), matching the binary specification in the instructions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the median of valid (>= 0) environment values is taken. Pre-trial values of -1 are excluded. The result is broadcast across all timepoints.

ii.
```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
else:
    trial_env[t] = 0
```

iii. Since environment is constant within a trial (either 0 or 1), taking the median is equivalent to taking any single valid value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session loop counter `t`, derived from the trial boundaries (trial_start and teleport signals).

ii.
```python
for t in range(n_trials):
    ...
    trial_num = float(t)
    trial_input[2, :] = trial_num
```

iii. The stored `trial number` variable in the NWB file was not used; instead the sequential 0-based trial index is assigned.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index as a float. The value is constant across all timepoints within a trial.

ii.
```python
trial_num = float(t)
trial_input[2, :] = trial_num
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward event timestamps are mapped to frame indices using `np.searchsorted`, and a per-trial `trial_rewarded` boolean array is computed.

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

iii. The Reward time series has its own timestamps separate from the behavior frame rate, so they are mapped to the nearest frame index.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward status is looked up from the pre-computed `trial_rewarded` array. The first trial defaults to 0 (no previous trial).

ii.
```python
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. The instructions specify binary previous trial outcome (omitted=0, rewarded=1). The code directly uses the previous trial's reward boolean.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. The reward zone for each trial is identified by computing the mean position where `reward_zone > 0` and mapping to the nearest zone (A, B, or C). For trials without reward zone entry, the last known zone is forward-filled.

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

# Forward fill for trials without zone entry
last_known_zone = None
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
    if zone is not None:
        last_known_zone = zone
    trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. The CONVERSION_NOTES describe the reward zone identification approach and verify that zone distributions are approximately balanced (34.3/32.8/32.9%).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from position to the nearest edge of the reward zone. Distance is 0 when inside the zone, negative when before the zone, positive when past it.

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

iii. This matches the paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit boolean conditions:

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

iii. The bins match the instruction specification. Bin 3 corresponds to exactly 0 (inside the reward zone).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices (`start:end`) within each trial, so no additional alignment is needed.

ii.
```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. Same indexing as neural data ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins of 90 cm each spanning 0-450 cm, using `np.floor(position / 90.0)` clipped to [0, 4].

ii.
```python
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)  # [0, 90, 180, 270, 360, 450]

def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

Resulting bins: 0-90 cm, 90-180 cm, 180-270 cm, 270-360 cm, 360-450 cm.

iii. The instructions specify "5 equal-sized bins". The AI assumes a 0-450 cm track (90 cm per bin). Positions < 0 are clipped to bin 0, and positions >= 450 are clipped to bin 4.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial -- no additional alignment needed.

ii.
```python
trial_pos = position[start:end]
```

iii. Same `start:end` indexing as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick']['data'][:]
```

iii. The `lick` variable records cumulative lick counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Multi-step processing: (1) clip raw lick values to [0, 1] using `np.clip`, (2) apply lick error correction -- trials where >35% of samples have cumulative lick > 2 have their lick data set to NaN, (3) NaN values are replaced with 0 in the final output.

ii.
```python
# Binarize
lick_binary = np.clip(lick, 0, 1).astype(np.float64)

# Error correction
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_lick = lick[start:end]
    if len(trial_lick) > 0:
        frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
        if frac_bad > LICK_ERROR_FRACTION:
            lick_binary[start:end] = np.nan
            lick_error_trials.append(t)

# In trial construction:
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The paper describes lick sensor error correction where trials with >35% of samples having cumulative lick >2 are flagged. The CONVERSION_NOTES report 69 such trials found, close to the paper's 81.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data -- no additional alignment needed.

ii.
```python
trial_lick = lick_binary[start:end]
```

iii. Same `start:end` indexing as neural data.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. The zone is identified by computing the mean position where `reward_zone > 0` and finding the closest of the three defined zones (A: 80-130, B: 200-250, C: 320-370).

ii. See answer 7-a for the `identify_reward_zone` function.

iii. See answer 7-a.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, mean position in reward zone is computed and mapped to nearest zone (A/B/C). Trials without reward zone entry are forward-filled from the last known zone. Leading None values are backward-filled. The zone label is then mapped to an index (A=0, B=1, C=2).

ii.
```python
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
...
trial_output[4, :] = rz_loc  # broadcast per-trial
```

iii. See answer 7-a.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series, which has its own timestamps separate from the regular frame rate.

ii.
```python
reward_data = behav['Reward']['data'][:]
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
```

iii. The Reward time series records reward delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices using `searchsorted`. For each trial, the output is 1 if any reward event frame falls within [trial_start, teleport), otherwise 0. The value is constant across all timepoints in the trial.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
...
reward_out = int(trial_rewarded[t])
trial_output[5, :] = reward_out
```

iii. Binary per-trial output as specified in the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is truncated to the minimum of behavioral and neural sample counts.
- **Very short trials**: Trials with fewer than 5 timepoints are skipped.
- **NaN in neural data**: Replaced with 0.
- **Missing reward zone**: Trials where `reward_zone` is never active use forward-fill (or backward-fill for leading trials) from neighboring trials.
- **Lick sensor errors**: Trials with >35% of samples having cumulative lick > 2 have lick set to NaN (then 0 in output).
- **Reward timestamp clipping**: Frame indices from `searchsorted` are clipped to valid range.

ii.
```python
n_samples = min(n_behav_samples, n_neural_samples)
if n_behav_samples != n_neural_samples:
    position = position[:n_samples]
    ...
    deconv_data = deconv_data[:n_samples, :]

if n_timepoints < 5:
    continue

trial_neural[np.isnan(trial_neural)] = 0

reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
```

iii. These defensive checks were implemented based on data exploration and reference paper descriptions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with h5py and reading large arrays -- I/O bound
2. **Computing dF/F** from fluorescence and neuropil data for interneuron detection (~4-5s per session)
3. **Interneuron detection** -- correlation computation
4. **Saving the large .pkl file** (~9.4 GB)

ii. N/A (timing information is printed during execution)

iii. The CONVERSION_NOTES report total conversion time of ~879s (~14.6 min) for 152 sessions, with dF/F computation being a significant per-session cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` (lines 509-589) iterates over each trial sequentially. Operations like `discretize_distance`, `discretize_position`, and `discretize_speed` could theoretically be applied to full session arrays before splitting into trials. The interneuron detection was already vectorized (matrix correlation instead of per-neuron loop).

ii. N/A

iii. Variable trial lengths make full vectorization awkward. The AI notes that interneuron detection was optimized from 4.3s to 0.4s per session by vectorizing.

## 13-c. What processing does the code repeat multiple times?

i. The dF/F computation is performed solely for interneuron detection but requires loading Fluorescence and Neuropil data, which are read and processed even though the final output only uses Deconvolved data. The fluorescence/neuropil data loading happens once per session but the dF/F computation is a significant overhead that could be avoided if interneuron labels were pre-computed.

ii. N/A

iii. Unlike the reference solution, the AI's code does not have a separate survey step, so each NWB file is only loaded once (but with the overhead of dF/F computation).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation (`compute_dff`) processes all neurons' fluorescence data to identify interneurons, but the dF/F values themselves are not used in the output -- only the deconvolved data is kept. This is a significant computational cost (loading Fluorescence + Neuropil data, neuropil subtraction, maximin baseline, smoothing) used solely for a filtering step that excludes ~0.28% of neurons.

ii.
```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
# dff is not used further
```

iii. The interneuron filtering is described in the paper, but the computational cost of dF/F for this marginal filtering may not justify the overhead.
