# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from NWB (HDF5) files stored in `/app/data/`. The code discovers subject directories (`sub-*`), then finds all `.nwb` files within each subject directory. Each NWB file represents one session and is opened with `h5py.File()`. Behavioral data (position, speed, lick, reward_zone, environment, trial_start, teleport, reward timestamps) and neural data (deconvolved calcium, fluorescence, neuropil, iscell, planeIdx) are extracted from specific HDF5 paths.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
# ...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
# ...
with h5py.File(nwb_path, 'r') as f:
    imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
    lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
    rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
    env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
    trial_start = f['processing/behavior/BehavioralTimeSeries/trial_start/data'][:]
    teleport = f['processing/behavior/BehavioralTimeSeries/teleport/data'][:]
    timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
    reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
    reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
    iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
    planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
    fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
    neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]
```

iii. The agent explored the NWB file structure using h5py and matched paths to the variables described in the reference code (behavior.py, dayData.py, preprocessing.py). All behavioral and neural variables are loaded from standardized NWB paths.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by directory names under the data directory. Each `sub-*` directory corresponds to one mouse. The subject ID is derived by stripping the `sub-` prefix. 11 switch mice are included (m3, m4, m7, m11-m15, m17-m19).

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. The agent verified against the paper that 11 switch mice should be included. The 3 fixed-condition mice are not in the DANDI dataset. Subject index tracking is done via `subject_idx_list.append(subj_i)`.

## 1-c. How are the data split into sessions?

i. Each NWB file within a subject directory represents one session. Sessions are sorted by filename, and the session number is extracted from the filename (e.g., `ses-01`). This yields 152 total sessions (12 for m11, 14 for all others).

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
# ...
fname = os.path.basename(nwb_file)
session_num = fname.split('ses-')[1].split('_')[0]
result = convert_session(nwb_file, subj_id, session_num)
```

iii. The agent identified that m11 has only 12 sessions (days 3-14) due to missing imaging on days 1-2, matching the paper's description.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` and `teleport` signals in the behavioral timeseries. The indices where `trial_start > 0` mark trial onsets, and indices where `teleport > 0` mark trial ends. Trials are paired sequentially. Invalid pairs (teleport <= trial_start) are removed. Trials with fewer than 2 timepoints are skipped.

ii.
```python
tstart_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport > 0)[0]
n_trials = min(len(tstart_inds), len(teleport_inds))
tstart_inds = tstart_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
valid = teleport_inds > tstart_inds
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
```

iii. The agent referenced the `get_trial_types()` function in `behavior.py` which uses `trial_start_inds` and `teleport_inds` to define trial boundaries. The approach of using `np.where(trial_start > 0)` matches the reference code pattern.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Lick sensor error trials are corrected (lick data zeroed out) but trials are NOT removed. Sessions with fewer than 2 valid trials or fewer than 2 cells after filtering are skipped entirely. No other trial-level filtering (e.g., by speed, by trial type) is applied.

ii.
```python
if n_timepoints < 2:
    continue
# ...
if n_kept < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_kept} cells after filtering")
    return None
# ...
if len(neural_trials) < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. The agent decided not to filter trials by type (e.g., auto-reward trials on first 10 trials of new conditions). All trials with valid data are included, which is reasonable for a decoder that needs to predict across conditions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the pre-computed deconvolved calcium activity stored in the NWB files at `processing/ophys/Deconvolved/plane0/data` (and `plane1/data` for multi-plane sessions m17, m18).

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
# ...
if has_multi_plane:
    deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
```

iii. The agent noted from the reference code (`dayData.py`: `ts_key: 'events'`) that deconvolved activity (OASIS algorithm) is the standard neural data representation used in the paper's analyses.

## 2-b. How is the `neural` data processed?

i. For multi-plane sessions (m17, m18), deconvolved data from plane0 and plane1 are concatenated along the cell axis. All data are truncated to the minimum length across behavioral and neural timeseries to handle off-by-one mismatches. The data are then filtered by cell masks and sliced per trial, transposed to (n_neurons, n_timepoints), and cast to float32.

ii.
```python
if has_multi_plane:
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
else:
    deconv_all = deconv_p0
min_len = min(len(pos), deconv_all.shape[0])
deconv_all = deconv_all[:min_len]
# ...
deconv_filtered = deconv_all[:, cell_mask]
# ...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The agent followed the paper's approach of pooling planes for all analyses. No additional processing (smoothing, normalization, rebinning) is applied to the deconvolved data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Manual curation from Suite2P (`iscell[:, 0] == 1`), and (2) Putative interneuron exclusion based on Pearson correlation > 0.5 between simplified dF/F and running speed. The dF/F is computed from raw fluorescence with neuropil subtraction (coefficient 0.7) and maximin baseline method.

ii.
```python
curated_mask = iscell[:, 0] == 1
# ...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
```

iii. The agent found the `int_thresh: 0.5` and `int_method: 'speed'` parameters in `dayData.py` and implemented interneuron detection accordingly. The paper reports 0.42 +/- 0.85% exclusion; the conversion removed 119 total interneurons across 152 sessions, which is consistent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. Each trial's neural data is sliced from `tstart_inds[i]` to `teleport_inds[i]`, so position 0 in the time dimension corresponds to the trial start frame.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The instructions specify "Temporally align based on start of the trial." The agent uses the `trial_start` signal indices directly, so t=0 is at the trial_start frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate: ~15.5 Hz (64.48 ms per frame). No temporal rebinning is applied.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ms (1/imaging_rate * 1000)
```

iii. The agent preserved the native imaging rate as the time bin size, which matches the reference code that operates at the imaging frame rate. The `imaging_rate` is read from the NWB file directly.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This is not derived from a raw data variable. It is computed from the frame index within the trial and the imaging rate.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The agent computed time as `frame_index * (1/imaging_rate)`, which gives seconds from trial start. This is a straightforward derivation from the frame count and sampling rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A linear time vector is created: `np.arange(n_timepoints) * frame_time`, where `frame_time = 1.0 / imaging_rate`. Frame 0 = 0.0 seconds, frame 1 = 1/imaging_rate seconds, etc.

ii.
```python
time_from_start = np.arange(n_timepoints) * frame_time
input_data[0, :] = time_from_start
```

iii. This is a direct computation with no complex processing. The agent chose to use frame indices rather than actual timestamps, which is reasonable given that the imaging rate is constant.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector has the same length as the neural data for each trial (`n_timepoints = e - s`), and both start at the trial start frame. Frame 0 of both corresponds to `tstart_inds[i]`.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
n_timepoints = e - s
neural = deconv_filtered[s:e, :].T.astype(np.float32)
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. Since both the neural data and the time vector are derived from the same trial boundaries and have the same number of timepoints, they are inherently aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` field in the NWB behavioral timeseries: `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
```

iii. The agent verified that the environment variable encodes 0 = ENV1, 1 = ENV2, and confirmed this matched expectations for m17/m18 which start in ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the environment values within the trial window are extracted, invalid values (< 0, which occur during teleport) are filtered out, and the median of valid values is taken. The result is a single integer per trial (0 or 1), broadcast across all timepoints.

ii.
```python
env_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    env_valid = env_vals[env_vals >= 0]
    if len(env_valid) > 0:
        env_per_trial[i] = int(np.median(env_valid))
    else:
        env_per_trial[i] = 0
```

iii. The agent took the median of valid environment values within each trial to handle potential transition artifacts. Since environment is constant within a trial (only changes at session level), the median is equivalent to taking any valid value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not directly derived from a raw data variable. It is the 1-indexed position of the trial in the session (loop index + 1).

ii.
```python
trial_number = float(i + 1)
```

iii. The agent used simple 1-based indexing of valid trials within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply `i + 1` where `i` is the 0-indexed trial position in the session's trial list. It is broadcast as a constant across all timepoints in the trial.

ii.
```python
trial_number = float(i + 1)
input_data[2, :] = trial_number
```

iii. No complex processing. The agent chose 1-based indexing. This represents the sequential trial count within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward event timestamps (`Reward/timestamps`) and the trial time windows defined by `trial_start` and `teleport` indices and `position/timestamps`.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
# ...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The agent matched sparse reward timestamps to trial windows to determine which trials had rewards, then used the previous trial's outcome as the current trial's input.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward outcome is determined by checking if any reward timestamp falls within the trial's time window. The previous trial's outcome (0=omission, 1=rewarded) is used as the current trial's input. The first trial of each session defaults to 0.

ii.
```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    n_trials = len(tstart_inds)
    outcomes = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
# ...
if i == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(reward_outcomes[i - 1])
```

iii. The agent used the sparse reward event timestamps and matched them to trial time windows, which is a correct approach. The first trial defaults to 0 (no previous trial).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from two raw variables: `position` (the animal's position along the track) and `reward_zone` (indicating when the animal is in the reward zone, used to determine which zone is active).

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
```

iii. Position provides the continuous location, and reward_zone activation helps determine which of the three zones (A, B, C) is active for each trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. First, the active reward zone is determined per trial by examining positions where `rzone > 0` and classifying into A/B/C using the median position with +/-20 cm tolerance. Omission trials inherit zone labels from neighbors via forward/backward fill. Then, signed distance is computed: negative if before zone, 0 if inside zone, positive if after zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0
    distance[after] = position[after] - rz_end
    return distance
```

iii. The reward zone boundaries are from the reference code: A=[80,130], B=[200,250], C=[320,370] cm. The distance computation gives signed distance to the nearest edge of the active zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is discretized into 7 bins matching the specification: 0: < -50cm, 1: -50 to -10cm, 2: -10 to <0cm, 3: 0cm (in zone), 4: >0 to +10cm, 5: +10 to +50cm, 6: >+50cm.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros_like(distance, dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bin edges match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from position data that shares the same time indices as the neural data. Both are sliced from `tstart_inds[i]` to `teleport_inds[i]`, ensuring frame-by-frame alignment.

ii.
```python
trial_pos = pos[s:e]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_disc = discretize_distance(dist)
output_data[0, :] = dist_disc
```

iii. Since position and neural data are sampled at the same imaging rate and sliced with the same indices, they are inherently aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `position` field in the NWB behavioral timeseries.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
```

iii. Position represents the animal's location along the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the valid range [0, 450] cm, then discretized into 5 equal bins of 90 cm each.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)

def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. The track is 450 cm long, yielding 90 cm bins: [0-90), [90-180), [180-270), [270-360), [360-450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is discretized into 5 equal bins using `floor(position / 90)`, clipped to [0, 4]. Bin 0: 0-90cm, Bin 1: 90-180cm, Bin 2: 180-270cm, Bin 3: 270-360cm, Bin 4: 360-450cm.

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 450/5 = 90
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. This matches the instruction for "5 equal-sized bins."

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same as distance to reward zone: position is sliced from the same trial window indices as neural data.

ii.
```python
trial_pos = pos[s:e]
pos_disc = discretize_position(pos_clipped, n_bins=5)
output_data[1, :] = pos_disc
```

iii. Frame-by-frame alignment through shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` field in the NWB behavioral timeseries.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. The lick data represents lick sensor readings at each imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick sensor error correction is applied first: trials where >35% of samples have cumulative lick count > 2 have their lick data zeroed out. Then, lick data is binarized (>0 = 1, else 0).

ii.
```python
lick_corrected, error_trials = correct_lick_sensor_error(
    lick, tstart_inds, teleport_inds, correction_thr=LICK_CORRECTION_THR)
lick_binary = (lick_corrected > 0).astype(float)
# ...
trial_lick = lick_binary[s:e]
lick_disc = trial_lick.astype(int)
```

iii. The lick sensor error correction follows the reference code (`behavior.py:correct_lick_sensor_error`) with the same threshold (0.35). The paper reports ~0.65% of trials affected.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data shares the same time indices as neural data and is sliced with the same trial boundaries.

ii.
```python
trial_lick = lick_binary[s:e]
output_data[3, :] = lick_disc
```

iii. Same frame-level alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `position` and `reward_zone` fields. The position when `rzone > 0` is used to classify which zone (A, B, or C) is active.

ii.
```python
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```

iii. The reward zone positions are defined as A=[80,130], B=[200,250], C=[320,370] from the reference code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, positions where `rzone > 0` are collected. The median position is compared to known zone boundaries (+/- 20 cm tolerance) to classify as A, B, or C. Trials without reward zone activation (omission trials) inherit the zone label from neighboring trials via forward then backward fill. The label is encoded as 0=A, 1=B, 2=C.

ii.
```python
def determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds):
    zone_labels = [None] * n_trials
    for i in range(n_trials):
        s, e = tstart_inds[i], teleport_inds[i]
        rz_mask = rzone[s:e] > 0
        if np.any(rz_mask):
            rz_positions = pos[s:e][rz_mask]
            zone_labels[i] = get_reward_zone_label(rz_positions)
    # Forward fill, then backward fill
    # ...
    return zone_labels
# ...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The agent devised this approach to determine the active reward zone from the data, since it's not directly stored as a simple label. The forward/backward fill handles omission trials correctly.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` (sparse reward delivery timestamps) and `position/timestamps` (continuous frame timestamps), combined with trial boundaries.

ii.
```python
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
```

iii. Reward events are stored as sparse timestamps in the NWB file rather than as a continuous timeseries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks if any reward timestamp falls within the trial's time window (from trial start to teleport). If yes, outcome = 1 (rewarded); otherwise, outcome = 0 (omitted).

ii.
```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    n_trials = len(tstart_inds)
    outcomes = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
```

iii. The resulting omission rate (15.3%) matches the paper's reported ~15%, validating this approach.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Off-by-one temporal mismatches between behavioral and neural data are handled by truncating to the minimum length. (2) Lick sensor errors are corrected by zeroing affected trials. (3) Reward zone labels for omission trials are inherited from neighbors. (4) Environment values of -1 (during teleport) are filtered out. (5) Sessions with fewer than 2 cells or trials are skipped. (6) Multi-plane cell count assertions catch data integrity issues.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
# ... (truncate all to min_len)

# Lick sensor error correction
lick_corrected, error_trials = correct_lick_sensor_error(...)

# Environment: filter invalid values
env_valid = env_vals[env_vals >= 0]

# Reward zone: forward/backward fill for omission trials
```

iii. The agent documented these edge cases in CONVERSION_NOTES.md and implemented practical solutions. The off-by-one mismatch was discovered during multi-plane processing (m17/m18) where neural data had 1 extra timepoint.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading large NWB files with h5py (each contains full imaging sessions). (2) Computing dF/F for interneuron detection (`compute_dff_simple`), which involves neuropil subtraction, smoothing, and min/max filtering for each trial. (3) Computing Pearson correlations between dF/F and speed for every cell (`detect_interneurons`). (4) The overall loop through 152 sessions with full data loading.

ii.
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds, ...)
```

iii. The dF/F computation involves per-trial processing with scipy filters for each of ~80 trials, and interneuron detection requires Pearson correlation for each cell (up to 2000+). The full conversion produced a 9.16 GB output file.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The per-cell Pearson correlation loop in `detect_interneurons` could use matrix operations. (2) The per-trial environment type computation. (3) The per-trial reward outcome determination could use `np.searchsorted`. (4) The per-trial lick sensor error detection loop. (5) The per-trial reward zone determination loop.

ii.
```python
# Example: cell-by-cell correlation loop
for cell in range(n_cells):
    dff_valid = dff[valid_mask, cell]
    r, _ = stats.pearsonr(dff_valid, speed_valid)

# Example: per-trial environment loop
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    ...
```

iii. These loops iterate over cells (up to ~2300) or trials (up to ~100 per session), which could be replaced with vectorized numpy/scipy operations for significant speedup.

## 13-c. What processing does the code repeat multiple times?

i. (1) The trial boundary slicing `s, e = tstart_inds[i], teleport_inds[i]` is repeated in multiple functions (reward zone determination, lick correction, environment extraction, dF/F computation, and the main trial assembly loop). (2) The position data is accessed in both the reward zone determination pass and the main trial loop. (3) Valid environment filtering logic.

ii.
```python
# Repeated in determine_reward_zone_per_trial, correct_lick_sensor_error,
# compute_dff_simple, detect_interneurons, determine_reward_outcome, and main loop:
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
```

iii. While the code is modular (each function handles its own variable), it means trial boundary indexing and data slicing are performed redundantly across functions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The full fluorescence and neuropil data are loaded and processed solely for interneuron detection (dF/F computation), then discarded - the actual neural data uses pre-computed deconvolved activity. (2) The `reward_data` (reward magnitudes) are loaded but never used - only `reward_ts` (timestamps) are used. (3) The `planeIdx` variable is loaded but never used. (4) Speed is discretized as an output but no speed-based filtering is applied to neural data (unlike the reference code which uses a 2 cm/s threshold for spatial analyses).

ii.
```python
# Loaded but only used for interneuron detection, then discarded:
fluor_all = np.concatenate([fluor_p0, fluor_p1], axis=1)
neuro_all = np.concatenate([neuro_p0, neuro_p1], axis=1)

# Loaded but unused:
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
```

iii. The fluorescence/neuropil loading is necessary for interneuron detection but represents significant memory and I/O overhead. The `planeIdx` and `reward_data` variables are truly unused.
