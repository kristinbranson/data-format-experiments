# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subjects by listing directories starting with `sub-` in the data directory, then finds all `.nwb` files in each subject directory. Each NWB file is loaded using `h5py` (not `pynwb`) and data is read directly from HDF5 paths.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subjects = [d.replace('sub-', '') for d in subjects_dirs]
...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
with h5py.File(nwb_path, 'r') as f:
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    ...
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
```

iii. The AI notes in CONVERSION_NOTES.md that the data contains 11 switch mice matching the paper. It uses h5py for direct HDF5 access rather than pynwb.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories starting with `sub-` in the data directory. The `sub-` prefix is stripped to get the subject ID.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. The AI verifies 11 subjects match the paper's description of 11 switch mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session number is parsed from the filename.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
session_num = fname.split('ses-')[1].split('_')[0]
```

iii. The AI notes sessions correspond to individual recording days, matching the paper's structure.

## 1-d. How are the data split into trials?

i. Trial starts are identified from the `trial_start` signal where it is > 0. Trial ends are identified from the `teleport` signal where it is > 0. The AI uses `np.where(teleport > 0)` directly to find teleport indices, rather than detecting teleport onset transitions.

ii.
```python
tstart_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport > 0)[0]

# Ensure matching number of starts and teleports
n_trials = min(len(tstart_inds), len(teleport_inds))
tstart_inds = tstart_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]

# Remove trials where teleport comes before or at trial start
valid = teleport_inds > tstart_inds
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
```

iii. The CONVERSION_NOTES say trials go from `trial_start` to `teleport`, matching the paper's definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Additionally, sessions with fewer than 2 valid trials are skipped entirely. There is no minimum trial length threshold beyond 2 timepoints.

ii.
```python
if n_timepoints < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. The AI applies minimal filtering, relying on the trial boundary detection to produce valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field in the ophys processing module, specifically `processing/ophys/Deconvolved/plane0/data` (and `plane1` for multi-plane sessions).

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
if has_multi_plane:
    deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
```

iii. The CONVERSION_NOTES state: "Used deconvolved activity (not dF/F) as neural data, matching the paper's use of deconvolved activity for most analyses."

## 2-b. How is the `neural` data processed?

i. For multi-plane sessions, data from plane0 and plane1 are concatenated along the cell axis. No further processing (no rebinning, normalization, etc.) is applied to the deconvolved data itself.

ii.
```python
if has_multi_plane:
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
else:
    deconv_all = deconv_p0
```

iii. The AI notes multi-plane data is pooled following the paper's approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) cells must pass manual curation (`iscell[:,0] == 1`), and (2) putative interneurons are excluded. Interneurons are detected by computing a simplified dF/F from raw fluorescence (with neuropil subtraction) and checking if the Pearson correlation between dF/F and running speed exceeds 0.5.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
deconv_filtered = deconv_all[:, cell_mask]
```

iii. The CONVERSION_NOTES say: "Interneuron exclusion: Cells with Pearson correlation > 0.5 between simplified dF/F and running speed are excluded" and references the paper's methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by indexing from the trial start index to the teleport index. No additional temporal shifting is needed since the alignment event is the trial start itself.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The CONVERSION_NOTES state: "Temporal alignment: start of each trial."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to `1000.0 / 15.5078125 = 64.48 ms`, corresponding to the 2-photon imaging frame rate. No temporal rebinning is applied.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ms (1/imaging_rate * 1000)
```

iii. The AI hardcodes the imaging rate rather than reading it dynamically from the data. The CONVERSION_NOTES confirm: "64.48 ms (1/15.5078125 Hz) Matches the 2-photon imaging frame rate."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from start is computed from the imaging rate, not from timestamps. It is calculated as `np.arange(n_timepoints) * frame_time` where `frame_time = 1.0 / imaging_rate`.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The AI computes time synthetically from the imaging rate rather than using the stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A time vector is created starting at 0, incrementing by `1/imaging_rate` seconds per frame. This assumes perfectly uniform frame timing.

ii.
```python
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. No explicit justification given in CONVERSION_NOTES beyond the imaging rate value.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Since both are indexed by the same frame indices (s:e), they are inherently aligned. The time vector has exactly `n_timepoints = e - s` entries matching the neural data.

ii.
```python
n_timepoints = e - s
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. Both use the same indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` field in the behavioral time series.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
env_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    env_valid = env_vals[env_vals >= 0]
    if len(env_valid) > 0:
        env_per_trial[i] = int(np.median(env_valid))
```

iii. The CONVERSION_NOTES say: "Directly from NWB environment field: 0 = ENV1, 1 = ENV2."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the environment values within the trial are filtered for non-negative values, and the median is taken. The value is broadcast as constant across all timepoints in the trial.

ii.
```python
env_vals = env[s:e]
env_valid = env_vals[env_vals >= 0]
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
else:
    env_per_trial[i] = 0
```

iii. The filtering of negative values and taking the median is a defensive measure, though the environment value should be constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 1-indexed loop counter over trials within a session.

ii.
```python
trial_number = float(i + 1)
```

iii. No justification in CONVERSION_NOTES. The stored `trial number` variable in the NWB file was not used.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply `i + 1` (1-indexed), broadcast as constant across all timepoints in the trial.

ii.
```python
input_data[2, :] = trial_number  # where trial_number = float(i + 1)
```

iii. No processing beyond the loop index plus 1.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward outcome array which is computed by matching sparse reward event timestamps to trial time windows.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The CONVERSION_NOTES explain: "Determined by matching sparse reward event timestamps to trial time windows."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window. The previous trial's outcome is then used as the input for the current trial. For the first trial, previous outcome is set to 0.

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
...
if i == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(reward_outcomes[i - 1])
```

iii. The AI uses timestamps for reward matching rather than index-based lookup, and correctly looks at the previous trial's outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. Reward zone per trial is determined from the `reward_zone` field (where rzone > 0) and position, with forward/backward fill for trials without rzone activation.

ii.
```python
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
...
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The CONVERSION_NOTES say: "Reward zone location determined from position data when reward_zone field > 0. Zone A: 80-130 cm, Zone B: 200-250 cm, Zone C: 320-370 cm. For omission trials, zone inherited from neighboring trials via forward/backward fill."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone. Negative = before zone, 0 = inside zone, positive = past zone. Then discretized into 7 bins.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position, dtype=float)
    before = position < rz_start
    distance[before] = position[before] - rz_start
    distance[after] = position[after] - rz_end
    return distance

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

iii. The AI implements the same distance computation as the reference. The discretization matches the specified bins in the instructions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Manual conditional assignment into 7 bins: `< -50`, `-50 to -10`, `-10 to < 0`, `0`, `>0 to +10`, `+10 to +50`, `> +50`.

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

iii. Matches the instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (s:e) used for both neural and behavioral data, so they are inherently aligned.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
```

iii. Both use the same indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
```

iii. Position directly records the animal's location in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, 450]` range, then discretized into 5 equal bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360-450) using `floor(position / 90)`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 450/5 = 90
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
...
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
```

iii. The CONVERSION_NOTES say: "Position discretized into 5 equal 90cm bins (0-90, 90-180, 180-270, 270-360, 360-450)."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Uses `floor(position / 90)` clipped to [0, 4], giving 5 bins of 90 cm each from 0 to 450.

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 90
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. Track length is 450 cm with 5 bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices (s:e) used for both neural and position data.

ii.
```python
trial_pos = pos[s:e]
```

iii. Both use the same indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. The `lick` field records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick sensor error correction is applied first: trials where >35% of samples have lick count > 2 have their lick data set to 0. Then lick data is binarized (>0 maps to 1).

ii.
```python
def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    licks_corrected = np.copy(licks)
    for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
        trial_licks = licks_corrected[s:e]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > correction_thr:
                licks_corrected[s:e] = 0
    return licks_corrected, error_trials
...
lick_binary = (lick_corrected > 0).astype(float)
```

iii. The CONVERSION_NOTES reference the paper's behavior.py for lick sensor error correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices (s:e) used.

ii.
```python
trial_lick = lick_binary[s:e]
```

iii. Both use the same indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series. For each trial, positions where `reward_zone > 0` are collected and the median position is matched to the closest known zone (A, B, or C with +/-20 cm tolerance).

ii.
```python
def get_reward_zone_label(position_when_in_rzone):
    if len(position_when_in_rzone) == 0:
        return None
    median_pos = np.median(position_when_in_rzone)
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 <= median_pos <= end + 20:
            return label
    return None

def determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds):
    for i in range(n_trials):
        s, e = tstart_inds[i], teleport_inds[i]
        rz_mask = rzone[s:e] > 0
        if np.any(rz_mask):
            rz_positions = pos[s:e][rz_mask]
            zone_labels[i] = get_reward_zone_label(rz_positions)
    # Forward/backward fill for missing
    ...
```

iii. The CONVERSION_NOTES describe the forward/backward fill approach for trials without rzone activation.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The median position when in the reward zone is matched to the closest known zone. Trials without rzone activation (omission trials) inherit the zone label from neighboring trials via forward then backward fill. The zone label is encoded as 0=A, 1=B, 2=C.

ii.
```python
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_data[4, :] = rz_loc
```

iii. See 10-a.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
```

iii. The CONVERSION_NOTES say: "Determined by matching sparse reward event timestamps to trial time windows."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window (trial_start to teleport). Binary: 1 if any reward, 0 otherwise.

ii.
```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    outcomes = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
```

iii. Uses timestamp matching rather than index-based matching.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data truncated to the minimum length of behavioral and neural arrays.
- **Trials with < 2 timepoints**: Skipped.
- **Missing reward zone data**: Forward/backward fill from neighboring trials.
- **Lick sensor errors**: Trials with >35% high-lick samples have lick data zeroed out.
- **Sessions with < 2 valid trials or < 2 cells**: Entire session skipped.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
if n_timepoints < 2:
    continue
...
if n_kept < 2:
    return None
```

iii. The AI applies defensive checks throughout the conversion process.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py (I/O bound, reading large neural arrays)
2. Computing dF/F for interneuron detection (requires loading fluorescence and neuropil data in addition to deconvolved data)
3. Computing Pearson correlations for all cells for interneuron detection
4. Saving the pickle file

ii. N/A

iii. The interneuron detection adds significant overhead compared to the reference, which does not perform this step.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron detection loop over cells (`detect_interneurons`) computes Pearson correlations cell-by-cell:
```python
for cell in range(n_cells):
    dff_valid = dff[valid_mask, cell]
    r, _ = stats.pearsonr(dff_valid, speed_valid)
```
This could be vectorized using matrix correlation. The reward outcome determination loop could also be vectorized.

ii. N/A

iii. These are relatively minor compared to I/O costs.

## 13-c. What processing does the code repeat multiple times?

i. No survey/conversion split exists in the AI code, so data is loaded only once per session. However, the interneuron detection requires loading fluorescence and neuropil data that are not used for anything else, representing extra I/O.

ii. N/A

iii. The AI code is more streamlined than the reference in this respect (single pass vs survey+convert).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron detection requires computing dF/F from fluorescence and neuropil data, which involves substantial processing (neuropil subtraction, smoothing, baseline estimation). This processing is only used to create a mask and the dF/F values themselves are discarded. Additionally, the lick sensor error correction processes all trials but only affects a small fraction (~0.65%).

ii.
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds, ...)
# dff is discarded after this
```

iii. The dF/F computation is only needed for interneuron detection.
