# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subjects by scanning for `sub-*` directories in the data directory, then finds all `.nwb` files within each subject directory using `glob`. Files are loaded using `h5py` (direct HDF5 access) rather than `pynwb`. All behavioral and neural data arrays are read directly from the HDF5 group paths.

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

iii. The agent explored the directory structure and NWB file contents using `h5py` in early trajectory steps, confirming the `sub-*` directory convention and the internal HDF5 paths.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories matching `sub-*` in the data directory. The subject ID is extracted by removing the `sub-` prefix.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. The agent noted 11 subjects were found, matching the paper's description.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session number is parsed from the filename (e.g., `ses-03` from `sub-m11_ses-03_behavior+ophys.nwb`).

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
session_num = fname.split('ses-')[1].split('_')[0]
```

iii. The agent confirmed 152 total sessions across 11 subjects, with 12-14 sessions each.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using the `trial_start` and `teleport` behavioral time series. Trial starts are where `trial_start > 0`, and trial ends are where `teleport > 0`. Trials where the teleport index is at or before the start index are removed.

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

iii. The agent identified `trial_start` and `teleport` as the relevant signals for trial boundaries and confirmed ~80.4 trials/session matching the paper's 80.5.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Sessions with fewer than 2 valid trials or fewer than 2 cells after filtering are also skipped.

ii.
```python
if n_timepoints < 2:
    continue
...
if n_trials < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
    return None
...
if n_kept < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_kept} cells after filtering")
    return None
```

iii. The agent set a minimal threshold for trial length (2 timepoints), compared to the reference's 50 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the `Deconvolved` data directly from the NWB file (suite2p's deconvolution stored in the NWB), not the raw Fluorescence and Neuropil traces. The Fluorescence and Neuropil are only loaded for a simplified dF/F computation used for interneuron detection.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
...
if has_multi_plane:
    deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
...
deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
...
deconv_filtered = deconv_all[:, cell_mask]
```

iii. The agent's docstring states "Neural data: deconvolved calcium activity (OASIS algorithm, already in NWB)" suggesting it believed the NWB's Deconvolved field was the paper's processed neural signal.

## 2-b. How is the `neural` data processed?

i. The neural data is taken directly from the NWB's `Deconvolved` field with no additional processing beyond cell filtering and length alignment. There is no recomputation of dF/F, no neuropil subtraction, no baseline correction, and no deconvolution by the AI — it uses the pre-stored suite2p deconvolution.

ii.
```python
deconv_filtered = deconv_all[:, cell_mask]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The agent did not discuss the distinction between suite2p's built-in deconvolution (stored in NWB) and the paper's custom dF/F + OASIS pipeline. The reference solution explicitly notes that "The NWB 'Deconvolved' array is NOT the signal the paper analyses."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` manual curation mask from suite2p, and (2) putative interneuron exclusion based on dF/F-speed correlation > 0.5. The dF/F for interneuron detection is computed using a simplified method (`compute_dff_simple`) that differs from the paper's exact pipeline.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
```

iii. The agent mentioned applying both `iscell` and interneuron filtering, matching the paper's methods. However, the simplified dF/F (uniform filter instead of Gaussian with sigma=15, no per-trial neuropil mean addition in the same way) could yield different interneuron classifications.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing from `tstart_inds[i]` to `teleport_inds[i]`, so the first sample of each trial corresponds to trial onset. No further alignment processing is needed.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The agent set `temporal_alignment_event` to "start of trial" in the metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The time bin size is set to `1000.0 / 15.5078125 ≈ 64.48 ms`, derived from the imaging rate. This is hardcoded rather than read per-session.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ms (1/imaging_rate * 1000)
```

iii. The agent used the imaging rate from the NWB file. However, for multi-plane sessions (2 planes), the effective per-plane rate would be half the scanner rate, which the reference handles with `nplanes/plane_data.rate*1000`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the imaging rate (a constant) rather than from the actual behavior timestamps. The AI computes time as `np.arange(n_timepoints) * frame_time` where `frame_time = 1.0 / imaging_rate`.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The agent used the imaging rate from the NWB acquisition metadata to derive time, rather than using the stored behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as a linear ramp from 0, incrementing by `1/imaging_rate` per timepoint. The first timepoint is always 0 seconds.

ii.
```python
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. This produces a uniform time grid. The reference instead uses actual timestamps with the first timestamp subtracted: `timestamps_curr - timestamps_curr[0]`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Since both neural and time-from-start are indexed by the same `[s:e]` range and time is computed from `np.arange(n_timepoints)`, they are inherently aligned.

ii.
```python
n_timepoints = e - s
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. Both use the same trial indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
env_vals = env[s:e]
env_valid = env_vals[env_vals >= 0]
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
```

iii. The agent recognized that environment is binary (0 or 1) and constant within each trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the median of valid (>= 0) environment values is taken. This effectively selects the most common value within the trial.

ii.
```python
env_valid = env_vals[env_vals >= 0]
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
else:
    env_per_trial[i] = 0
```

iii. Since environment is constant within a trial, the median is equivalent to just reading the value. The filtering of negative values is a defensive measure.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 1-indexed loop counter over trials within a session, derived from the trial iteration variable `i + 1`.

ii.
```python
trial_number = float(i + 1)
```

iii. The agent used a sequential index rather than any stored trial number variable from the NWB file.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply `i + 1` (1-indexed), broadcast as a constant across all timepoints.

ii.
```python
trial_number = float(i + 1)
input_data[2, :] = trial_number
```

iii. No additional processing. The reference uses 0-indexed (`trial`).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps and the per-trial time windows. The `determine_reward_outcome` function matches reward timestamps to trial boundaries.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The agent used reward timestamps to determine whether a reward event occurred within each trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the previous trial is used. For the first trial, it defaults to 0. The reward outcome per trial is determined by checking if any reward timestamp falls within the trial's time window.

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

iii. Similar approach to the reference: check for reward events in the previous trial's time window.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. The reward zone label per trial is determined using the `determine_reward_zone_per_trial` function which matches reward zone activation positions to known zone ranges (A=[80,130], B=[200,250], C=[320,370]).

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
...
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
...
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The agent used the median position where `reward_zone > 0` to classify each trial's zone, with forward/backward filling for trials without zone activation.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from position to the nearest edge of the reward zone: negative before the zone, 0 inside, positive after. Then discretized into 7 bins.

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

iii. Same logic as the reference for computing distance. The discretization uses explicit conditional assignments rather than `np.digitize`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using conditional array assignments matching the instruction bins.

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

iii. The bin edges match the instructions. The boundary handling differs slightly from the reference: `distance == 0` for bin 3 (exactly 0) vs the reference's `[0, 1e-6)` range using `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data (`[s:e]`), so no additional alignment needed.

ii.
```python
trial_pos = pos[s:e]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Both use the same trial index range.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
```

iii. Direct use of the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] range, then discretized into 5 equal bins of 90 cm using `floor(position / 90)`, clipped to [0, 4].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
...
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
```

iii. The clipping ensures all positions fall within the track range before discretization.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each: [0,90), [90,180), [180,270), [270,360), [360,450]. Uses `floor(position/90)` clamped to [0,4].

ii.
```python
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. This effectively produces bins at [0,90), [90,180), [180,270), [270,360), [360,450], which is equivalent to the reference's `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]` except at exact boundaries (e.g., position=90 goes to bin 1 in both).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
trial_pos = pos[s:e]
```

iii. Both use the same `[s:e]` range.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Direct use of the lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error correction before binarization. Trials where more than 35% of samples have lick count > 2 are set to 0 (considered erroneous). Then licks are binarized: any positive value becomes 1.

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
...
trial_lick = lick_binary[s:e]
lick_disc = trial_lick.astype(int)
```

iii. The agent referenced `behavior.py: correction_thr=0.35` from the paper's code for the lick error correction threshold.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
trial_lick = lick_binary[s:e]
```

iii. Both use the same `[s:e]` range.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavioral time series, as described in 7-a. The zone label per trial is determined by matching the median position where `reward_zone > 0` to the known zone ranges (A, B, C). Trials without zone activation are filled from neighboring trials.

ii.
```python
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_data[4, :] = rz_loc
```

iii. The agent used the `reward_zone` variable's activation pattern and position to classify zones, with forward/backward filling for missing labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial: (1) find timepoints where `reward_zone > 0`, (2) compute median position at those timepoints, (3) match to the nearest known zone range (A/B/C) with ±20 cm tolerance, (4) forward-fill then backward-fill for trials without zone activation. Then map A=0, B=1, C=2.

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
```

iii. The reference uses a Viterbi algorithm for more robust zone assignment across trials; the AI's approach is simpler but less robust to noisy data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps and the per-trial time windows.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. Reward timestamps are matched to trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within `[t_start, t_end]` (using behavior timestamps at trial boundaries). Binary: 1 if rewarded, 0 if not.

ii.
```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
...
rew_out = int(reward_outcomes[i])
output_data[5, :] = rew_out
```

iii. Similar to the reference approach, but uses timestamp-based matching rather than index-based matching with `searchsorted`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: All data arrays are truncated to the minimum length of behavioral and neural data.
- **Short trials**: Trials with < 2 timepoints are skipped.
- **Missing reward zone data**: Trials without `reward_zone > 0` activation are filled from neighboring trials (forward then backward fill). If no zone can be determined, defaults to 'A'.
- **Lick sensor errors**: Trials with excessive lick rates are zeroed out.
- **Invalid environment values**: Negative environment values are filtered out, with fallback to 0.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
if n_timepoints < 2:
    continue
...
if rz_label is None:
    rz_label = 'A'  # fallback
```

iii. The agent implemented defensive checks for data alignment issues discovered during development (off-by-one errors between neural and behavioral data).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with `h5py` and reading all data arrays per session
2. **Computing simplified dF/F** for interneuron detection (`compute_dff_simple` with filtering operations)
3. **Interneuron detection** with per-cell Pearson correlations
4. **Iterating over all 152 sessions** with full data loading

ii. N/A

iii. The agent noted conversion took several minutes for the full dataset.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially; discretization operations could be applied to full session arrays. The `determine_reward_outcome` function loops per trial. The `detect_interneurons` function loops per cell for correlation computation. The `correct_lick_sensor_error` function loops per trial.

ii. N/A

iii. N/A

## 13-c. What processing does the code repeat multiple times?

i. The code loads each NWB file once per session and processes it in a single pass. No significant repeated processing.

ii. N/A

iii. N/A

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes a simplified dF/F (`compute_dff_simple`) solely for interneuron detection, but this dF/F is never used for the actual neural data output (which uses the NWB's pre-stored Deconvolved). The lick sensor error correction is an extra processing step not present in the reference. The `create_sample` function creates a sample dataset that isn't part of the required output. The `print_sanity_checks` function produces diagnostic output.

ii.
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
...
sample = create_sample(data)
```

iii. The dF/F computation is needed for interneuron detection but is otherwise discarded.
