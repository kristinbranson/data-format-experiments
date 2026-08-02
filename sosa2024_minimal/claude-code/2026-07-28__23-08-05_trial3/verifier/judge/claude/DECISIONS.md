# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subject directories matching `sub-*` in the data directory, then finds all `.nwb` files within each subject directory. Each NWB file corresponds to one session. Data is loaded using `h5py` (not `pynwb`), reading directly from HDF5 paths within the NWB file structure.

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

iii. The AI chose h5py for direct HDF5 access. It found 11 subjects matching the paper. The sorted directory listing ensures reproducible ordering.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of the data directory matching `sub-*`. The subject ID is extracted by removing the `sub-` prefix.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. Found 11 subjects matching the paper's count of 11 switch mice.

## 1-c. How are the data split into sessions?

i. Each NWB file within a subject's directory corresponds to one session. Session numbers are parsed from the filename.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
fname = os.path.basename(nwb_file)
session_num = fname.split('ses-')[1].split('_')[0]
```

iii. Files follow the naming convention `sub-<id>_ses-<num>_behavior+ophys.nwb`.

## 1-d. How are the data split into trials?

i. Trial start is identified by `trial_start > 0`. Trial end is identified by `teleport > 0`. The AI takes ALL timepoints where teleport > 0 (not just the onset), then truncates to match the number of trial starts.

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
```

iii. The AI uses trial_start and teleport signals. However, it does NOT detect the onset of teleport (transition from <=0 to >0) like the reference does. Instead it uses all timepoints where teleport > 0. If teleport is active for multiple consecutive frames, this would produce many more teleport indices than trial starts, and the first N would not correctly correspond to trial end times.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Trials where teleport comes before or at trial start are removed.

ii.
```python
valid = teleport_inds > tstart_inds
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
...
if n_timepoints < 2:
    continue
```

iii. The minimum trial length threshold is 2 timepoints (vs. 50 in the reference). The reference uses a more conservative filter of 50 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field in the NWB ophys processing module (`processing/ophys/Deconvolved/plane0/data` and `plane1/data` for multi-plane).

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
if has_multi_plane:
    deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
```

iii. The paper describes using deconvolved calcium activity for analyses.

## 2-b. How is the `neural` data processed?

i. Multi-plane data is concatenated. Cells are filtered by iscell (manual curation) AND putative interneuron exclusion (dF/F-speed correlation > 0.5). The interneuron detection involves computing dF/F from raw fluorescence with neuropil subtraction, then correlating with running speed.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
deconv_filtered = deconv_all[:, cell_mask]
```

iii. The AI implemented interneuron exclusion based on the paper's methods section, which describes excluding putative interneurons with Pearson correlation > 0.5 between dF/F and running speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters applied: (1) iscell manual curation flag == 1, and (2) putative interneuron exclusion based on dF/F-speed correlation > 0.5.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
cell_mask = curated_mask & ~is_interneuron
```

iii. The reference code only uses `iscell`. The AI adds interneuron exclusion, which the paper mentions but the reference code does not implement for the decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to trial start. Data is sliced from trial_start index to teleport index, so the first timepoint of each trial corresponds to the trial start event.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. Straightforward alignment to trial start by slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning applied. The time bin size is computed from the imaging rate: `1000.0 / 15.5078125 = 64.48 ms`. The neural data is kept at the stored imaging frame rate.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ms (1/imaging_rate * 1000)
```

iii. The imaging rate is read from the NWB file but the metadata hardcodes 15.5078125 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the imaging rate, NOT from stored timestamps. It is computed as `np.arange(n_timepoints) * frame_time` where `frame_time = 1.0 / imaging_rate`.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The AI computes time from the imaging rate rather than using actual timestamps. The reference uses the actual behavior timestamps (`timestamps_curr - timestamps_curr[0]`).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Frame indices are multiplied by the frame duration (1/imaging_rate). The first timepoint is 0 seconds.

ii.
```python
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. This assumes perfectly regular sampling, which should be approximately true but may differ slightly from actual timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same frame indices, so they are inherently aligned. The time array has the same number of timepoints as the neural data.

ii.
```python
n_timepoints = e - s
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. Since both are indexed by the same trial boundaries, alignment is implicit.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

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

iii. The AI takes the median of valid (>= 0) environment values within each trial, treating it as a per-trial variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of non-negative environment values within the trial is taken. This value is then broadcast to all timepoints within the trial.

ii.
```python
env_valid = env_vals[env_vals >= 0]
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
else:
    env_per_trial[i] = 0
...
input_data[1, :] = env_type
```

iii. The reference simply uses the raw environment values per timepoint without taking a median. Since environment should be constant within a trial, the median approach should yield the same result in most cases.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the loop counter over trials (1-indexed: `i + 1`).

ii.
```python
trial_number = float(i + 1)
input_data[2, :] = trial_number
```

iii. Uses the sequential trial index within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The 0-based loop index is incremented by 1 to get a 1-indexed trial number. The value is constant across all timepoints within the trial.

ii.
```python
trial_number = float(i + 1)
```

iii. The reference uses 0-indexed trial numbers (`input_curr[2] = trial`). The AI uses 1-indexed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from sparse `Reward` event timestamps in the NWB file, matched to trial time windows.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. Reward events are stored with their own timestamps, separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event timestamp falls within the trial's time window. The previous trial's outcome is used as the input for the current trial. For the first trial, set to 0.

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

iii. Consistent with the reference approach. The reference uses `searchsorted` to map reward times to behavior indices, while the AI directly compares timestamps.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the reward zone location for the current trial. The reward zone location is determined from the `reward_zone` behavior variable and `position` data.

ii.
```python
trial_pos = pos[s:e]
rz_label = zone_labels[i]
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Same reward zone definitions as the reference: A=[80,130], B=[200,250], C=[320,370].

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to nearest edge of reward zone. Negative before zone, 0 inside zone, positive after zone.

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

iii. Same logic as the reference implementation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using explicit conditional logic into 7 bins matching the instructions.

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

iii. The reference uses `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The AI's explicit conditionals achieve the same result. Note: the boundary cases (e.g. exactly -50, exactly -10) differ slightly. The AI uses `>=` -50 for bin 1 and `>=` -10 for bin 2, which matches the instruction's "to" wording. The reference uses `np.digitize` which puts boundary values in the higher bin.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial slicing indices used for both neural and behavioral data, so alignment is implicit.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T
trial_pos = pos[s:e]
```

iii. Both use the same `s:e` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
```

iii. Position is clipped to [0, 450] before discretization.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, TRACK_LENGTH=450] and then discretized into 5 equal bins of 90 cm each.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
```

iii. The reference does NOT clip and uses different bin edges: `[-inf, 50, 150, 250, 350, inf]` (100 cm bins spanning approximately -50 to 450). The AI uses 90 cm bins: [0-90, 90-180, 180-270, 270-360, 360-450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each using `floor(position / 90)`, clipped to [0, 4].

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 450/5 = 90
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. The reference uses `np.digitize` with edges `[-inf, 50, 150, 250, 350, inf]`, giving 100 cm bins offset by -50. The AI's bins are 90 cm and start at 0. This produces different bin assignments for many positions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial slicing as neural data.

ii.
```python
trial_pos = pos[s:e]
```

iii. Same `s:e` slice as neural.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Direct from the NWB lick field.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick sensor error correction is applied first: trials where >35% of samples have lick count >2 are set to 0. Then licks are binarized (>0 = 1).

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

iii. The reference does NOT apply lick sensor error correction. It simply binarizes: `(licks_curr > 0).astype(int)`. The AI's correction is based on the paper's reference code (`behavior.py`).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slicing as neural data.

ii.
```python
trial_lick = lick_binary[s:e]
```

iii. Same `s:e` slice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` data. The median position when reward_zone > 0 is compared to known zone boundaries with a tolerance of 20 cm.

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
    # Forward fill, then backward fill for missing
    ...
```

iii. The reference uses a Viterbi algorithm for reward zone assignment, which considers transition probabilities and Gaussian emission on zone distance. The AI uses a simpler approach with median position matching and forward/backward fill for omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial where reward_zone > 0, take median position and match to nearest known zone (with 20 cm tolerance). For trials without reward_zone activation, forward-fill then backward-fill from neighboring trials. Map to integers: A=0, B=1, C=2.

ii.
```python
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_data[4, :] = rz_loc
```

iii. The reference's Viterbi approach is more robust to noise, but the AI's simpler approach should produce similar results in most cases since reward zone positions are typically well-separated.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from sparse `Reward` event timestamps matched to trial time windows.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. Same source as the reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within [trial_start_time, trial_end_time]. Binary output: 1 if reward occurred, 0 otherwise.

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

iii. The reference uses `searchsorted` to map reward times to behavior indices and then checks per-trial. Both approaches should produce the same result.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Neural/behavior length mismatch**: Data truncated to minimum length.
- **Short trials**: Trials with < 2 timepoints skipped.
- **Missing reward zone**: Forward/backward fill from neighboring trials; fallback to 'A' if all else fails.
- **Invalid environment values**: Negative values filtered out, median of valid values used.
- **Lick sensor errors**: Trials with high fraction of high lick counts zeroed out.
- **Teleport before trial start**: Such trials filtered out.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
...
if n_timepoints < 2:
    continue
...
if rz_label is None:
    rz_label = 'A'  # fallback
```

iii. The reference handles neural/behavior length mismatch similarly (cropping to minimum) and filters short trials (with threshold of 50 vs AI's 2). The reference does not apply lick correction or environment value filtering.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py (I/O bound, reading large neural arrays)
2. Computing dF/F for interneuron detection (additional neural data processing not in reference)
3. Interneuron detection (Pearson correlation for each cell)
4. Saving the pickle file

ii. N/A

iii. The interneuron detection adds significant overhead compared to the reference, which only needs to read deconvolved data and iscell.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates sequentially over trials. The `determine_reward_outcome` function loops over trials to match reward timestamps. The `detect_interneurons` function loops over each cell for Pearson correlation. The `compute_dff_simple` function loops over trials for baseline computation.

ii. N/A

iii. The per-cell loop in `detect_interneurons` could use vectorized correlation. The reward outcome determination could use broadcasting.

## 13-c. What processing does the code repeat multiple times?

i. No explicit survey/conversion two-pass structure like the reference. However, the interneuron detection requires loading fluorescence AND neuropil data in addition to deconvolved data, which is extra I/O. The dF/F computation repeats neuropil subtraction and baseline computation that could theoretically be avoided if interneurons were pre-identified.

ii. N/A

iii. The AI's single-pass approach is actually more efficient than the reference's two-pass (survey then convert) structure.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron detection pipeline (dF/F computation from raw fluorescence + neuropil, followed by correlation with speed) is additional processing not present in the reference. Whether it's "unnecessary" depends on interpretation - the paper mentions it but the reference decoder code does not implement it. The lick sensor error correction is also additional processing not in the reference.

ii.
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds, ...)
...
lick_corrected, error_trials = correct_lick_sensor_error(...)
```

iii. These additional processing steps follow the paper's methods more closely but are not required by the decoder task instructions.
