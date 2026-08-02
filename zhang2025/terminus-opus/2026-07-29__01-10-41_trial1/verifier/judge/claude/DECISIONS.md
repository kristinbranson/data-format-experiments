# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `code/code_zhang2025/data/bwm_release.csv`, groups by `eid` to identify sessions and their associated probes, then iterates over each session. For each session, it constructs the directory path in the ONE cache (`data/one_cache/{lab}/Subjects/{subject}/{date}/001/`) and loads trials from parquet files, spikes from numpy arrays, wheel from numpy arrays, and whisker motion energy from numpy arrays. Sessions are skipped if any essential data file (spikes, trials) is missing.

ii.
```python
# Load session info from bwm_release.csv
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()
# ...
for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    session_dir = find_session_dir(lab, subject, date)
    # ...
    result = process_session(session_info, session_dir, ...)
```

iii. The AI documented in CONVERSION_NOTES.md that bwm_release.csv contains 459 sessions and 699 probes, and that it processes all sessions with available data on disk. This matches the reference code's approach of using bwm_release.csv as the session manifest.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in the bwm_release.csv metadata. A mapping (`subject_map`) is built incrementally as sessions are processed, assigning each unique subject name an integer index. The `subjects` list and `subject_idx` array are constructed accordingly.

ii.
```python
subject_map = {}  # subject name -> index
# ...
subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. This is a straightforward mapping from session metadata. The CONVERSION_NOTES.md reports 139 subjects in the data, matching the paper's stated 139 mice.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from bwm_release.csv. Each row in `session_groups` (after grouping by eid) represents one session. The AI iterates over all sessions and processes each independently, producing one entry per session in the `neural`, `input`, and `output` lists.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({...}).reset_index()
for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    # process each session independently
```

iii. The reference code `0_data_caching.py` also iterates over sessions by eid. The AI's approach matches. 340 sessions were successfully processed out of 459 in the manifest (119 skipped due to missing data files on disk).

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` parquet files. Each trial corresponds to a row in the trials DataFrame. Trial time intervals are defined by the alignment event (`stimOn_times`) plus the time window `(-0.5, 1.5)` seconds. Neural and behavioral data are extracted per trial based on these time windows.

ii.
```python
trials_df = load_trials(session_dir)
# ...
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

iii. This matches the reference code's approach in `bin_spiking_data`, where intervals are computed as `trials_df[align_time] + time_window`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters via `create_trials_mask()`:
- Reaction time (firstMovement_times - stimOn_times) must be between 0.08s and 2.0s
- Trial length (feedback_times - goCue_times) must be <= 10.0s
- No NaN in: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType
- Exclude no-choice trials (choice == 0)
- Additionally, trials must have valid wheel and whisker motion energy data (at least 2 non-NaN interpolation points)

ii.
```python
def create_trials_mask(trials_df):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times',
                   'probabilityLeft', 'firstMovement_times', 'feedbackType']
    mask = pd.Series(True, index=trials_df.index)
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    if MIN_RT is not None:
        mask &= (rt >= MIN_RT)
    if MAX_RT is not None:
        mask &= (rt <= MAX_RT)
    if MAX_TRIAL_LEN is not None:
        trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN)
    for event in nan_exclude:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    mask &= (trials_df['choice'] != 0)
    return mask
# Later combined:
combined_mask = mask.values & wheel_valid & me_valid
```

iii. The CONVERSION_NOTES.md documents these criteria under Step 1 and Step 3, matching `load_trials_and_mask()` from the reference code. The reference code uses the same default parameters: min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True, and the same nan_exclude list.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from the pykilosort spike sorting directories for each probe.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
```

iii. The CONVERSION_NOTES.md Step 1 documents that the reference code uses `load_spiking_data` to load spike times and clusters, which is equivalent to loading these numpy files directly.

## 2-b. How is the `neural` data processed?

i. Spike times and cluster IDs from multiple probes are merged (with cluster ID offsets to avoid collisions), sorted by time, then binned into 20ms time bins within each trial's time window. Binning is done by computing `floor((spike_time - trial_start) / binsize)` to get bin indices, then using `np.add.at` to accumulate spike counts per cluster per bin.

ii.
```python
# Merge probes
merged_times = np.concatenate(all_spike_times)
merged_clusters = np.concatenate(all_spike_clusters)
sort_idx = np.argsort(merged_times, kind='stable')
# ...
# Bin per trial
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. The reference code uses `merge_probes` and `bincount2D` for equivalent operations. The AI's custom binning approach is functionally similar but uses a different algorithm (np.add.at vs bincount2D).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses ALL clusters without QC filtering (no quality-based neuron filtering), matching the reference code's `qc=None` parameter.

ii.
```python
# In load_spikes(), all clusters from all probes are loaded without filtering
n_clusters = len(clusters_channels)
# No filtering based on label/quality metrics
```

iii. The CONVERSION_NOTES.md Step 1 notes: "The code loads ALL clusters (not just good ones) - qc=None." The reference code `prepare_data` calls `load_spiking_data(one, pid, ...)` without passing qc, which defaults to `qc=None`, including all clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. The trial window is defined as `stimOn_times + (-0.5, 1.5)` seconds. For each trial, spikes within this window are binned relative to the window start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
# ...
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

iii. The task instructions say "Temporally align based on stimulus onset" and the reference code `0_data_caching.py` uses `align_time='stimOn_times'` with `time_window=(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 time bins per 2-second trial. No temporal rebinning is applied; this is the native binning resolution.

ii.
```python
BINSIZE = 0.02  # 20 ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The reference code uses `binsize=0.02` and interval_len=2.0, producing 100 bins. The methods paper states "divided into 20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from any raw data variable. It is computed analytically as the center of each time bin relative to stimulus onset time (which is time 0 by construction of the alignment).

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. Since trials are aligned to stimOn_times, the time axis is deterministic. The values range from -0.49s to 1.49s (bin centers), with the alignment event at time 0.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced array of bin center times is computed. Each value equals the bin start time plus half the bin width. This array is identical for all trials.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
# = np.arange(100) * 0.02 + (-0.5) + 0.01
# = [-0.49, -0.47, ..., 1.49]
```

iii. This is a straightforward computation. The same time array is shared across all trials as the alignment is always to stimOn_times.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset array and the neural data share the same 100-bin temporal structure by construction. Bin `i` of the neural data and bin `i` of the time input correspond to the same time window `[TIME_WINDOW[0] + i*BINSIZE, TIME_WINDOW[0] + (i+1)*BINSIZE]`. The input is broadcast to shape `(1, N_BINS)` for each trial.

ii.
```python
inp = np.vstack([
    time_since_stim[np.newaxis, :],  # (1, N_BINS)
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])
```

iii. Since both neural and time input use the same binning structure, alignment is guaranteed.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from `probabilityLeft` in the trials DataFrame. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The CONVERSION_NOTES.md notes that blocks are defined by changes in probabilityLeft (0.2, 0.5, or 0.8).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through all trials in order. A counter starts at 1 and increments for each trial. When `probabilityLeft` changes compared to the previous trial, the counter resets to 1. The computation is done on ALL trials first (including excluded ones), then valid trial indices are selected. The value is broadcast to all time bins within a trial.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
    return trial_nums
```

iii. This approach correctly counts trials within each block and resets on block boundaries. Computing on all trials before filtering ensures the block structure is preserved correctly.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials DataFrame (`_ibl_trials.table.pqt`).

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The reference code extracts choice directly from `trials_df['choice']` in `bin_behaviors`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL convention is: choice = -1 for left, choice = 1 for right, choice = 0 for no-choice (excluded). The AI maps -1 to 0 (left) and 1 to 1 (right). No-choice trials (choice == 0) are already excluded by the trial mask. The per-trial value is broadcast to all 100 time bins.

ii.
```python
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
# ...
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
```

iii. The task instructions specify "left = 0, right = 1" which matches this encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials DataFrame.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The reference code extracts block info from `trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The probabilityLeft values (0.2, 0.5, or 0.8) are mapped to categorical integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Any unexpected values default to 1 (0.5 category). The per-trial value is broadcast to all 100 time bins.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
# ...
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The task instructions specify "0.2 -> 0, 0.5 -> 1, 0.8 -> 2" which matches.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (wheel timestamps).

ii.
```python
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The reference code uses `SessionLoader.load_wheel()` which internally loads the same raw wheel position and timestamp arrays.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI re-implements the wheel processing pipeline from the reference `wheel.py`:
1. Load raw wheel position and timestamps
2. Interpolate position to 1000 Hz uniform sampling using linear interpolation
3. Apply Butterworth lowpass filter (order=8, corner frequency=20 Hz) using `sosfiltfilt`
4. Compute velocity as `diff(filtered_position) * fs`, prepend a zero
5. Speed = absolute value of velocity
6. Per trial: interpolate the speed signal to 100 uniform bins within the trial window using linear interpolation with extrapolation at edges

ii.
```python
# Interpolate to 1000 Hz
t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
# Butterworth filter and velocity
sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
```

iii. This matches the reference `interpolate_position()` and `velocity_filtered()` functions in `brainbox/behavior/wheel.py`. The reference `load_target_behavior` for 'wheel-speed' returns `np.abs(sess_loader.wheel['velocity'])`.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using quantile-based binning. All speed values across all trials and timepoints are pooled, quantile boundaries at 1/3 and 2/3 are computed, and `np.digitize` is used to assign each value to bins 0, 1, or 2.

ii.
```python
def discretize_time_varying(values, n_bins=N_DISC_BINS):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # [0.333, 0.667]
    boundaries = np.quantile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int64)
    return result
```

iii. The task instructions say "discretized into 3 bins" but do not specify the method. Quantile-based discretization ensures approximately equal representation of each category.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. The continuous wheel speed signal is interpolated to 100 uniform time bins within each trial's stimulus-aligned window `(-0.5, 1.5)s`. The interpolation uses `np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2` as interpolation targets (bin centers), with linear interpolation and extrapolation for edge values.

ii.
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
y_interp = interpolate.interp1d(
    beh_t_clean, beh_v_clean, kind='linear',
    fill_value='extrapolate')(x_interp)
```

iii. The reference code `get_behavior_per_interval` uses `np.linspace(interval_begs + binsize, interval_ends, n_bins)` as interpolation targets (right bin edges), which is slightly different from the AI's bin centers approach.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
# ...
me = np.load(me_file).flatten()
cam_times = np.load(cam_file).flatten()
```

iii. The reference code `bin_behaviors` loads 'whisker-motion-energy' which internally tries 'left-whisker-motion-energy' first, then 'right-whisker-motion-energy'.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional processing of the signal itself). Per trial, the values within the trial window are interpolated to 100 uniform bins using linear interpolation.

ii.
```python
# In load_whisker_motion_energy:
me = np.load(me_file).flatten()
cam_times = np.load(cam_file).flatten()
# Then in interpolate_behavior_to_bins:
y_interp = interpolate.interp1d(beh_t_clean, beh_v_clean, kind='linear',
    fill_value='extrapolate')(x_interp)
```

iii. The reference code also loads the raw motion energy and interpolates per trial using `get_behavior_per_interval`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: quantile-based discretization into 3 bins across all trials and timepoints.

ii.
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. Same as wheel speed (question 9-c).

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: the continuous signal is interpolated to 100 uniform time bins within the stimulus-aligned trial window, using bin centers as interpolation targets.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. Same method as wheel speed (question 9-d).

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data scenarios are handled:
- Sessions with missing spike data, trial data, wheel data, or motion energy data are skipped entirely
- Trials where wheel or motion energy have fewer than 2 non-NaN interpolation points are excluded
- NaN values in behavior signals are filtered before interpolation
- Sessions with fewer than 2 valid trials are skipped
- For `probabilityLeft` values not in {0.2, 0.5, 0.8}, the default category 1 is used
- Trial time intervals with NaN start/end times are skipped during spike binning
- Behavior interpolation uses `fill_value='extrapolate'` to handle edge effects

ii.
```python
# Skip sessions with missing data
if spike_times is None:
    return None
# Behavior validity check
if len(beh_v) < 2:
    valid[trial_idx] = False
# NaN handling in behavior
nan_mask = ~np.isnan(beh_v)
if nan_mask.sum() < 2:
    valid[trial_idx] = False
# Minimum trials
if n_valid < 2:
    return None
```

iii. The CONVERSION_NOTES.md documents that sessions with missing data are skipped. The reference code handles missing data through the `align_spike_behavior` function and the behavior mask system.

## 12-a. What are the most time-consuming steps of the code?

i. Based on the timing information printed during conversion, spike binning is the most time-consuming step per session. For a session with ~1300 clusters and ~800 trials, binning takes 1.5-2.1 seconds. Wheel processing (interpolation, filtering) and behavior interpolation to trial bins are secondary bottlenecks. Total per-session time ranges from 1.3s to 4.9s.

ii.
```python
print(f"  Binning spikes ({n_clusters} clusters, {len(trials_df)} trials)...", end=' ')
# Binning takes 0.3s-2.1s per session
```

iii. The conversion output shows per-session timing. Total conversion of 340 sessions took approximately 15-20 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level spike binning loop iterates over each valid trial sequentially:
```python
for trial_idx in valid_indices:
    # searchsorted + np.add.at per trial
```
This could potentially be vectorized using a single bincount2D call across all trials, similar to the reference code. Additionally, the behavior interpolation loop iterates over trials:
```python
for trial_idx in range(n_trials):
    # interp1d per trial
```
This is inherently hard to vectorize due to per-trial time ranges, but the reference code parallelizes it using multiprocessing.

ii.
```python
# Spike binning loop
for trial_idx in valid_indices:
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    # ...
    np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)

# Behavior interpolation loop
for trial_idx in range(n_trials):
    y_interp = interpolate.interp1d(...)(x_interp)
```

iii. The reference code uses multiprocessing (`multiprocessing.Pool`) for both spike binning and behavior interpolation. The AI's code does not use multiprocessing.

## 12-c. What processing does the code repeat multiple times?

i. The wheel processing pipeline (interpolation to 1000 Hz + Butterworth filter + velocity computation) is done once per session, which is appropriate. However, the behavior interpolation to trial bins (`interpolate_behavior_to_bins`) is called separately for wheel and whisker ME, each performing a similar loop over all trials. These two calls could potentially be combined.

Additionally, the `fill_value='extrapolate'` creates a new `interp1d` object for each trial, which involves some overhead. The searchsorted operations are done independently for spike binning and for each behavior signal.

ii.
```python
# Two separate calls to interpolate_behavior_to_bins:
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. This is not a major efficiency concern but represents a minor opportunity for optimization.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes for ALL trials first (including those that will be filtered out), then applies the combined mask. This means spike binning is performed on trials that are subsequently discarded. For the first session, 692 trials are binned but only 198 survive the mask. This wastes approximately 71% of the spike binning computation for that session.

Similarly, behavior interpolation is done for all trials before filtering. Wheel and motion energy are interpolated for all trials, then valid trials are selected.

ii.
```python
# Bin ALL trials
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
# ...
# Filter after binning
valid_idx = np.where(combined_mask)[0]
neural_data = binned_spikes[valid_idx]
```

iii. While the reference code also processes all trials before filtering (for behavior data), a more efficient approach would be to apply the trial mask before the expensive spike binning step. However, the behavior validity check requires interpolation first (to determine which trials have valid data), creating a chicken-and-egg situation that the AI resolved by processing everything first.
