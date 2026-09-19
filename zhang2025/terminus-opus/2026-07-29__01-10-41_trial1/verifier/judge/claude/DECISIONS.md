# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading `code/code_zhang2025/data/bwm_release.csv` to get a list of sessions with their lab, subject, date, and probe names. It then constructs file paths directly using `find_session_dir()` (building paths like `data/one_cache/{lab}/Subjects/{subject}/{date}/001`) and loads individual numpy/parquet files using `glob.glob()` and `np.load()` / `pd.read_parquet()`. It does NOT use the ONE API.

ii.
```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()
```

```python
def find_session_dir(lab, subject, date, base_dir='data/one_cache'):
    session_dir = os.path.join(base_dir, lab, 'Subjects', subject, date, '001')
    if os.path.exists(session_dir):
        return session_dir
    return None
```

iii. The AI chose direct file loading over the ONE API, using the bwm_release.csv as a session index. The CONVERSION_NOTES do not provide justification for this approach over the ONE API.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject name from `bwm_release.csv` (grouped by eid). Subjects are tracked in a `subject_map` dictionary that assigns indices incrementally as new subjects are encountered during processing.

ii.
```python
subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. No specific justification provided; subject identity comes from the bwm_release.csv table.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`, grouped with their associated probes. Each eid is processed independently.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({...}).reset_index()
```

iii. The session structure comes directly from the bwm_release.csv file.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI loads it and applies a mask to select valid trials.

ii.
```python
def load_trials(session_dir):
    trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
    trials_df = pd.read_parquet(trial_files[0])
    return trials_df
```

iii. The trials table is already organized as one row per trial; no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the following filters: reaction time between 0.08s and 2.0s, max trial length of 10.0s (feedback_times - goCue_times), NaN exclusion for stimOn_times/choice/feedback_times/probabilityLeft/firstMovement_times/feedbackType, no-choice exclusion (choice != 0), and behavioral data availability (wheel and whisker ME interpolation must succeed with at least 2 data points).

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
```

iii. The CONVERSION_NOTES cite the reference code's `load_trials_and_mask()` function (min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True) as the source of these parameters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spikes.times.npy and spikes.clusters.npy from each probe's pykilosort output, plus clusters.channels.npy and channels.brainLocationIds_ccf_2017.npy for brain region mapping.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
```

iii. These are the standard IBL spike sorting outputs.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over a 2s trial window (-0.5 to 1.5s around stimulus onset). The result is spike COUNTS (not firing rates -- the counts are NOT divided by bin width). When a session has multiple probes, clusters are merged by offsetting cluster IDs.

ii.
```python
def bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends):
    binned_all = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    for trial_idx in valid_indices:
        time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
        time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
        np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
    return binned_all
```

```python
neural_list = [neural_data[i].astype(np.float32) for i in range(n_valid_trials)]
```

iii. The CONVERSION_NOTES note the reference uses 20ms bins. The AI does not mention whether it should divide by bin width to get rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter clusters by quality. ALL clusters from the spike sorting are included, regardless of their quality label. The only filtering is that sessions without any spike data are skipped. Beryl mapping is applied for brain region labels, but no `void` or quality-based filtering is performed on the actual neural data.

ii.
```python
def load_spikes(session_dir, probe_names):
    # No quality filtering - all clusters kept
    spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
    spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
    # ...
    # Beryl mapping for region labels only
    beryl = br.acronym2acronym(acronyms, mapping='Beryl')
    # No is_good filter applied
```

iii. The CONVERSION_NOTES explicitly state: "Reference code uses ALL clusters (no QC filtering, qc=None)" based on the `prepare_data` function in the reference code's `ibl_data_utils.py`. The AI followed this interpretation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to stimulus onset (stimOn_times). The trial window starts at stimOn_times - 0.5s and ends at stimOn_times + 1.5s. Spike times within this window are binned relative to the trial start.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]   # stimOn - 0.5
trial_ends = stim_on + TIME_WINDOW[1]     # stimOn + 1.5
```

```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

iii. The AI notes alignment to stimOn_times matching the reference code's `align_time: 'stimOn_times'` parameter.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, 100 bins per trial. No rebinning is applied; spikes are directly binned at this resolution.

ii.
```python
BINSIZE = 0.02  # 20 ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize: 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a computed variable: an array of bin centers from -0.5 to 1.5s in 20ms steps, derived from the time window constants and bin size. It is the same for every trial.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. This is the neural binning grid itself.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The input is a deterministic sequence of bin centers: `[-0.49, -0.47, ..., 1.49]`.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. N/A -- derived from constants.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time array represents the centers of the same bins the spikes are binned into, so the alignment is exact by construction.

ii. Same bin grid used for spikes and time input.

iii. N/A.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where probabilityLeft changes value.

ii.
```python
def compute_trial_number_in_block(prob_left):
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
    return trial_nums
```

iii. No explicit justification provided in CONVERSION_NOTES.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected by comparing consecutive probabilityLeft values. The trial number within each block is counted starting from 1 (not 0). This is computed on ALL trials before filtering, then the valid indices are selected. The value is broadcast to all time bins as a per-trial constant.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

```python
inp = np.vstack([
    time_since_stim[np.newaxis, :],
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])
```

iii. The AI counts from 1; no justification is given for this choice vs counting from 0.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which uses IBL convention: +1 = left, -1 = right, 0 = no response.

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. No-choice trials (choice==0) are excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice values using `np.where(choice == -1, 0, 1)`. This maps IBL's -1 (right) to 0 and +1 (left) to 1. However, the instructions specify "left = 0, right = 1", so the mapping is REVERSED: the AI assigns left=1 and right=0.

ii.
```python
# Choice: -1 -> 0 (left), 1 -> 1 (right)    <-- comment is wrong about IBL convention
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The code comment says "-1 -> 0 (left)" but in IBL, -1 is RIGHT. The mapping contradicts the instruction's "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, 0.8.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The mapping matches the instruction: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping from probabilityLeft values to integer categories. Unknown values default to 1 (0.5). The value is broadcast to all time bins.

ii.
```python
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

```python
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. N/A.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, loaded directly as numpy files.

ii.
```python
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. These are the standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) Interpolate raw wheel position to 1000 Hz, (2) Apply Butterworth lowpass filter (order 8, corner freq 20 Hz) to the position, then differentiate to get velocity, (3) Take absolute value for speed. Note: the AI filters position then differentiates, while the standard brainbox approach (used by SessionLoader) differentiates then filters. This may produce slightly different results at edges.

ii.
```python
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
```

iii. The CONVERSION_NOTES reference the brainbox wheel.py functions (interpolate_position, velocity_filtered) but the AI reimplements them rather than using SessionLoader.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using quantile-based boundaries. The quantiles are computed across ALL valid trials and timepoints in a session (flattened), then np.digitize is applied.

ii.
```python
def discretize_time_varying(values, n_bins=N_DISC_BINS):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]   # [0.333, 0.667]
    boundaries = np.quantile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int64)
    return result
```

iii. The quantile approach creates approximately equal-sized bins within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The continuous wheel speed trace is interpolated to 100 uniform time bins per trial (bin centers from -0.49 to 1.49s), matching the neural data grid. The interpolation uses scipy.interpolate.interp1d with extrapolation.

ii.
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
y_interp = interpolate.interp1d(beh_t_clean, beh_v_clean, kind='linear',
                                fill_value='extrapolate')(x_interp)
```

iii. The AI uses the same bin centers as the neural data to ensure alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy`, with corresponding camera times from `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
def load_whisker_motion_energy(session_dir):
    me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
    # ... falls back to right camera
```

iii. Left camera preferred, matching reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. No processing of the raw motion energy trace. It is loaded as-is and interpolated to the trial time bins using the same `interpolate_behavior_to_bins` function as the wheel.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(me_times, me_values, trial_starts, trial_ends)
```

iii. The released motion energy values are used directly.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: quantile-based discretization into 3 bins, computed across all valid trials and timepoints in a session.

ii.
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. Same discretization function used for both wheel and whisker.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Interpolated to the same 100 bin centers as the neural data, using scipy.interpolate.interp1d with extrapolation.

ii. Same `interpolate_behavior_to_bins` function as wheel speed.

iii. Same alignment approach ensures temporal consistency.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Sessions without spike data, trials data, or session directory are skipped. (2) Trials with NaN in key fields are excluded by the trial mask. (3) Trials where wheel or whisker ME interpolation fails (fewer than 2 data points, NaN values) are marked invalid. (4) Sessions with fewer than 2 valid trials are skipped. (5) Wheel data with fewer than 10 samples is rejected.

ii.
```python
if spike_times is None:
    return None  # skip session
```
```python
if len(beh_v) < 2:
    valid[trial_idx] = False  # skip trial
```
```python
if n_valid < 2:
    return None  # skip session
```

iii. The AI handles missing data by dropping trials/sessions, which is reasonable.

## 10-a. What are the most time-consuming steps of the code?

i. Spike binning is the most time-consuming per-session step, iterating over all trials and using np.add.at for each trial. Loading spike data from disk is also expensive. The code processes sessions sequentially (no parallelization).

ii.
```python
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
```

iii. The code prints timing information for the binning step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops run per-trial: `bin_spikes_per_trial` loops over trials to bin spikes, and `interpolate_behavior_to_bins` loops over trials for behavior interpolation. The spike binning loop uses np.add.at inside, but the outer trial loop could potentially be replaced with a fully vectorized approach.

ii.
```python
for trial_idx in valid_indices:
    # ... spike binning per trial
```
```python
for trial_idx in range(n_trials):
    # ... behavior interpolation per trial
```

iii. No justification provided for not vectorizing.

## 10-c. What processing does the code repeat multiple times?

i. The behavior interpolation function `interpolate_behavior_to_bins` is called separately for wheel and whisker, each doing similar per-trial interpolation. The input/output list construction loops over trials a second time after the main processing.

ii.
```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(wheel_times, wheel_speed, ...)
me_binned, me_valid = interpolate_behavior_to_bins(me_times, me_values, ...)
```

iii. The repetition is inherent in processing two different signals.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins ALL clusters (including low-quality ones that would typically be filtered), which means it processes far more neural data than necessary. The code also computes spike binning for ALL trials before applying the trial mask, rather than filtering trials first.

ii.
```python
# Bins spikes for ALL trials first
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
# Then applies mask
neural_data = binned_spikes[valid_idx]
```

iii. No justification provided for this ordering.
