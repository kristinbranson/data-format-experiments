# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading a CSV file (`bwm_release.csv`) from the Zhang reference code directory to get the list of sessions. It then navigates the ONE cache directory structure on disk directly (constructing file paths from lab, subject, date, session number) rather than using the ONE API. For each session, it loads trials, spikes, wheel, and whisker motion energy by finding and reading `.npy` and `.pqt` files directly.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
# ...
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
```

iii. The AI chose direct disk loading because "cache is read-only" (CONVERSION_NOTES.md Step 6). This bypasses the ONE API entirely, constructing paths manually from the CSV metadata.

## 1-b. How are the data split into subjects?

i. Subject names come from the `bwm_release.csv` file's `subject` column. Each session's subject is tracked, and a list of unique subjects is built incrementally during processing. Subject indices are assigned in encounter order (not sorted).

ii.
```python
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

iii. The subject information is directly available from the BWM release CSV, so no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Sessions are the rows in `bwm_release.csv`, grouped by `eid`. Each unique `eid` represents one session. Probes within a session are grouped together.

ii.
```python
session_groups = bwm_df.groupby('eid')
session_list = []
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((eid, row['lab'], row['subject'], row['date'],
                         row['session_number'], probe_names))
```

iii. The BWM release CSV already organizes data by session (eid), with one row per probe insertion.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) is loaded per session, providing one row per trial.

ii.
```python
trials_df = load_trials(session_path)
# ...
trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
return pd.read_parquet(trials_file)
```

iii. The trials table is inherently one row per trial; no additional splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with multiple criteria: (1) NaN exclusion on stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType; (2) reaction time between 0.08s and 2.0s; (3) trial length (feedback_times - goCue_times) <= 10s; (4) exclude no-choice trials (choice == 0); (5) behavior coverage masks from wheel speed and whisker ME interpolation (requiring data within one bin of trial edges).

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trials_mask(trials_df):
    mask = np.ones(n_trials, dtype=bool)
    for event in NAN_EXCLUDE:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    if MAX_TRIAL_LEN is not None:
        trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    mask &= (trials_df['choice'] != 0)
    return mask

combined_mask = mask & wheel_mask & me_mask
```

iii. The AI states these filters match the reference code's `load_trials_and_mask()` function, which includes NaN exclusion, RT bounds, no-choice exclusion, and max trial length filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Brain region information comes from `channels.brainLocationIds_ccf_2017.npy` mapped through `clusters.channels.npy`.

ii.
```python
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```

iii. These are the standard spike sorting output files from pykilosort.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms time bins over the 2s trial window (-0.5 to 1.5s around stimulus onset), producing 100 bins per trial. Spike counts are stored as uint8 (not converted to firing rates). When multiple probes exist, they are merged by offsetting cluster IDs and sorting by time.

ii.
```python
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
lin_idx = c * N_BINS + b
counts = np.bincount(lin_idx, minlength=minlength)
binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
# ...
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. The AI notes this matches `bin_spiking_data()` from the reference code, using bincount-based approach with 20ms bins. The uint8 format was chosen to minimize memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron quality filtering is applied.** All clusters from pykilosort are used regardless of quality label. The AI explicitly chose this based on the Zhang reference code which uses `qc=None`.

ii.
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1
# No filtering by label or quality metric
```

From CONVERSION_NOTES.md:
```
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)'
```

iii. The AI justified this by noting: "Zhang code: uses ALL neurons (qc=None), NOT just well-isolated neurons" and "We follow Zhang code: use ALL neurons (cluster label filtering is NOT applied)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the interval begins at `stimOn_times + (-0.5)` and ends at `stimOn_times + 1.5` (absolute times). Spikes are selected within this interval and binned relative to the interval start.

ii.
```python
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]  # stimOn - 0.5
interval_ends = stim_times + TIME_WINDOW[1]  # stimOn + 1.5
# In bin_spikes_fast:
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
```

iii. The AI confirms alignment to stimulus onset matching the reference code's unified approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms (0.02s), producing 100 bins over the 2s window. No rebinning is applied; spikes are binned directly into 20ms bins.

ii.
```python
BINSIZE = 0.02          # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. Matches the reference code's `'binsize': 0.02` and the methods paper description of "divided into 20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a synthetic time axis computed from the bin parameters, not directly from any raw data variable. It represents the time of each bin relative to stimulus onset.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI describes this as "bin centers from interval_beg + binsize to interval_end" matching the reference code.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time axis is computed as `np.linspace(-0.48, 1.5, 100)`, producing 100 evenly-spaced values from -0.48 to 1.5.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
# Result: [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The AI states this matches the reference code's bin center computation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time axis values are intended to represent the centers of the neural data bins, but they are computed differently from the neural bin edges. The neural bins use floor division from -0.5, while the time input uses linspace from -0.48 to 1.5.

ii.
```python
# Neural binning reference point:
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
# interval_begs = stimOn - 0.5, so bins start at -0.5 relative to stim onset

# Time input:
time_since_onset = np.linspace(-0.48, 1.5, 100)
# First bin center should be -0.49 (midpoint of [-0.5, -0.48]), but AI uses -0.48
```

iii. The AI states these are aligned but does not note the discrepancy between the bin edge grid and the time values.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block boundary is detected when `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_numbers = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_val = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_val:
            current_block_start = i
            current_val = prob_left[i]
        trial_numbers[i] = i - current_block_start
    return trial_numbers
```

iii. The trials table has no explicit block identifier, so blocks are recovered from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The block number is computed from the full (unfiltered) trials table, so excluded trials still advance the count. The trial number is the 0-based position within the current block.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. Using the full trials table before filtering ensures the block count reflects the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which takes values -1, 0, or 1.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]  # -1 or 1
```

iii. The IBL convention is choice = 1 for left, choice = -1 for right, choice = 0 for no response.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice values using the formula `((choice_raw + 1) / 2)`. The comment says "-1 (left) -> 0, 1 (right) -> 1", but in the IBL convention -1 is right and 1 is left. The actual mapping produced is: IBL left (+1) -> 1, IBL right (-1) -> 0. This is reversed from the instructions which specify left = 0, right = 1.

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1  [Comment is wrong about IBL convention]
choice = ((choice_raw + 1) / 2).astype(np.int32)  # 0 or 1
# Actual effect: IBL +1 (left) -> 1, IBL -1 (right) -> 0
```

iii. The AI misidentified the IBL choice convention, resulting in a reversed mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. This is the block prior probability directly available in the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three values are mapped to integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, matching the instructions.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
```

iii. This mapping directly follows the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. The position is processed into velocity using interpolation and filtering, then speed is the absolute value.

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. These are the raw wheel data files from the IBL data release.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel processing reproduces the brainbox `SessionLoader.load_wheel()` pipeline: (1) interpolate position to 1000 Hz uniform grid, (2) apply Butterworth low-pass filter (order=8, corner=20Hz) using `sosfiltfilt`, (3) compute velocity as filtered diff * fs, (4) speed = abs(velocity). The speed is then interpolated to trial bin times and discretized into 3 equal-frequency bins.

ii.
```python
fs = 1000
t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

iii. The AI states this matches the brainbox processing exactly.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel speed values (flattened across all trials and time bins in a session) are discretized into 3 bins using the 33.3rd and 66.7th percentiles as thresholds, producing classes 0 (low), 1 (medium), 2 (high).

ii.
```python
def discretize_to_bins(values, n_bins=3):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(valid, quantiles)
    result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
    result = np.clip(result, 0, n_bins - 1)
    return result

wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
```

iii. Equal-frequency binning produces roughly 1/3 of values in each bin, matching the reference approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to bin times using `interp1d` with `np.linspace(t_beg + BINSIZE, t_end, N_BINS)` as the interpolation targets. These targets are -0.48 to 1.5 relative to stimulus onset, which differs slightly from the neural bin centers (-0.49 to 1.49).

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. The AI states this matches the reference code's `get_behavior_per_interval()`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
if me_file is not None and times_file is not None:
    me = np.load(me_file).flatten()
    times = np.load(times_file).flatten()
```

iii. Left camera is tried first, falling back to right, matching the reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated to trial bin times using `interp1d` and then discretized into 3 equal-frequency bins using the same quantile approach as wheel speed.

ii.
```python
binned_me, me_mask = interpolate_behavior(me_times, me_values, interval_begs, interval_ends)
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: flattened across all trials and time bins in the session, then discretized at the 33.3rd and 66.7th percentiles into 3 bins (0=low, 1=medium, 2=high).

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. Same justification as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated to `np.linspace(t_beg + BINSIZE, t_end, N_BINS)`, which is shifted by half a bin from the neural bin centers.

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. Same as wheel speed alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) NaN values in key trial variables cause trial exclusion; (2) missing spike data files cause probe/session to be skipped; (3) behavior interpolation checks for adequate temporal coverage (data within one bin of window edges); (4) sessions with fewer than 2 valid trials are skipped; (5) revision directories are searched for latest versions of files.

ii.
```python
# NaN exclusion
for event in NAN_EXCLUDE:
    mask &= ~trials_df[event].isna()

# Missing spike data
if spikes is not None:
    spikes_list.append(spikes)

# Coverage check in interpolate_behavior
if np.abs(t_beg - trial_times[0]) > BINSIZE:
    continue

# Minimum trials
if n_good_trials < 2:
    return None
```

iii. The AI handles missing data by dropping affected trials or sessions rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. According to timing output, spike binning and behavior loading/interpolation are the most expensive steps. Loading spike data from disk takes 0.3-0.5s per session, and wheel processing (interpolation, filtering) takes 0.4-1.0s per session.

ii.
```python
t_spike = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike_done = time.time()
# ...
t_beh_done = time.time()
print(f"spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s")
```

iii. Total processing time was ~36 minutes for 459 sessions (~4.6s/session average).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes_fast` function loops over trials, and `interpolate_behavior` loops over trials. Both could potentially be vectorized. Additionally, `compute_trial_number_in_block` uses an explicit Python loop that could use pandas groupby/cumcount. The brain region index building also uses a Python loop over neurons.

ii.
```python
# Trial loop in bin_spikes_fast
for i, trial_idx in enumerate(valid_idx):
    s, e = starts[i], ends[i]
    # ...

# Trial loop in interpolate_behavior
for trial_idx in range(n_trials):
    # ...

# Python loop for trial number
for i in range(len(prob_left)):
    if prob_left[i] != current_val:
        # ...

# Python loop for region indexing
for neuron_idx in range(result['n_neurons']):
    region = result['cluster_beryl'][neuron_idx]
    if region not in region_set:
        region_set.append(region)
```

iii. The per-trial loops handle variable-length data slices, making full vectorization non-trivial.

## 10-c. What processing does the code repeat multiple times?

i. The reaction time computation (`firstMovement_times - stimOn_times`) is computed twice in `create_trials_mask` (once for MIN_RT check, once for MAX_RT check). The `find_file` function's directory scanning is repeated for each file type. `BrainRegions()` is instantiated inside `process_session` for every session instead of being shared.

ii.
```python
# RT computed twice
if MIN_RT is not None:
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt <= MAX_RT)

# BrainRegions() per session
br = BrainRegions()  # inside process_session, called 459 times
```

iii. These are minor inefficiencies that don't significantly affect total runtime.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and processes `clusters.depths.npy` and `clusters.metrics.pqt` but never uses them in the final output. The `spike_bin_vectorized` function is defined but superseded by `bin_spikes_fast`. The code also computes `MAX_TRIAL_LEN` filtering on `feedback_times - goCue_times` which is an extra filtering step beyond what the reference human solution applies.

ii.
```python
# Loaded but unused
if os.path.exists(depths_file):
    clusters['depths'] = np.load(depths_file).flatten()
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)

# Unused function
def bin_spikes_vectorized(...):
    # ...
```

iii. These are leftover artifacts from development that don't affect correctness but waste I/O and memory.
