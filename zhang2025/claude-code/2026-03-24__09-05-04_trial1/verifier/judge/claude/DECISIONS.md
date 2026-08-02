# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the BWM release CSV (`bwm_release.csv`) to get the list of sessions. For each session, it constructs the path to the ONE cache directory (`data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/alf/`) and loads trials, spikes, wheel, and motion energy data from individual files (parquet tables for trials, numpy arrays for spikes/wheel/ME). Revision directories (`#YYYY-MM-DD#`) are searched to find the latest revision of each file.

ii.
```python
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')
DATA_DIR = Path('/app/data/one_cache')

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)  # groupby('eid').first()

for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

```python
def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
    if sess_path.exists():
        return sess_path
    return None
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that the reference code uses BWM release CSV as the session list with `prepare_data()` iterating over eids. The AI chose to load files directly from the ONE cache rather than using the ONE API, since the data was already cached locally. This matches the reference code's approach of loading from the same underlying data files.

## 1-b. How are the data split into subjects (mice)?

i. The AI collects unique subject names from successfully processed sessions and creates a mapping from subject to index. Each session's result includes the subject name, which is used to build `subject_idx` mapping sessions to subjects.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

for r in all_results:
    subject_idx.append(subject_to_idx[r['subject']])

data = {
    'subjects': all_subjects,
    'subject_idx': np.array(subject_idx),
    ...
}
```

iii. The AI identified from the BWM CSV that each session has a `subject` field. Subject splitting follows from the session-level organization of the data. The AI documented finding 129 subjects out of the expected 139, attributing the difference to 67 skipped sessions with missing data.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in the BWM release CSV. The AI groups the CSV by `eid` to get one row per session, then processes each session independently. Each session's data (neural, inputs, outputs) is stored as a separate entry in the output lists.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

iii. The AI documented in CONVERSION_NOTES.md Step 4 that the BWM CSV contains 459 unique sessions, which it uses as the ground truth session list. This matches the reference code's `0_data_caching.py` which also iterates over eids from the BWM CSV.

## 1-d. How are the data split into trials?

i. For each session, trials are loaded from the `_ibl_trials.table.pqt` parquet file. A trial mask is applied to filter out invalid trials, and the remaining valid trials define the per-session trial structure. Each trial contributes one entry in the session's neural, input, and output lists.

ii.
```python
def load_trials(alf_path):
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
    trials = pd.read_parquet(trials_file)
    return trials

# In process_session:
trials = load_trials(alf_path)
mask = create_trial_mask(trials)
valid_trials = trials[mask].reset_index(drop=True)
```

iii. The AI's approach matches the reference code which loads trials via `SessionLoader.load_trials()` and applies the same mask. Both ultimately read from the same `_ibl_trials.table.pqt` files.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a boolean mask that excludes:
- Trials with NaN in key event columns: `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
- Reaction time < 0.08s or > 2.0s (RT = `firstMovement_times - stimOn_times`)
- No-choice trials (`choice == 0`)
- Trials exceeding maximum length of 10.0s (`feedback_times - goCue_times > 10.0`)

Additionally, a combined mask excludes trials where wheel speed or whisker ME data is unavailable (NaN in any time bin).

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0

def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    mask &= (trials['choice'] != 0)
    if 'goCue_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    return mask
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 and 4 that these filtering criteria match the reference code's `load_trials_and_mask()` function exactly, including the default NaN exclusion list, RT bounds, no-choice exclusion, and max trial length of 10.0s.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) for each probe, plus `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
def load_spike_sorting(alf_path, probe_name):
    probe_path = alf_path / probe_name / 'pykilosort'
    spikes = {
        'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
        'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
    }
    clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
    clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
```

iii. The AI identified these as the core spike sorting outputs from Kilosort 2.5 (pykilosort), matching the reference code which loads the same data through `SpikeSortingLoader.load_spike_sorting()`.

## 2-b. How is the `neural` data processed?

i. Spike data from multiple probes is merged with cluster ID offsets to avoid collisions. Spikes are then binned into 20ms time windows aligned to stimulus onset, producing a (n_trials, n_clusters, n_bins) array. The final output is stored as uint8 per trial with shape (n_clusters, n_bins).

ii.
```python
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_starts, interval_ends, binsize, n_bins):
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    for trial_idx in range(n_trials):
        idx_beg = np.searchsorted(spike_times, t_start, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='left')
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
        np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
    return binned

# Later converted to uint8:
neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. The AI documented that spike binning follows the reference code's `get_spike_data_per_interval()` and `bin_spiking_data()` approach. The uint8 conversion was a memory optimization justified by max spike count of 68 fitting within uint8 range (0-255).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to clusters — all clusters are included regardless of their quality label (equivalent to `qc=None` in the reference code).

ii.
```python
# No quality filtering in load_spike_sorting or anywhere else
# All clusters from pykilosort output are used
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

iii. The AI explicitly documented this decision in CONVERSION_NOTES.md Step 4: "Follow reference code: use all units (qc=None)". The reference code's `load_spiking_data()` is called without a `qc` parameter (default `None`), which returns all clusters without quality filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the time window is `[stimOn_times - 0.5, stimOn_times + 1.5]`, giving a 2-second window with 100 bins at 20ms each.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. The AI documented in CONVERSION_NOTES.md Step 4 that alignment to `stimOn_times` follows both the task specification and the reference code's `params['align_time'] = 'stimOn_times'` with `time_window = (-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins for the 2-second window. No temporal rebinning is applied — this is the native bin size used throughout the conversion.

ii.
```python
BINSIZE = 0.02  # 20 ms time bins
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The AI documented this matches the reference code's `params['binsize'] = 0.02` and the methods paper's description of "20-ms bins".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not derived from a raw data variable. It is a constructed time axis based on the time window parameters (TIME_WINDOW and BINSIZE), representing the time relative to stimulus onset for each bin.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# = np.linspace(-0.48, 1.5, 100)
```

iii. The AI constructed this as a uniform time grid matching the bin centers/right-edges of the neural data bins. This is consistent with the reference code's interpolation grid `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced array from -0.48s to 1.5s with 100 points. The start is offset by one bin size (0.02s) from the window start (-0.5s), matching the reference code's convention for bin time points.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The AI chose this time grid to match the reference code's `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)` used for behavior interpolation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time axis is identical for all trials and represents the same time bins as the neural data (both have 100 bins spanning the [-0.5, 1.5] window). The time input is broadcast to all time bins as a time-varying input.

ii.
```python
input_trials = []
for t in range(len(final_trials)):
    inp = np.array([
        time_since_stim,  # time-varying
        np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
    ], dtype=np.float32)  # (2, n_bins)
    input_trials.append(inp)
```

iii. The AI ensured alignment by using the same number of time bins (100) and the same time window for both neural data and inputs.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column in the trials table (`_ibl_trials.table.pqt`). Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def compute_trial_in_block(prob_left):
    trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
    return trial_in_block
```

iii. The AI noted in CONVERSION_NOTES.md Step 5 that trial number in block is "computed from probabilityLeft changes" and "resets at block boundary". The IBL task uses blocks of trials with fixed probability of left stimulus, and the trial number within each block is a useful decoder input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates over all trials in the session (before filtering), maintaining a running count that resets to 1 whenever `probabilityLeft` changes. The count is computed on the full unfiltered trial set, then the appropriate values are selected for the filtered trials via mask indexing.

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
```

iii. The AI justified computing trial-in-block on the full trial set before applying the mask so that the numbering accurately reflects position within the block, even if some trials are subsequently filtered out.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`), where IBL convention uses -1 for left and 1 for right.

ii.
```python
choice = final_trials['choice'].values.copy()
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
```

iii. The AI documented that IBL uses -1=left, 1=right and the task specification requires left=0, right=1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL choice values (-1 for left, 1 for right) are transformed to binary encoding: `(choice + 1) // 2`, mapping -1 to 0 (left) and 1 to 1 (right). The result is broadcast to all time bins as a per-trial constant.

ii.
```python
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
# In output construction:
np.full(N_BINS, choice_encoded[t], dtype=int),  # per-trial, broadcast
```

iii. The AI followed the task specification: "Choice, binary, per-trial, left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

iii. The AI followed the task specification: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical encoding of the three possible probability values into integer classes. The result is broadcast to all time bins as a per-trial constant.

ii.
```python
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
# In output:
np.full(N_BINS, prior_encoded[t], dtype=int),  # per-trial, broadcast
```

iii. The task specification explicitly defines the mapping, and the AI implemented it directly.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (wheel timestamps).

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```

iii. The AI identified these as the raw wheel data files, matching the reference code which loads the same data through `SessionLoader.load_wheel()`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Processing steps:
1. Interpolate wheel position to 1000 Hz uniform sampling
2. Apply 8th-order Butterworth low-pass filter with 20 Hz corner frequency using `sosfiltfilt` (zero-phase filtering)
3. Compute velocity as `diff(filtered_position) * fs` with a zero prepended
4. Take absolute value of velocity to get speed

ii.
```python
def interpolate_wheel(timestamps, position, fs=WHEEL_FS, corner_freq=WHEEL_CORNER_FREQ, order=WHEEL_FILTER_ORDER):
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t)
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    return t, np.abs(vel).astype(np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 6 that this matches ibllib's wheel processing pipeline: "Wheel velocity via Butterworth filter matching ibllib (1kHz interp, 20Hz corner, order 8)". The reference code uses `SessionLoader.load_wheel()` which internally calls `interpolate_position()` and `velocity_filtered()` with the same parameters.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency (quantile) bins. All timepoints across all trials in a session are pooled, quantile boundaries are computed at 33.3% and 66.7%, and values are assigned to bins 0 (low), 1 (medium), or 2 (high).

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(int), boundaries

all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)
```

iii. The AI chose quantile-based discretization to ensure equal class frequencies (33.3% per bin), which is a standard approach for discretizing continuous variables. The discretization is done per-session.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. After computing wheel speed at 1000 Hz, the continuous signal is interpolated to the same trial-aligned time bins as the neural data. For each trial, the speed values are interpolated to 100 uniformly spaced time points within the [-0.5, 1.5] window around stimulus onset.

ii.
```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)

def interpolate_behavior_to_bins(beh_times, beh_values, interval_starts, interval_ends, binsize, n_bins):
    x_interp = np.linspace(t_start + binsize, t_end, n_bins)
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. This matches the reference code's `get_behavior_per_interval()` which uses the same interpolation approach with `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_leftCamera.times.npy` (camera timestamps). Falls back to right camera files if left camera data is unavailable.

ii.
```python
def load_motion_energy(alf_path, side='left'):
    if side == 'left':
        me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    else:
        me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')
```

iii. The AI matches the reference code's approach in `bin_behaviors()` and `load_target_behavior()` which loads left whisker motion energy first, falling back to right if unavailable.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Motion energy values are loaded directly (pre-computed in the IBL pipeline) and truncated to the minimum length of timestamps and values to handle length mismatches. No additional processing is applied to the raw signal before interpolation to trial bins.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

iii. The AI noted that motion energy is pre-computed in the IBL pipeline (whiskerMotionEnergy from camera ROI analysis). The reference code also loads it as-is through `SessionLoader.load_motion_energy()`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical approach to wheel speed: quantile-based discretization into 3 equal-frequency bins per session.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. Same justification as wheel speed discretization — uses equal-frequency bins to ensure balanced class distributions.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical to wheel speed alignment: the continuous motion energy signal is interpolated to the same 100 trial-aligned time bins using linear interpolation with extrapolation.

ii.
```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. Uses the same `interpolate_behavior_to_bins()` function as wheel speed, ensuring consistent alignment with neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data at multiple levels:
- **Missing session paths**: Sessions are skipped if the path doesn't exist in the cache
- **Missing probes**: Individual probes with loading errors are skipped; sessions proceed with remaining probes
- **Missing wheel/ME**: Sessions are skipped if no wheel or motion energy data is available
- **Length mismatches**: Camera timestamp/value length mismatches are handled by truncation to minimum length
- **Behavior data gaps**: Trials where behavioral data starts too late or ends too early (> 1 bin width gap) are excluded via the combined mask
- **NaN handling**: Trials with NaN in any wheel or ME time bin are excluded
- **Few valid trials**: Sessions with fewer than 2 valid trials are skipped

ii.
```python
# Missing session path
if alf_path is None:
    print(f"  Session {eid} ({subject}/{date}): path not found, skipping")
    return None

# Length mismatch
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)

# Behavior data gap check
if np.abs(t_start - t_sel[0]) > binsize:
    continue
if np.abs(t_end - t_sel[-1]) > binsize:
    continue

# Combined mask for NaN in behavioral data
combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. The AI documented in CONVERSION_NOTES.md Step 9 that 67 sessions were skipped due to "missing data/paths/wheel or <2 valid trials". The data gap checks match the reference code's `get_behavior_per_interval()` validation logic.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning (`bin_spikes_vectorized`), which loops over all trials for each session. The AI estimated ~20s per session initially and optimized to ~3s/session, for a total of ~29 minutes for 459 sessions.

ii.
```python
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_starts, interval_ends, binsize, n_bins):
    for trial_idx in range(n_trials):
        idx_beg = np.searchsorted(spike_times, t_start, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='left')
        # ... binning logic
```

iii. The AI documented timing in CONVERSION_NOTES.md Step 7: "Total ~20s/session, full run ~9200s (~2.5 hours)". After optimization, the full conversion completed in 1741s (29 min).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could potentially be vectorized:
1. `bin_spikes_vectorized`: The trial-level loop with searchsorted + add.at could potentially use a fully vectorized approach with histogram2d or sparse matrices
2. `interpolate_behavior_to_bins`: The trial-level loop for interpolating behavioral data could be parallelized or vectorized
3. `compute_trial_in_block`: Simple sequential loop that could use vectorized diff/cumsum operations

ii.
```python
# bin_spikes_vectorized: loops over trials
for trial_idx in range(n_trials):
    ...

# interpolate_behavior_to_bins: loops over trials
for trial_idx in range(n_trials):
    ...

# compute_trial_in_block: sequential loop
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
```

iii. The AI noted that the reference code uses multiprocessing pools for spike binning and behavior interpolation, while the AI chose numpy-based vectorization within each trial instead of multiprocessing.

## 12-c. What processing does the code repeat multiple times?

i. The following processing is repeated:
1. `find_latest_revision()` is called multiple times per session for different files, each time scanning the directory for revision folders
2. Time axis construction (`np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)`) is computed per trial in the input construction loop despite being identical for all trials
3. The `np.full(N_BINS, ...)` calls for per-trial constants are repeated for each trial

ii.
```python
# time_since_stim is computed once, but then used in a loop:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
for t in range(len(final_trials)):
    inp = np.array([
        time_since_stim,  # same for every trial
        np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
    ], dtype=np.float32)
```

iii. The time axis is at least computed once and reused via reference, but the per-trial array construction loop creates many small arrays that could be pre-allocated.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of processing are done but not used in the final output:
1. `clusters.depths.npy` is loaded but never used in the output
2. `clusters.metrics.pqt` is loaded but never used (since qc=None, no quality filtering)
3. `plot_processing()` function exists but is only called with `--show-processing` flag
4. The `wheel_good` and `me_good` masks from behavior interpolation are computed but superseded by the NaN-based combined mask
5. Brain atlas initialization (`BrainRegions()`) includes full atlas data but only acronym mapping is used

ii.
```python
# Loaded but unused:
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()

# Loaded but unused for filtering:
metrics_file = rev_dir / 'clusters.metrics.pqt'
if metrics_file.exists():
    metrics = pd.read_parquet(metrics_file)
```

iii. The AI loaded these files defensively to have them available if needed, but since `qc=None` was chosen (matching reference code), the metrics are not used for filtering. The depths are not included in the output format.
