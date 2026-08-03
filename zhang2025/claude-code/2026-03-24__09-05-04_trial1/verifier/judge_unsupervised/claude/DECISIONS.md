# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from the BWM release CSV (`bwm_release.csv`), which lists 459 unique sessions across 699 probe insertions. For each session, it constructs the file path into the ONE cache directory (`/app/data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`) and loads trials (parquet), spike sorting (numpy), wheel (numpy), and motion energy (numpy) files directly from disk. It does not use the ONE API or SpikeSortingLoader classes from ibllib.

ii.
```python
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')
DATA_DIR = Path('/app/data/one_cache')

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)  # groups by eid, one row per session

def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
    if sess_path.exists():
        return sess_path
    return None
```

iii. The AI noted that the BWM release CSV is the ground truth for session selection, matching the reference code's `0_data_caching.py` which also uses this CSV. Direct file loading was chosen because the ONE API was not available/needed given the local cache structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the BWM release CSV. Each session row has a subject field. After processing, unique subjects are collected and sorted alphabetically, and each session is assigned a `subject_idx` into this sorted list.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx.append(subject_to_idx[r['subject']])
```

iii. The AI followed the target data format specification which requires a `subjects` list and `subject_idx` array mapping sessions to subjects.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in the BWM release CSV. The CSV is grouped by `eid` to get one row per session (459 total). Each session is processed independently, yielding one entry in the `neural`, `input`, and `output` lists.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

iii. The BWM CSV has one row per probe insertion (699 rows). Grouping by `eid` yields 459 unique sessions, matching the data paper's count.

## 1-d. How are the data split into trials?

i. For each session, trials are loaded from `_ibl_trials.table.pqt` (parquet format). Each row in the trials table is one trial. After filtering (see 1-e), each valid trial becomes one entry in the session's trial list.

ii.
```python
def load_trials(alf_path):
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
    trials = pd.read_parquet(trials_file)
    return trials
```

iii. This matches the reference code which also loads trials from the same parquet files via SessionLoader.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) a trial mask excluding trials with NaN in key events, bad reaction times, no-choice trials, and excessively long trials; (2) a combined mask additionally requiring valid wheel speed and whisker motion energy data for each trial.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)   # 0.08 <= RT <= 2.0
    mask &= (trials['choice'] != 0)
    if 'goCue_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()  # <= 10s
    return mask

# Second stage: require valid wheel and ME data
combined_mask = np.ones(len(valid_trials), dtype=bool)
if wheel_speed_binned is not None:
    combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
else:
    combined_mask[:] = False
if me_binned is not None:
    combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
else:
    combined_mask[:] = False
```

iii. The first stage exactly matches the reference code's `load_trials_and_mask()` with default parameters (min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True, default nan_exclude list). The second stage mirrors the reference code's `align_spike_behavior()` which removes trials without valid behavioral data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from each probe's pykilosort directory, plus `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
```

iii. Same source variables as the reference code's `SpikeSortingLoader.load_spike_sorting()`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (cluster IDs offset, time-sorted). Spikes are then binned into trial-aligned 20ms windows over a 2-second window (-0.5s to 1.5s relative to stimulus onset), producing shape (n_clusters, 100) per trial. The spike counts are stored as uint8.

ii.
```python
BINSIZE = 0.02  # 20 ms
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_starts, interval_ends, binsize, n_bins):
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    for trial_idx in range(n_trials):
        idx_beg = np.searchsorted(spike_times, t_start, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='left')
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
        np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
    return binned
```

iii. The binning approach matches the reference code's `get_spike_data_per_interval()` which also produces (n_clusters, n_bins) per trial with 20ms bins. The AI used searchsorted + flat indexing for efficiency rather than the reference's `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron quality filtering is applied. All clusters from pykilosort are included regardless of their quality label (equivalent to `qc=None` in the reference code).

ii.
```python
# No filtering code - all clusters loaded and used
spikes, clusters = merge_probes(spikes_list, clusters_list)
n_clusters = len(clusters['channels'])
```

iii. The AI explicitly documented this decision: "Use all clusters (qc=None): Matches reference code, not just well-isolated neurons." The reference code's `load_spiking_data()` is called without the `qc` parameter in `prepare_data()`, defaulting to `qc=None` which returns all clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the binning window starts at `stimOn_times - 0.5` and ends at `stimOn_times + 1.5`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. Matches the reference code's `0_data_caching.py` parameters: `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`, and the task specification which says "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial. No rebinning is applied; spikes are directly binned at 20ms resolution.

ii.
```python
BINSIZE = 0.02  # 20 ms time bins
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize=0.02` and the methods paper's "20-ms bins."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from any raw data variable. It is synthetically constructed from the time window parameters and bin size, since it is the same for every trial by definition.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The AI recognized that this input is deterministic given the alignment to stimulus onset. The formula `linspace(-0.48, 1.5, 100)` matches the reference code's interpolation grid: `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced array from -0.48s to 1.5s with 100 points is created. The start is offset by one bin size (0.02s) from the window start (-0.5s), matching the reference code's behavior interpolation grid convention.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# Result: array from -0.48 to 1.5, 100 evenly spaced values
```

iii. This follows the same convention as the reference code's `get_behavior_per_interval()` interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time array has the same number of bins (100) as the neural data, and both cover the same time window (-0.5 to 1.5s). The time array is broadcast identically across all trials as the first row of the input matrix.

ii.
```python
inp = np.array([
    time_since_stim,  # time-varying
    np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
], dtype=np.float32)  # (2, n_bins)
```

iii. Alignment is implicit: both neural and input have 100 time bins spanning the same window.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

iii. The AI noted: "Block boundaries are where probabilityLeft changes." This is a reasonable approach since probabilityLeft directly encodes the block identity (0.2, 0.5, or 0.8).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Iterates through all trials in order. A counter starts at 1 and increments each trial. When `probabilityLeft` changes, the counter resets to 1. The computation is done on ALL trials first, then the appropriate values are selected for filtered/valid trials. The result is broadcast to all 100 time bins as a per-trial constant.

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

# Applied to all trials, then masked
trial_in_block_all = compute_trial_in_block(prob_left_all)
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
```

iii. Computing on all trials before filtering ensures accurate block boundary detection and trial counting.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column in `_ibl_trials.table.pqt`.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. Same source as the reference code which reads `trials_df['choice'].to_numpy()`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL encodes choice as -1 (left) and 1 (right). The AI converts to 0 (left) and 1 (right) using `((choice + 1) // 2)`. No-choice trials (choice == 0) are already excluded by the trial mask. The result is broadcast to all 100 time bins as a per-trial constant.

ii.
```python
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1

# Broadcast to time bins in output
np.full(N_BINS, choice_encoded[t], dtype=int),  # per-trial, broadcast
```

iii. Matches the task specification: "Choice, binary, per-trial, left = 0, right = 1."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column in `_ibl_trials.table.pqt`.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
```

iii. Same source as the reference code which reads `trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The result is broadcast to all 100 time bins as a per-trial constant.

ii.
```python
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2

# Broadcast in output
np.full(N_BINS, prior_encoded[t], dtype=int),
```

iii. Matches the task specification: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2."

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (timestamps).

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

iii. Same source as the reference code which loads wheel data via `SessionLoader.load_wheel()`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1000 Hz uniform sampling, then filtered with a Butterworth low-pass filter (order 8, corner frequency 20 Hz), velocity computed as `np.diff(filtered) * fs`, and absolute value taken for speed.

ii.
```python
def interpolate_wheel(timestamps, position, fs=1000, corner_freq=20, order=8):
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t)
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    return t, np.abs(vel).astype(np.float32)
```

iii. This exactly replicates the reference code's `velocity_filtered()` from ibllib: same interpolation frequency (1000 Hz), same Butterworth filter parameters (order=8, corner=20 Hz), same velocity computation, same absolute value for speed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency (quantile) bins per session. Quantile boundaries at 0%, 33.3%, 66.7%, 100% are computed across all valid timepoints in the session, then `np.digitize` assigns each value to a bin (0=low, 1=medium, 2=high).

ii.
```python
def discretize_to_bins(values, n_bins=3):
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

iii. The task spec says "Wheel speed discretized into 3 bins" without specifying the method. The AI chose equal-frequency quantile bins computed per session. This ensures balanced classes within each session but means bin boundaries differ across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Continuous wheel speed is interpolated to trial-aligned bins using the same time grid as neural data: `linspace(t_start + binsize, t_end, n_bins)` where t_start/t_end are defined by stimOn_times +/- the time window.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, interval_starts, interval_ends, binsize, n_bins):
    x_interp = np.linspace(t_start + binsize, t_end, n_bins)
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
    result[trial_idx] = y_interp
```

iii. Matches the reference code's `get_behavior_per_interval()` which uses the identical interpolation grid and method (linear interpolation with extrapolation).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_leftCamera.times.npy` (timestamps). Falls back to right camera if left is unavailable.

ii.
```python
def load_motion_energy(alf_path, side='left'):
    if side == 'left':
        me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    else:
        me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')

me_times, me_values = load_motion_energy(alf_path, side='left')
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
```

iii. Matches the reference code's `bin_behaviors()` which tries left whisker ME first and falls back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Motion energy values are loaded directly from the numpy files with no additional processing (no filtering, no normalization). Length mismatch between timestamps and values is handled by truncating to the shorter length.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

iii. Matches the reference code which loads whisker ME directly via `sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy']` without additional processing.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: discretized into 3 equal-frequency quantile bins per session.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. Same rationale as wheel speed discretization (7-c).

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation method as wheel speed: continuous motion energy interpolated to the trial-aligned bin grid using linear interpolation with extrapolation.

ii.
```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. Uses the same `interpolate_behavior_to_bins()` function as wheel speed, matching the reference code's approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of missing data handling:
- Sessions with missing file paths are skipped entirely.
- Probes that fail to load are skipped (session continues if at least one probe succeeds).
- Trials with NaN in key event columns are excluded by the trial mask.
- Trials without valid wheel or motion energy data are excluded by the combined mask.
- Sessions with fewer than 2 valid trials after all filtering are skipped.
- Camera timestamp/value length mismatches are handled by truncating to the shorter length.
- Revision directories are searched for the latest revision of each file.

ii.
```python
# Session path check
if alf_path is None:
    print(f"  Session {eid}: path not found, skipping")
    return None

# Probe error handling
try:
    spk, clu = load_spike_sorting(alf_path, probe_name)
except Exception as e:
    print(f"  Session {eid}, {probe_name}: error loading spikes: {e}")
    continue

# Minimum trial check
if n_valid < 2:
    print(f"  Session {eid}: fewer than 2 trials with all data, skipping")
    return None

# Camera length mismatch
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]
```

iii. The AI documented that 67/459 sessions were skipped due to missing data. The approach is conservative: skip rather than impute.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning (`bin_spikes_vectorized`), which iterates over every trial and uses searchsorted + flat indexing to bin spikes. The wheel interpolation (1000 Hz resampling + Butterworth filtering) is also significant per session. Total runtime was ~29 minutes for 459 sessions (~4.4s per session average).

ii.
```python
# Spike binning - loops over all trials
for trial_idx in range(n_trials):
    idx_beg = np.searchsorted(spike_times, t_start, side='left')
    idx_end = np.searchsorted(spike_times, t_end, side='left')
    ...
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

iii. The AI estimated ~20s per session during sample testing, but full run averaged less due to variation in session sizes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The trial loop in `bin_spikes_vectorized` - each trial is processed independently in a Python loop. Could potentially use vectorized operations across trials.
2. The trial loop in `interpolate_behavior_to_bins` - per-trial interpolation in a Python loop.
3. The `compute_trial_in_block` function uses a sequential Python loop to detect block boundaries.
4. The final data structure construction loops (building `neural_trials`, `input_trials`, `output_trials` lists).

ii.
```python
# Trial loop in bin_spikes_vectorized (lines 232-256)
for trial_idx in range(n_trials):
    ...

# Trial loop in interpolate_behavior_to_bins (lines 301-332)
for trial_idx in range(n_trials):
    ...

# Block detection loop (lines 357-363)
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
    ...
```

iii. The reference code also uses per-trial loops for spike binning (with multiprocessing via `n_workers`), so this is consistent. The block detection loop is short enough that vectorization would provide negligible speedup.

## 10-c. What processing does the code repeat multiple times?

i. The behavior interpolation function `interpolate_behavior_to_bins` is called twice per session (once for wheel speed, once for whisker ME) with the same trial intervals. The interval computation (searchsorted for finding behavior data within trial windows) is duplicated. The `find_latest_revision` function is called repeatedly for different files in the same directory.

ii.
```python
# Called twice with same interval_starts/interval_ends
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS)
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS)
```

iii. This duplication has minimal performance impact since the interpolation is fast compared to spike binning.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that is not used in the final output:
1. `clusters.depths.npy` is loaded but never used in the final data structure.
2. `clusters.metrics.pqt` (quality metrics) is loaded but never used since `qc=None`.
3. The merged spikes are time-sorted (`np.argsort` + reindexing), which is necessary for correctness but the original probes' spikes are already time-sorted individually, so the merge-sort could be done more efficiently.
4. Wheel and whisker ME continuous values are computed but discarded after discretization. However, this is necessary to perform the discretization.

ii.
```python
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()  # loaded but unused
metrics = pd.read_parquet(metrics_file)  # loaded but unused (qc=None)
sort_idx = np.argsort(all_times, kind='stable')  # full sort of merged spikes
```

iii. The unused data loading adds modest overhead. The depths and metrics loading is inherited from the reference code's data structure but not needed for this specific conversion.
