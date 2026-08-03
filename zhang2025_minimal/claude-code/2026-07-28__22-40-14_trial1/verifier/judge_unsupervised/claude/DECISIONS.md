# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the local IBL ONE cache. It reads a BWM release CSV (`bwm_release.csv`) to get session EIDs, cross-references with cached `sessions.pqt` and `datasets.pqt` tables, and iterates over available sessions. For each session, it loads trials from parquet files, spike data from numpy/parquet files per probe, and behavioral data (wheel, whisker motion energy) from numpy files in the ALF directory structure.

ii.
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]

for eid_idx, eid in enumerate(available_eids):
    session_path = find_session_path(eid, sessions_df, datasets_df)
    # ... load trials, spikes, behavior
```

iii. The AI documented this approach in CONVERSION_NOTES.md: "Available sessions: 354 of 459 total sessions available in local cache." The AI used the BWM release CSV as the session list, matching the reference code's `bwm_df = pd.read_csv(freeze_file)`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `sessions_df` table by looking up the `subject` field for each session EID. A `subject_to_idx` dictionary maps subject names to indices. Each session's subject is tracked and stored in `subject_idx`.

ii.
```python
row = sessions_df.loc[eid]
subject = row['subject']
# ...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The AI noted 107 subjects were found across 323 processed sessions, consistent with the BWM dataset.

## 1-c. How are the data split into sessions?

i. Each unique EID from the BWM release CSV corresponds to one session. The AI iterates over all available EIDs and processes each as a separate session. Sessions that fail to load (missing data, too few trials, etc.) are skipped.

ii.
```python
for eid_idx, eid in enumerate(available_eids):
    # Each iteration processes one session
    session_path = find_session_path(eid, sessions_df, datasets_df)
    # ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The AI processed 323 sessions out of 354 available (31 skipped due to missing data).

## 1-d. How are the data split into trials?

i. Within each session, trials are loaded from the `_ibl_trials.table.pqt` parquet file. A boolean mask filters trials based on quality criteria (see 1-e). Each valid trial is then processed individually — neural data is binned per trial aligned to stimulus onset, and behavioral data is interpolated per trial.

ii.
```python
trials = pd.read_parquet(trials_file)
mask = pd.Series(True, index=trials.index)
# ... apply filters to mask
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
```

iii. The AI documented that trials are aligned to `stimOn_times` with a window of (-0.5, 1.5) seconds.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) any of the required fields contain NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (2) reaction time is outside 0.08–2.0 seconds, (3) choice == 0 (no-choice trials), (4) behavioral data (wheel or whisker motion energy) is unavailable for the trial interval.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
MIN_RT = 0.08
MAX_RT = 2.0

for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
mask &= (trials['choice'] != 0)

# Later, additional behavioral data mask:
combined_mask = wheel_mask & me_mask
```

iii. The AI cited the reference code `ibl_data_utils.py:load_trials_and_mask()` defaults. However, the reference code's `prepare_data` also passes `max_trial_len=10.0`, which filters trials where `feedback_times - goCue_times > 10.0`. The AI does not implement this filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times (`spikes.times.npy`), spike cluster assignments (`spikes.clusters.npy`), cluster-to-channel mappings (`clusters.channels.npy`), cluster quality metrics (`clusters.metrics.pqt`), and channel brain location IDs (`channels.brainLocationIds_ccf_2017.npy`).

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. The AI loads raw spike sorting outputs from the pykilosort directory structure, matching the reference code's use of `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (cluster IDs offset and re-indexed, sorted by time). Spikes are then binned into 20ms bins within a (-0.5, 1.5)s window around stimulus onset for each trial. The result is a (n_clusters, n_bins) matrix per trial with spike counts.

ii.
```python
# Merge probes
merged_times = np.concatenate(all_spike_times)
merged_clusters = np.concatenate(all_spike_clusters)
sort_idx = np.argsort(merged_times, kind='stable')

# Bin spikes
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The AI states this matches the reference code's `bin_spiking_data -> get_spike_data_per_interval` pipeline with 20ms bins and 100 time steps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies quality filtering: only clusters with `label >= 1.0` in `clusters.metrics.pqt` are included. This contradicts the reference code, which uses `qc=None` (all clusters).

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
```

iii. The AI noted in CONVERSION_NOTES.md: "The Zhang reference code uses qc=None (all clusters), but we apply quality filtering to match the BWM paper's decoding methodology and keep file sizes manageable." This is a deliberate deviation from the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the alignment time is `stimOn_times[trial]`, and spikes are binned in the window `[stimOn_times - 0.5, stimOn_times + 1.5]`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = valid_trials[ALIGN_TIME].values
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
```

iii. The AI cited reference code `0_data_caching.py` line 53: `'align_time': 'stimOn_times'` and `'time_window': (-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, producing 100 time bins per trial (2.0s / 0.02s = 100). No temporal rebinning is applied — spikes are directly binned at this resolution.

ii.
```python
BINSIZE = 0.02  # 20 ms bins
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The AI cited the methods paper: "divided into 20-ms bins, producing T = 100 time steps" and the reference code `0_data_caching.py` line 52: `'binsize': 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from any raw data variable directly. It is computed analytically as a linear spacing from -0.48s to 1.5s (matching the bin centers of the time window).

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The AI treats this as a deterministic time axis — the same for every trial — representing the center of each 20ms bin relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linspace from `TIME_WINDOW[0] + BINSIZE` (-0.48) to `TIME_WINDOW[1]` (1.5) with `N_TIMEBINS` (100) points. This creates bin centers for each time bin.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
# Result: [-0.48, -0.46, ..., 1.48, 1.5]
```

iii. The AI's computation follows the same bin center convention used in the reference code's `get_behavior_per_interval`: `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset array is the same for all trials and uses the same bin centers as the neural data and behavioral data interpolation. Both use `linspace(t_beg + binsize, t_end, n_bins)`.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
# Stored as first row of input: (2, T)
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)
])
```

iii. The time axis is inherently aligned with the neural bins since both use the same window and bin size.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column in the trials table. Block transitions are detected when `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    prob_left = trials_df['probabilityLeft'].values
    trial_nums = np.zeros(len(trials_df), dtype=np.float64)
    block_start = 0
    for i in range(1, len(prob_left)):
        if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
            block_start = i
        trial_nums[i] = i - block_start
    return trial_nums[mask]
```

iii. The AI computes this as a 0-indexed count from block start, where a new block starts whenever `probabilityLeft` changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number within each block is computed by iterating over all trials (including filtered ones) and detecting block boundaries based on changes in `probabilityLeft`. The counter resets to 0 at each block boundary. The result is then indexed by the trial mask to select only valid trials.

ii.
```python
block_start = 0
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start
return trial_nums[mask]
```

iii. This is stored as a per-trial constant, tiled across all time bins: `np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column in the trials table, which has values -1 (left) and 1 (right).

ii.
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

iii. The AI maps -1 (left) to 0 and 1 (right) to 1, matching the instruction specification: "left = 0, right = 1".

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple remapping: -1 -> 0 (left), 1 -> 1 (right). Stored as a constant across all time bins.

ii.
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1
np.full((1, N_TIMEBINS), choice, dtype=np.int64)
```

iii. No-choice trials (choice == 0) are already excluded by the trial mask.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column in the trials table, which has values 0.2, 0.5, or 0.8.

ii.
```python
prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
```

iii. The AI maps these to categorical values as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Stored as a constant across all time bins.

ii.
```python
prob = prob_left_vals[trial_global_idx]
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
np.full((1, N_TIMEBINS), prior, dtype=np.int64)
```

iii. Matches the instruction specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from raw wheel position (`_ibl_wheel.position.npy`) and timestamps (`_ibl_wheel.timestamps.npy`).

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The AI loads the raw wheel data and processes it using ibllib functions.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The raw wheel position is interpolated to 1kHz uniform sampling using `brainbox.behavior.wheel.interpolate_position`, then velocity is computed using Gaussian-smoothed differentiation via `velocity_filtered`, and speed is computed as absolute velocity.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
```

iii. The AI documented this as matching ibllib's `SessionLoader.load_wheel` and the reference code's `load_target_behavior` for 'wheel-speed'. The reference code uses `SessionLoader.load_wheel()` which internally calls `interpolate_position` and computes velocity, then takes `np.abs(sess_loader.wheel['velocity'].to_numpy())`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency bins using the 33.33rd and 66.67th percentiles computed per session. Values are categorized as 0 (low), 1 (medium), 2 (high).

ii.
```python
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)  # 0, 1, 2
```

iii. The AI uses per-session percentile-based binning for equal frequency distribution.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to match neural bin centers using linear interpolation. The interpolation targets are `linspace(t_beg + binsize, t_end, n_bins)`, the same as the neural bin centers.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The AI's interpolation matches the reference code's `get_behavior_per_interval` function which uses the same linspace formula.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), along with corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The AI noted: "Left camera preferred, right camera fallback" matching the reference code's `bin_behaviors` which tries `'left-whisker-motion-energy'` first, then falls back to `'right-whisker-motion-energy'`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy values are loaded directly from the preprocessed ROI motion energy files. NaN values are removed. The data is then linearly interpolated to match neural bin centers.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(ts_file).flatten()
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]
valid = ~np.isnan(me) & ~np.isnan(times)
return times[valid], me[valid]
```

iii. The reference code uses `SessionLoader.load_motion_energy(views=['left'])` which loads the same underlying data files.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into 3 equal-frequency bins using the 33.33rd and 66.67th percentiles computed per session. Values are categorized as 0 (low), 1 (medium), 2 (high).

ii.
```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)  # 0, 1, 2
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed — linearly interpolated to match neural bin centers using `linspace(t_beg + binsize, t_end, n_bins)`.

ii.
```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
# Inside interpolate_behavior_to_bins:
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. Matches the reference code's approach using `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions with missing probe data or trial data are skipped entirely. (2) Trials with NaN in required fields are excluded via the trial mask. (3) Trials where behavioral data doesn't cover the full interval are excluded via the combined behavioral mask. (4) Behavioral time series with NaN values are excluded. (5) Sessions with fewer than 5 neurons or fewer than 2 valid trials are skipped.

ii.
```python
# Session-level skipping
if session_path is None or not session_path.exists():
    n_skipped += 1; continue
if n_clusters < MIN_NEURONS:
    n_skipped += 1; continue
if n_final < 2:
    n_skipped += 1; continue

# Trial-level behavioral data coverage checks
if np.abs(t_beg - local_times[0]) > binsize:
    good_mask[trial_idx] = False
if np.abs(t_end - local_times[-1]) > binsize:
    good_mask[trial_idx] = False
if np.any(np.isnan(local_vals)):
    good_mask[trial_idx] = False
```

iii. The AI documented 31 sessions skipped and the 3 all-zero neural trials in session 276 as warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Spike binning — iterating over all trials and using `searchsorted` + `np.add.at` for each trial. (2) Behavioral data interpolation — loading and interpolating wheel and whisker data for each trial. (3) Loading spike data from disk — reading large numpy arrays for each probe.

ii.
```python
# Spike binning is O(n_trials * n_spikes_per_trial)
for trial_idx in range(n_trials):
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    bin_indices = np.minimum(((trial_times - t_beg) / binsize).astype(np.int64), n_bins - 1)
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The conversion output shows "Binning spikes (this may take a moment)..." messages, suggesting this is acknowledged as slow.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-trial spike binning loop could potentially be vectorized using `bincount2D` (as the reference code does). The behavioral interpolation loop over trials could also be parallelized (the reference code uses multiprocessing). The trial number in block computation is a sequential loop that could use numpy operations.

ii.
```python
# This loop iterates over every trial individually:
for trial_idx in range(n_trials):
    # ... bin spikes for one trial
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)

# This loop also iterates per trial:
for trial_idx in range(n_trials):
    # ... interpolate behavior for one trial
```

iii. The reference code uses `bincount2D` for spike counting and `multiprocessing.Pool` for parallelism.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly computes the same derived quantities: (1) `n_bins = int(np.ceil(interval_len / binsize))` is computed in multiple functions. (2) Time window boundaries are computed per-trial in both `bin_spikes_per_trial` and `interpolate_behavior_to_bins`. (3) The linspace for bin centers is computed for behavior but also separately as `time_since_stim`.

ii.
```python
# Computed in bin_spikes_per_trial, interpolate_behavior_to_bins, and main:
n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))

# Bin centers computed both in interpolation and as input:
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)  # in interpolate_behavior_to_bins
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS)  # in main
```

iii. These are minor redundancies that don't significantly impact correctness but add slight computational overhead.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code applies quality filtering to neurons (label >= 1.0) which differs from the reference code approach. Additionally, the code loads and processes all probe data from disk even if the session is later skipped for having too few neurons. The code also computes `cluster_brain_ids` for all clusters before filtering, which is partially wasted work. The sanity check section at the end recomputes distributions from the already-stored output data.

ii.
```python
# Quality filtering that reference code doesn't do:
if qc_threshold is not None and clusters_metrics_file is not None:
    metrics = pd.read_parquet(clusters_metrics_file)
    good_mask = metrics['label'].values >= qc_threshold

# Full spike data loaded before checking neuron count:
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
# ... later:
if n_clusters < MIN_NEURONS:
    n_skipped += 1; continue
```

iii. The reference code does not filter by cluster quality and uses all available clusters.
