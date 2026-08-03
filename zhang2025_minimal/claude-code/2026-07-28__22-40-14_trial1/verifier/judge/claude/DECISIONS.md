# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads parquet index files (`sessions.pqt`, `datasets.pqt`) directly from the ONE cache directory and cross-references them with a BWM release CSV (`bwm_release.csv`) from the reference code directory. Session paths are resolved manually by constructing paths from lab/subject/date/number. Individual data files (spikes, trials, wheel, camera) are loaded by directly opening `.npy` and `.pqt` files from the filesystem, with logic to search revision subdirectories (e.g., `#2025-03-03#/`).

ii.
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
```

iii. The AI chose to bypass the ONE API and load files directly from disk, constructing file paths manually. The CONVERSION_NOTES.md does not explain why the ONE API was not used. The BWM CSV is used to identify which sessions belong to the release.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `sessions.pqt` table by looking up the `subject` column for each session eid. A dictionary `subject_to_idx` maps subject names to indices, assigned in the order sessions are encountered (not sorted).

ii.
```python
row = sessions_df.loc[eid]
subject = row['subject']

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The subject is read from the session metadata. The ordering is encounter-order rather than sorted.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their unique `eid` from the BWM CSV cross-referenced with the local cache. Each eid corresponds to one session and is processed independently.

ii.
```python
available_eids = [eid for eid in bwm_eids if eid in cache_eids]

for eid_idx, eid in enumerate(available_eids):
    session_path = find_session_path(eid, sessions_df, datasets_df)
```

iii. No additional splitting is needed; sessions are already the unit of organization.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` file, which has one row per trial. No additional splitting is needed.

ii.
```python
trials_file = find_file_with_revision(alf_dir, '_ibl_trials.table.pqt')
trials = pd.read_parquet(trials_file)
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in multiple steps: (1) exclude trials with NaN in required fields (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType); (2) exclude trials with reaction time outside 0.08-2.0 s; (3) exclude no-choice trials (choice == 0); (4) exclude trials where wheel or whisker data doesn't fully cover the trial window.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

And later, behavioral coverage filtering:
```python
combined_mask = wheel_mask & me_mask
```

iii. The CONVERSION_NOTES.md cites the reference code `load_trials_and_mask()` defaults for the NaN exclusion list and reaction time bounds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times (`spikes.times.npy`) and spike cluster assignments (`spikes.clusters.npy`) from each probe's pykilosort directory. Cluster channel assignments (`clusters.channels.npy`) and brain region IDs (`channels.brainLocationIds_ccf_2017.npy`) are used for region mapping.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. The raw spike data is loaded directly from the pykilosort output files.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s around stimulus onset). The counts are stored directly as float32 WITHOUT conversion to firing rate (no division by bin width). Multi-probe sessions have their clusters merged with offset cluster indices, sorted by time.

ii.
```python
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
neural_trial = binned_spikes[trial_global_idx]
session_neural.append(neural_trial.astype(np.float64))
```

iii. The CONVERSION_NOTES.md does not mention whether the neural data is converted to firing rates. The code stores raw spike counts, not rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with quality label >= 1.0 are kept, matching the BWM paper's well-isolated neuron criterion. The filtering uses `clusters.metrics.pqt` to read the label column. Additionally, sessions with fewer than 5 neurons are skipped (MIN_NEURONS = 5).

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    metrics = pd.read_parquet(clusters_metrics_file)
    if 'label' in metrics.columns:
        good_mask = metrics['label'].values >= qc_threshold
        good_cluster_ids = np.where(good_mask)[0]
```

```python
MIN_NEURONS = 5
if n_clusters < MIN_NEURONS:
    print(f"  SKIP: too few clusters ({n_clusters} < {MIN_NEURONS})")
    n_skipped += 1
    continue
```

iii. The CONVERSION_NOTES.md says the quality filter matches the BWM paper methodology, but notes it differs from the Zhang reference code which uses `qc=None` (all clusters). However, the file header comment contradicts this by saying "All spike-sorted clusters included (no quality filter, matching reference code's qc=None)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are selected within the window [stimOn_times + T_START, stimOn_times + T_STOP] and bin indices are computed relative to the trial start time.

ii.
```python
align_times = valid_trials[ALIGN_TIME].values

t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]

i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')

trial_times = spike_times[i_start:i_end]
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
```

iii. The alignment is to stimulus onset as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial covering a 2 s window. No rebinning is applied; spikes are directly counted into these bins.

ii.
```python
BINSIZE = 0.02  # 20 ms bins
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The CONVERSION_NOTES.md cites the reference code and methods paper for 20 ms bins and 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial window parameters (T_START, T_STOP, BINSIZE) and computed as evenly spaced bin centers. It does not directly come from a raw data variable.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The CONVERSION_NOTES.md states the values are "linearly spaced from -0.48s to 1.5s".

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as `np.linspace(-0.48, 1.5, 100)`, which produces bin centers starting at -0.48 and ending at 1.5.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
# = np.linspace(-0.48, 1.5, 100)
```

iii. No justification given for this specific formula. The reference code uses `EDGES[:-1] + BIN/2` which gives bin centers from -0.49 to 1.49.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values are intended to represent bin centers matching the neural bins. However, the formula used (`np.linspace(-0.48, 1.5, 100)`) produces different values than the neural bin centers computed from `(spike_time - t_beg) / binsize`. The neural bins are indexed as `(spike_time - t_beg) / binsize` where `t_beg = align_time - 0.5`, so bin 0 covers [-0.5, -0.48) and the center would be at -0.49 in the reference, but the AI's time input starts at -0.48.

ii.
```python
# Neural binning uses:
bin_indices = ((trial_times - t_beg) / binsize).astype(np.int64)

# Time input uses:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS)
```

iii. The AI's CONVERSION_NOTES.md cites the reference code `get_behavior_per_interval()` as the source for this formula.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in probabilityLeft (or a NaN value) marks the start of a new block.

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

iii. The block boundaries are inferred from changes in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Trial number is computed as the 0-indexed position within each block, calculated before trial filtering so that dropped trials still advance the count. NaN values in probabilityLeft are treated as block boundaries.

ii.
```python
block_start = 0
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start
return trial_nums[mask]
```

iii. The computation is done on all trials before the mask is applied, preserving the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

iii. The AI's comment says "-1->0 (left), 1->1 (right)" but in IBL convention, -1 is rightward choice, not leftward.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice is remapped from IBL convention to binary values. The AI maps -1 to 0 and +1 to 1. However, since IBL convention is +1 = left and -1 = right, this produces right=0, left=1, which is the INVERSE of the instructions (left=0, right=1).

ii.
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

iii. The AI's comment misidentifies the IBL convention. The instructions specify left=0, right=1. The reference maps {1.0: 0, -1.0: 1} which correctly gives left=0, right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob = prob_left_vals[trial_global_idx]
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
```

iii. The mapping follows the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical remapping of 0.2, 0.5, 0.8 to 0, 1, 2. No additional processing.

ii.
```python
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
```

iii. Straightforward mapping as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From raw wheel position (`_ibl_wheel.position.npy`) and timestamps (`_ibl_wheel.timestamps.npy`).

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The wheel data is loaded directly from the ALF directory.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to uniform 1 kHz sampling using `interpolate_position`, then velocity is computed with Butterworth filtering via `velocity_filtered`. Speed is the absolute value of velocity. The speed trace is then linearly interpolated to bin centers and discretized into 3 equal-frequency bins using session-level percentiles.

ii.
```python
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
```

```python
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. The processing uses the same ibllib functions as the reference.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using the 33.33rd and 66.67th percentiles computed across all valid trials in the session.

ii.
```python
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. Equal-frequency binning matching the reference approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to bin centers using `np.linspace(t_beg + binsize, t_end, n_bins)`, which differs from the neural bin centers. The neural bins are indexed from `t_beg` while the behavior bins start at `t_beg + binsize`.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The CONVERSION_NOTES.md cites `get_behavior_per_interval()` from the reference code.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From the camera motion energy file (`leftCamera.ROIMotionEnergy.npy` preferred, `rightCamera.ROIMotionEnergy.npy` as fallback) and corresponding timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')

if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. Left camera preferred, right camera as fallback, matching the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is loaded, NaN values are removed, the trace is linearly interpolated to bin centers, and then discretized into 3 equal-frequency bins using session-level percentiles.

ii.
```python
valid = ~np.isnan(me) & ~np.isnan(times)
me = me[:min_len]
times = times[:min_len]
```

```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Discretized into 3 bins using the 33.33rd and 66.67th percentiles computed across all valid trials in the session.

ii.
```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. Same equal-frequency approach as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to `np.linspace(t_beg + binsize, t_end, n_bins)`, which does not exactly match the neural bin centers.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. Same method as wheel speed alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Sessions without a valid path are skipped; (2) Sessions without probe directories are skipped; (3) Probes without required files are skipped; (4) NaN values in trials are excluded; (5) Behavioral data with NaN or insufficient coverage is excluded per-trial; (6) Sessions with fewer than 2 valid trials or fewer than 5 neurons are skipped; (7) Whisker ME NaN values are removed before interpolation.

ii.
```python
if session_path is None or not session_path.exists():
    n_skipped += 1; continue

if n_valid_trials < 2:
    n_skipped += 1; continue

if n_clusters < MIN_NEURONS:
    n_skipped += 1; continue

combined_mask = wheel_mask & me_mask
if n_final < 2:
    n_skipped += 1; continue
```

iii. The approach is generally conservative, skipping sessions or trials with any missing data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk (reading large `.npy` files for spike times and clusters) and binning spikes into trials.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
```

```python
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters,
    align_times, TIME_WINDOW, BINSIZE
)
```

iii. The spike sorting files are the largest data files per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) The spike binning loop iterates per trial (`for trial_idx in range(n_trials)`) and uses `np.add.at` for counting. This could be vectorized by offsetting spike indices across trials. (2) The behavioral interpolation loop iterates per trial to interpolate wheel and whisker data. (3) The output assembly loop iterates per trial to construct input/output arrays.

ii.
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')(x_interp)
```

iii. The per-trial loops are straightforward but could potentially be vectorized for better performance.

## 10-c. What processing does the code repeat multiple times?

i. The behavioral data interpolation is computed separately for wheel and whisker using the same `interpolate_behavior_to_bins` function with separate calls and separate coverage checks. The coverage checking logic is duplicated between the trial mask (`load_trials`) and the behavioral interpolation (`interpolate_behavior_to_bins`).

ii.
```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
)
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
```

iii. The pattern is repeated but with different data, which is a reasonable implementation choice.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and processes all NaN-exclusion fields (feedback_times, feedbackType) that are not used in downstream analysis - they are only used for filtering. The code also computes trial filtering statistics that are printed but not stored.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
```

iii. The extra NaN exclusion fields add filtering that goes beyond what the reference does, potentially removing valid trials.
