# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading parquet index files directly from the ONE cache directory rather than using the ONE API. It reads `sessions.pqt` and `datasets.pqt` from the BWM release tables directory, and uses `bwm_release.csv` from the reference code directory to determine which session EIDs to process. For each session, it constructs file paths manually and loads raw `.npy` and `.pqt` files from disk, searching revision subdirectories (e.g. `#2025-03-03#/`) when files are not at the base path.

ii.
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
```

```python
def find_file_with_revision(base_dir, filename):
    direct = base_dir / filename
    if direct.exists():
        return direct
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith('#') and d.name.endswith('#'):
            candidate = d / filename
            if candidate.exists():
                return candidate
    return None
```

iii. The AI attempted to use the ONE API initially but encountered issues with SessionLoader not finding versioned files. It then switched to a direct file-loading approach, reading parquet tables and numpy files manually. The AI noted (trajectory step 48): "The trials table has all the data we need. The issue is that ONE's SessionLoader isn't finding the versioned table. Let me write a direct data loading approach."

## 1-b. How are the data split into subjects?

i. The subject for each session is looked up from the `sessions.pqt` parquet table using the session EID as the index. Subjects are assigned indices in the order they are first encountered during processing.

ii.
```python
row = sessions_df.loc[eid]
subject = row['subject']

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The sessions table provides the subject name directly. No special splitting logic is needed.

## 1-c. How are the data split into sessions?

i. Each EID from the BWM release CSV corresponds to one session. The code iterates over available EIDs, processing one session at a time.

ii.
```python
for eid_idx, eid in enumerate(available_eids):
    session_path = find_session_path(eid, sessions_df, datasets_df)
    ...
```

iii. Sessions are the natural unit of the data; no splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The code reads this parquet file and each row corresponds to one trial.

ii.
```python
trials_file = find_file_with_revision(alf_dir, '_ibl_trials.table.pqt')
trials = pd.read_parquet(trials_file)
```

iii. The trials table is already one row per trial; no splitting is required.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on several criteria: (1) NaN exclusion in required columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`); (2) reaction time between 0.08 s and 2.0 s; (3) no-choice trials (choice == 0) are excluded; (4) behavioral data coverage — trials must have wheel and whisker motion energy data spanning the trial window.

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

Coverage check in `interpolate_behavior_to_bins`:
```python
if np.abs(t_beg - local_times[0]) > binsize:
    good_mask[trial_idx] = False
if np.abs(t_end - local_times[-1]) > binsize:
    good_mask[trial_idx] = False
```

iii. The AI stated it matched the reference code's `load_trials_and_mask` defaults (trajectory step 63). The NaN exclusion list includes `feedback_times` and `feedbackType` which are not directly used by the conversion but were part of the reference code's default NaN checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Additionally, `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` are used for brain region mapping, and `clusters.metrics.pqt` for quality filtering.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s around stimulus onset), giving one count per cluster per bin. The raw spike counts are stored directly as float64 **without** dividing by the bin width. When a session has multiple probes, their clusters are merged by offsetting cluster IDs, and the merged spike train is sorted by time.

ii.
```python
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
session_neural.append(neural_trial.astype(np.float64))
```

iii. The AI followed the reference code's binning approach. However, the reference solution divides by BIN to convert to firing rates (Hz); the AI stores raw counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1.0` in the metrics parquet are kept, matching the BWM paper's "well-isolated neurons" criterion. However, clusters mapped to 'void' brain regions (outside the brain) are NOT explicitly filtered out.

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
            spike_mask, ib = ismember(spike_clusters, good_cluster_ids)
            spike_times = spike_times[spike_mask]
            spike_clusters = ib.astype(np.int32)
```

iii. The AI deliberated extensively on whether to use all clusters (matching Zhang reference code's qc=None) or to filter (matching the BWM data paper). The AI ultimately applied the quality filter (`label >= 1.0`) for practical reasons — it greatly reduced file size and aligned with the data paper's methodology. The top-of-file comment incorrectly states "All spike-sorted clusters included (no quality filter)" which contradicts the actual code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are selected from the merged spike train by finding the time window [stimOn_time - 0.5, stimOn_time + 1.5] using searchsorted, then bin indices are computed relative to the window start.

ii.
```python
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
trial_times = spike_times[i_start:i_end]
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
```

iii. All data streams share the same session clock, so aligning is simply subtracting the stimulus onset time and windowing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2-second window. No rebinning or smoothing is applied.

ii.
```python
BINSIZE = 0.02  # 20 ms bins
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. This matches the reference code's `binsize=0.02` and both papers' use of 20 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial window parameters and bin size — it is a computed time grid, not from any raw data variable. The alignment event is `stimOn_times`.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The time grid is defined by the processing parameters, identical for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The input is computed as `np.linspace(-0.48, 1.5, 100)`, which produces 100 evenly spaced values. This follows the Zhang reference code's `np.linspace(t_beg + binsize, t_end, n_bins)` convention.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. No additional processing is needed; the variable is defined by construction.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values are the bin centers of the neural binning grid, so they are inherently aligned. However, the AI uses the Zhang code's `linspace(t_beg + binsize, t_end, n_bins)` convention which starts at -0.48 s and ends at 1.50 s, rather than the standard bin center convention (-0.49 to 1.49). The bin spacing is the same (0.02 s) but the values are shifted by half a bin (0.01 s).

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
)
```

iii. The AI followed the Zhang reference code's linspace formulation for bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` between consecutive trials marks the start of a new block.

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

iii. The trials table has no explicit block identifier, so blocks must be inferred from where probabilityLeft changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code iterates over all trials and tracks when `probabilityLeft` changes (or is NaN), resetting the block counter. The trial number is the distance from the block start. This is computed on the full trial table before filtering, so filtered-out trials still advance the count, preserving the animal's true position in the block.

ii.
```python
block_start = 0
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start
return trial_nums[mask]
```

iii. The trial number is computed before the mask is applied, which is important for preserving real block positions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which takes values +1 (left in IBL convention), -1 (right), and 0 (no response).

ii.
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
...
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

iii. The AI mapped choice values to binary 0/1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps IBL choice values to binary: -1 → 0, +1 → 1. However, in the IBL convention +1 is a **leftward** choice and -1 is **rightward**. The instructions specify left=0, right=1. The AI's mapping therefore **inverts** the labels: it assigns left(+1)→1 and right(-1)→0, the opposite of what is required.

ii.
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

The AI's comment says `-1->0 (left)` but -1 is rightward in IBL, not left.

iii. The AI misidentified the IBL convention. The comment says "-1 is left" but IBL uses -1 for right and +1 for left. The reference solution correctly maps: `CHOICE = {1.0: 0, -1.0: 1}  # +1 is a leftward choice, -1 rightward`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

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

iii. The three values are the block prior probabilities, and the instructions give the mapping 0.2→0, 0.5→1, 0.8→2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A simple recoding of 0.2→0, 0.5→1, 0.8→2. No additional processing.

ii.
```python
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:
    prior = 2
```

iii. Matches the instructions directly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` — the wheel position and its timestamps.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. These are the standard wheel data files from the IBL pipeline.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) The wheel position is interpolated to a uniform 1 kHz grid and velocity is computed using `velocity_filtered` (which applies a Butterworth low-pass filter), matching the ibllib `SessionLoader.load_wheel` behavior. (2) Speed is the absolute value of velocity. (3) The speed trace is interpolated to the trial bin centers using `scipy.interpolate.interp1d`. Finally, the trace is discretized into 3 equal-frequency bins based on session-level percentiles (33.33rd and 66.67th).

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

iii. The AI explicitly imported and used the same ibllib functions as SessionLoader for wheel processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous speed values are discretized into 3 bins using the 33.33rd and 66.67th percentiles of all valid speed values across the session, producing roughly equal-frequency bins labeled 0, 1, 2.

ii.
```python
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. This matches the reference's approach of equal-sized classes at session-level percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated to the same time grid as the neural bins using `interp1d`. The interpolation targets are `np.linspace(t_beg + binsize, t_end, n_bins)`, the same linspace-based bin centers as the neural data.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. The wheel is on the same session clock as the spikes, so interpolating to matching bin centers aligns the data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from a side camera: `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding timestamps `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The AI follows the reference's preference for the left camera with right camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no additional filtering or normalization). NaN values are removed. The trace is interpolated to trial bin centers using `interp1d`, then discretized into 3 equal-frequency bins at session-level percentiles (same as wheel speed).

ii.
```python
me = np.load(me_file).flatten()
times = np.load(ts_file).flatten()
valid = ~np.isnan(me) & ~np.isnan(times)
times, me = times[valid], me[valid]
```

```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. No additional processing beyond NaN removal and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: discretized into 3 bins at the 33.33rd and 66.67th percentiles of all valid whisker motion energy values across the session.

ii.
```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. This produces roughly equal-frequency bins labeled 0, 1, 2.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker motion energy trace is interpolated to the same bin centers as the neural data using `interp1d`, the same approach used for wheel speed.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. Camera frame times are on the same session clock, so interpolation to matching bin centers aligns the data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers of handling: (1) Trials with NaN in any of six key columns are excluded. (2) Sessions with fewer than 2 valid trials or fewer than 5 neurons are skipped. (3) Probes with missing spike sorting files are skipped. (4) Behavioral data coverage is checked — trials where wheel or whisker data doesn't span the window are excluded. (5) NaN values in whisker motion energy are removed before interpolation.

ii.
```python
if n_valid_trials < 2:
    n_skipped += 1; continue

if n_clusters < MIN_NEURONS:  # MIN_NEURONS = 5
    n_skipped += 1; continue

combined_mask = wheel_mask & me_mask
```

iii. The AI's approach is generally conservative — missing data leads to exclusion rather than imputation.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (large numpy arrays) and binning spikes per trial. The AI noted the binning was slow and optimized it with `np.add.at` for vectorized counting within each trial.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
```

```python
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The AI explicitly optimized the spike binning during development (trajectory step 94).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: (1) `bin_spikes_per_trial` loops over trials to bin spikes, and (2) `interpolate_behavior_to_bins` loops over trials to interpolate behavioral data. Both could potentially be vectorized by offsetting indices across trials, but the current approach is clear and functional.

ii.
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(local_times, local_vals, kind='linear', ...)(x_interp)
```

iii. The per-trial loops are the natural structure; vectorization would add complexity for modest gains.

## 10-c. What processing does the code repeat multiple times?

i. The behavioral interpolation function `interpolate_behavior_to_bins` is called separately for wheel and whisker data, performing the same interpolation and coverage-checking logic on two different data streams. This is a reasonable code reuse pattern rather than true repeated processing.

ii.
```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE)
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE)
```

iii. The repeated pattern is by design — both behavioral streams need the same processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `feedback_times` and `feedbackType` as part of the NaN exclusion check, even though these variables are not used in the decoder. The code also loads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for every probe, including the mapping computation, even for clusters that will be filtered out by quality control. Additionally, the code stores neural data as float64 instead of float32, doubling memory usage unnecessarily.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
```

```python
session_neural.append(neural_trial.astype(np.float64))
```

iii. The broader NaN exclusion follows the Zhang reference code defaults, which check more fields than strictly necessary for this decoder task.
