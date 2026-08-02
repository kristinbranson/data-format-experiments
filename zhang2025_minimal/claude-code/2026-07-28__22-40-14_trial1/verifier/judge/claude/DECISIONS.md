# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from parquet files (`sessions.pqt`, `datasets.pqt`) and identifies sessions from the BWM release CSV (`bwm_release.csv`). It filters to sessions available in the local cache, then iterates over each session eid. For each session, it loads trials from `_ibl_trials.table.pqt`, spike data from probe directories under `alf/probeXX/pykilosort/`, and behavioral data (wheel, whisker ME) from the `alf/` directory. All files are loaded using numpy (`np.load`) and pandas (`pd.read_parquet`).

ii.
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
# ...
bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
```

iii. The AI documented that it uses the BWM release CSV and local cache, processing 354 of 459 total sessions available. This approach matches the reference code's use of `bwm_df` to identify sessions, but replaces the ONE API with direct file loading from the local cache.

## 1-b. How are the data split into subjects (mice)?

i. The AI extracts the subject name from the `sessions_df` table for each session eid. Subjects are tracked via a `subject_to_idx` dictionary, building the `subjects` list and `subject_idx` array incrementally as new subjects are encountered.

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

iii. The AI noted 107 subjects in the full dataset. The reference data paper reports 139 mice total, but not all sessions are available in the local cache.

## 1-c. How are the data split into sessions?

i. Each unique eid from the BWM release represents one session. The AI iterates over all available eids, processing one session at a time. Each session's data (neural, input, output) is stored as a separate list entry.

ii.
```python
for eid_idx, eid in enumerate(available_eids):
    # ... process session ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The AI processes 323 sessions (31 skipped due to missing data out of 354 available). The reference code similarly processes sessions by eid.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file for each session. A trial mask is applied to select valid trials, then each valid trial's data is stored as a separate element in the session lists.

ii.
```python
trials = pd.read_parquet(trials_file)
# ... create mask ...
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
# ... for each trial in combined_indices, append neural/input/output
```

iii. The AI splits data per trial by aligning to stimulus onset times and extracting the 2-second window around each trial's alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding trials with NaN in required fields (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType); (2) reaction time between 0.08 and 2.0 seconds; (3) excluding no-choice trials (choice == 0); (4) requiring both wheel and whisker motion energy data to be available for the trial's time window.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
# ...
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)  # 0.08
mask &= (rt <= MAX_RT)  # 2.0
mask &= (trials['choice'] != 0)
# ...
combined_mask = wheel_mask & me_mask
```

iii. The AI documented these filters match the reference code's `load_trials_and_mask` defaults. However, the reference code also applies `max_trial_len=10.0` (filtering by `feedback_times - goCue_times`), which the AI omits.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps), `spikes.clusters.npy` (cluster assignment per spike), `clusters.channels.npy` (channel per cluster), `channels.brainLocationIds_ccf_2017.npy` (brain region IDs per channel), and `clusters.metrics.pqt` (quality metrics).

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. The AI loads the raw spike-sorted data directly from pykilosort output files, matching the reference code's use of `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Processing steps: (1) Load spike times and cluster assignments for each probe; (2) Optionally filter by cluster quality (label >= 1.0); (3) Merge spikes across probes by offsetting cluster IDs and sorting by time; (4) Bin spikes into 20ms bins per trial, aligned to stimulus onset, producing (n_clusters, 100) matrices per trial.

ii.
```python
# Merge probes
all_spike_clusters.append(spike_clusters + cluster_offset)
cluster_offset += n_clusters
sort_idx = np.argsort(merged_times, kind='stable')
# Bin spikes
bin_indices = np.minimum(((trial_times - t_beg) / binsize).astype(np.int64), n_bins - 1)
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The AI documented this matches the reference code's `merge_probes()` and `bin_spiking_data()` functions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies quality filtering with `qc_threshold=1.0` (cluster label >= 1.0), keeping only "well-isolated" neurons. This filters out multi-neuron activity units.

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    # ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
```

iii. The AI acknowledged in CONVERSION_NOTES.md: "The Zhang reference code uses qc=None (all clusters), but we apply quality filtering to match the BWM paper's decoding methodology." However, the reference code (`prepare_data` in `ibl_data_utils.py`) explicitly calls `load_spiking_data(one, pid, eid=eid, pname=probe_name)` without specifying `qc`, which defaults to `qc=None`. The methods paper also states "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are extracted from a window of -0.5 to 1.5 seconds relative to stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
# ...
align_times = valid_trials[ALIGN_TIME].values
# ...
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
```

iii. Matches the reference code's `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` and the methods paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 time bins per trial (2.0s / 0.02s). No rebinning is applied; spikes are directly binned at this resolution.

ii.
```python
BINSIZE = 0.02  # 20 ms bins
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. Matches the reference code (`'binsize': 0.02`) and methods paper ("divided into 20-ms bins, producing T = 100 time steps").

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from any raw data variable. It is computed as a linearly spaced array from the time window parameters.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. This represents the time coordinate of each bin center, from -0.48s to 1.5s. This is a deterministic input derived from the alignment parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced array of 100 values from `TIME_WINDOW[0] + BINSIZE` (-0.48s) to `TIME_WINDOW[1]` (1.5s) is created. This matches the bin centers used for behavior interpolation.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The values represent bin centers (shifted by one binsize from the start), consistent with the reference code's interpolation grid `np.linspace(interval_begs + binsize, interval_ends, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time array is identical for all trials since it represents relative time from stimulus onset. It uses the same bin center formula as the behavior interpolation, ensuring alignment with neural bins.

ii.
```python
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),  # (1, T) time-varying
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)  # (1, T) constant
])  # (2, T)
```

iii. Since neural data and this input share the same 100-bin structure aligned to stimulus onset, they are inherently aligned.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected by changes in `probabilityLeft`.

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

iii. The AI computes trial number within each block by detecting transitions in `probabilityLeft`. The reference code stores raw `probabilityLeft` as `block`, not trial number in block. The decoder task instructions explicitly request "Trial number in block" as an input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates over all trials (before masking) to detect block boundaries where `probabilityLeft` changes value. For each trial, the trial number is computed as the 0-indexed offset from the block start. The result is then filtered to only include valid (masked) trials.

ii.
```python
block_start = 0
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start
return trial_nums[mask]
```

iii. The computation uses the full (unmasked) trial sequence for block detection, which is correct since excluded trials shouldn't affect block boundary detection. The value is broadcast across all time bins for each trial as a constant input.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
# ...
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

iii. The raw data uses -1 for left and 1 for right choices. The AI converts to 0 (left) and 1 (right) as specified in the decoder task instructions.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple remapping: -1 (left) -> 0, 1 (right) -> 1. The value is broadcast as a constant across all 100 time bins.

ii.
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1
# ...
np.full((1, N_TIMEBINS), choice, dtype=np.int64),
```

iii. Matches the decoder task specification: "Choice, binary, per-trial, left = 0, right = 1". The reference code stores raw choice values (-1, 1) but the instructions require the 0/1 mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
```

iii. The reference code stores this as `block` = `trials_df['probabilityLeft']`. The AI uses the same source variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Categorical mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The value is broadcast as a constant across all 100 time bins.

ii.
```python
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
# ...
np.full((1, N_TIMEBINS), prior, dtype=np.int64),
```

iii. Matches the decoder task specification: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (wheel timestamps).

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The reference code loads wheel data via `SessionLoader.load_wheel()`, which internally uses the same raw files.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Processing steps: (1) Load raw wheel position and timestamps; (2) Interpolate position to uniform 1kHz sampling; (3) Compute velocity using Gaussian-smoothed derivative; (4) Take absolute value for speed; (5) Interpolate to match neural bin centers per trial.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
```

iii. The AI uses the same brainbox functions that `SessionLoader.load_wheel()` uses internally. The reference code calls `np.abs(sess_loader.wheel['velocity'].to_numpy())` for wheel-speed, which is equivalent.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session discretization using equal-frequency (tercile) bins. The 33.33rd and 66.67th percentiles of all valid wheel speed values within each session define two thresholds, producing 3 bins (0=low, 1=medium, 2=high).

ii.
```python
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
# ...
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)  # 0, 1, 2
```

iii. The reference code does not discretize wheel speed (it stores continuous values). The decoder task instructions require "Wheel speed discretized into 3 bins, time-varying." The AI's approach of per-session equal-frequency binning is reasonable but means bin edges vary across sessions.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same bin centers used for neural data: `np.linspace(t_beg + binsize, t_end, n_bins)` for each trial. This produces 100 values aligned to the neural time bins.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. Matches the reference code's `get_behavior_per_interval` which uses the same interpolation formula.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_leftCamera.times.npy` (camera timestamps), with fallback to right camera equivalents.

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The reference code uses `load_target_behavior(one, eid, 'left-whisker-motion-energy')` with fallback to `'right-whisker-motion-energy'`, which loads the same underlying files via `SessionLoader.load_motion_energy()`.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Processing: (1) Load pre-computed motion energy and camera timestamps; (2) Truncate to matching lengths; (3) Remove NaN values; (4) Interpolate to neural bin centers per trial using linear interpolation.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(ts_file).flatten()
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]
valid = ~np.isnan(me) & ~np.isnan(times)
```

iii. The AI loads the pre-computed ROI motion energy rather than computing it from video frames. This matches the reference code approach. The AI's NaN removal is an additional step not in the reference's `load_target_behavior`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: per-session equal-frequency (tercile) binning using the 33.33rd and 66.67th percentiles, producing 3 bins (0=low, 1=medium, 2=high).

ii.
```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
# ...
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)  # 0, 1, 2
```

iii. Same considerations as wheel speed discretization (9-c). Per-session binning is reasonable but results in inconsistent bin edges across sessions.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same alignment as wheel speed: interpolated to neural bin centers using `np.linspace(t_beg + binsize, t_end, n_bins)` with linear interpolation.

ii.
```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
```

iii. Uses the same `interpolate_behavior_to_bins` function as wheel speed, matching the reference code's `get_behavior_per_interval`.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Sessions with missing probe data, trial files, or fewer than 5 neurons are skipped entirely; (2) Trials with NaN in required fields are excluded via the trial mask; (3) Trials where behavioral data doesn't cover the full time window are excluded via `interpolate_behavior_to_bins`; (4) Whisker ME NaN values are pre-filtered before interpolation; (5) Sessions with fewer than 2 valid trials after all filtering are skipped; (6) Revision directories (e.g., `#2025-03-03#/`) are searched as fallback for file locations.

ii.
```python
# Coverage check in interpolate_behavior_to_bins
if np.abs(t_beg - local_times[0]) > binsize:
    good_mask[trial_idx] = False
if np.abs(t_end - local_times[-1]) > binsize:
    good_mask[trial_idx] = False
# NaN check
if np.any(np.isnan(local_vals)):
    good_mask[trial_idx] = False
```

iii. The AI's handling is generally thorough and matches the reference code's `get_behavior_per_interval` tolerance checks. The revision directory fallback is a practical addition for the local cache structure.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning (`bin_spikes_per_trial`), which loops over every trial and uses `np.searchsorted` and `np.add.at` to bin spikes. This is called once per session and processes all trials sequentially. Behavioral interpolation (`interpolate_behavior_to_bins`) is the second most expensive, also looping per trial. Loading large numpy arrays (spike times, clusters) from disk is also significant.

ii.
```python
for trial_idx in range(n_trials):
    # ... searchsorted, add.at per trial ...
```

iii. The reference code uses multiprocessing for both spike binning and behavior interpolation, while the AI's code is single-threaded.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop (`bin_spikes_per_trial`) could potentially be vectorized using `bincount2D` (as in the reference code) or by computing all trial bins simultaneously. The behavior interpolation loop could also benefit from vectorization or at least multiprocessing.

ii.
```python
# Current: loop over trials
for trial_idx in range(n_trials):
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    # ...
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The reference code uses `bincount2D` for spike binning and `multiprocessing.Pool` with `imap_unordered` for parallelization, which would be more efficient.

## 12-c. What processing does the code repeat multiple times?

i. (1) The `find_file_with_revision` function is called multiple times per session, iterating over directory contents each time; (2) Brain region mapping (`id2acronym`, `acronym2acronym`) is done per session even though the BrainRegions object is reusable; (3) The `discretize_to_bins` helper function is defined but not used (the main code uses `np.digitize` directly).

ii.
```python
# Called multiple times per probe
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
clusters_channels_file = find_file_with_revision(probe_dir, 'clusters.channels.npy')
clusters_metrics_file = find_file_with_revision(probe_dir, 'clusters.metrics.pqt')
channels_brain_ids_file = find_file_with_revision(probe_dir, 'channels.brainLocationIds_ccf_2017.npy')
```

iii. These repetitions are minor inefficiencies that don't affect correctness.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `discretize_to_bins` function (lines 381-397) is defined but never called; (2) The `session_metadata` list stores per-session information that is only used in the metadata dict and not in the decoder; (3) The sanity check statistics (choice/prior distributions) are computed and printed but not stored in a reusable form; (4) The code computes and stores `n_neurons` per session in metadata, which duplicates information derivable from the neural data arrays.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    """Discretize continuous values into n_bins equal-frequency bins."""
    # ... defined but never called
```

iii. The unused `discretize_to_bins` function suggests the AI initially planned a different discretization approach and later switched to inline `np.digitize` without cleaning up.
