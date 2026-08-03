# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by: (1) reading a CSV file (`bwm_release.csv`) from the reference code's data directory to get the list of session eids, subjects, dates, and probe names; (2) building a session map by traversing the ONE cache directory structure (`data/one_cache/<lab>/Subjects/<subject>/<date>/<session_num>/`); (3) matching sessions from the CSV to on-disk directories by (subject, date) key; (4) for each matched session, loading trials from parquet (`_ibl_trials.table.pqt`), spike data from numpy files, wheel data from numpy files, and whisker motion energy from numpy files.

ii.
```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'
DATA_DIR = 'data/one_cache'

def build_session_map(data_dir):
    """Build mapping from (subject, date) to session directory path."""
    session_map = {}
    for lab_dir in os.listdir(data_dir):
        lab_path = os.path.join(data_dir, lab_dir, 'Subjects')
        # ... traverse subject/date/session_num directories
        key = (subject, date)
        session_map[key] = sess_path
    return session_map

# In main():
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({...})
```

iii. The AI documented in CONVERSION_NOTES.md that the `bwm_release.csv` file contains 459 session eids, and the data cache has 461 directories. The AI matched sessions from the CSV to the on-disk data, resulting in 459 matched sessions. This approach is consistent with the reference code which also uses `bwm_release.csv` to identify sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. Each unique subject name is tracked via a `subject_to_idx` dictionary, building the `subjects` list and `subject_idx` array as sessions are processed.

ii.
```python
subject_to_idx = {}
# In processing loop:
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The AI noted 136 subjects in the final data (vs 139 in the reference) because 3 subjects were lost when all their sessions were skipped due to missing whisker motion energy data.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. Each eid maps to a session directory on disk. Sessions are processed one at a time. If a session fails (missing data), it is skipped. 444 of 459 sessions were successfully processed.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    # ... build sessions_info list
for sess_i, sess_info in enumerate(sessions_info):
    result = process_session(eid, sess_info['path'], sess_info['probe_names'], br, ...)
    if result is None:
        print(f'  SKIPPED')
        continue
```

iii. The AI identified 459 sessions from the CSV and processed 444 successfully. 15 were skipped due to missing whisker motion energy data. The reference code's `0_data_caching.py` also processes sessions by eid from `bwm_release.csv`.

## 1-d. How are the data split into trials?

i. Trials are loaded from the parquet file `_ibl_trials.table.pqt` in each session's `alf` directory. Each row is a trial. Valid trials are selected via a mask, and then further filtered by behavioral data availability. Neural and behavioral data are binned per trial using time windows relative to stimulus onset.

ii.
```python
def load_trials(alf_dir):
    trials_file = find_file(alf_dir, '_ibl_trials.table.pqt')
    trials_df = pd.read_parquet(trials_file)
    return trials_df

# In process_session:
trials_df = load_trials(alf_dir)
mask = create_trials_mask(trials_df)
valid_trials_df = trials_df[mask].reset_index(drop=True)
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The AI documented loading trials from parquet files, consistent with the reference code which also loads from ONE's trial tables.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the following trial filters: (1) reaction time < 0.08s or > 2.0s excluded; (2) trial duration > 10.0s excluded (feedback_times - goCue_times); (3) NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType excluded; (4) no-choice trials (choice == 0) excluded; (5) unbiased trials NOT excluded (EXCLUDE_UNBIASED = False). Additionally, trials where behavioral interpolation fails are excluded.

ii.
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trials_mask(trials_df):
    query_parts = []
    if MIN_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
    if MAX_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
    if MAX_TRIAL_LEN is not None:
        query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
    for event in NAN_EXCLUDE:
        query_parts.append(f'{event}.isnull()')
    if EXCLUDE_NOCHOICE:
        query_parts.append('(choice == 0)')
    query = ' | '.join(query_parts)
    mask = ~trials_df.eval(query)
    return mask
```

iii. The AI noted that the reference code's `load_trials_and_mask` function has defaults `min_rt=0.08`, `max_rt=2.0`, `exclude_unbiased=False`, `exclude_nochoice=True`, and `prepare_data` calls it with `max_trial_len=10.0`. The AI decided to include unbiased trials (matching the function default of `exclude_unbiased=False`) since prior probability of 0.5 is a valid output class in the decoder task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) files from the pykilosort spike sorting output, plus `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
def load_spikes(alf_dir, probe_name):
    spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
    spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
    cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
    channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
    return spike_times, spike_clusters, cluster_channels, channel_brain_ids
```

iii. The AI documented these as the standard IBL spike sorting output files. The reference code also loads spike times and clusters via `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (with cluster ID offsets). Spikes are then binned into 20ms time bins within a [-0.5, 1.5]s window around stimulus onset, producing spike counts per cluster per bin. The result is a (n_clusters, n_bins) matrix per trial with shape (n_clusters, 100). No firing rate normalization is applied — raw spike counts are used.

ii.
```python
def bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                    interval_begs, interval_ends, binsize, n_bins):
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        trial_times = spike_times[i_start:i_end]
        trial_clusters = spike_clusters[i_start:i_end]
        bin_idx = np.minimum(
            ((trial_times - t_beg) / binsize).astype(np.int32),
            n_bins - 1
        )
        linear_idx = trial_clusters * n_bins + bin_idx
        np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
    return binned
```

iii. The AI documented using 20ms bins and [-0.5, 1.5]s window, matching the reference code parameters. The binning approach is functionally equivalent to the reference `bincount2D` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No QC filtering is applied to clusters/neurons. All clusters from pykilosort are included regardless of quality metrics. This matches the reference code which calls `load_spiking_data` with `qc=None` (the default).

ii.
```python
# In load_spikes: all clusters loaded without filtering
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
# No quality filtering applied
```

iii. The AI noted in CONVERSION_NOTES.md: "No QC filtering (all clusters used, qc=None in reference code)" and "In prepare_data, load_spiking_data is called without qc parameter, so ALL clusters are included."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the time window is [stimOn_times - 0.5, stimOn_times + 1.5] seconds. Spikes within this window are binned into 100 bins of 20ms each.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

# In process_session:
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. The AI documented alignment to stimulus onset with the [-0.5, 1.5] window, matching the reference code's `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial over the 2-second window. No temporal rebinning is applied — this is the native binning resolution used throughout.

ii.
```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. The AI documented 20ms bins producing T=100 time steps, matching the reference paper statement "divided into 20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from any raw data variable directly. It is computed as a linspace array representing the time axis of each trial's bins, relative to stimulus onset time. The alignment event `stimOn_times` defines the zero point.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The AI created this as a deterministic time axis. Since trials are aligned to stimulus onset, the time values represent elapsed time from -0.5s to +1.5s relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linspace array from -0.49 to 1.49 (bin centers) with 100 points is generated. The same array is used for every trial since all trials share the same time window relative to stimulus onset.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,   # -0.5 + 0.01 = -0.49
    TIME_WINDOW[1] - BINSIZE/2,   # 1.5 - 0.01 = 1.49
    N_BINS                         # 100
).astype(np.float32)
```

iii. No explicit justification was given for using bin centers vs bin edges. This is a reasonable representation of elapsed time.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time_since_stim array has the same number of time points (100) as the neural data bins. Each time point corresponds to the center of the corresponding neural bin. The input is stacked as a (2, 100) array with trial_number_in_block.

ii.
```python
inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)  # (2, n_bins)
input_list.append(inp)
```

iii. Alignment is implicit through shared indexing — both neural data and time_since_stim have 100 bins covering the same window.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_prob = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
        trial_nums[i] = i - current_block_start + 1
    return trial_nums
```

iii. The AI computed block boundaries from changes in `probabilityLeft`, which encodes the block structure of the task. This is a reasonable approach since block identity is defined by the prior probability.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The computation: (1) iterate through all trials (before filtering); (2) detect block changes when `probabilityLeft` changes value; (3) assign trial number as position within the current block (starting at 1); (4) select only the values for valid (filtered) trials.

ii.
```python
# Compute from ALL trials, then select valid ones
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
# Select only valid trials
valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```

iii. The AI correctly computed trial numbers from all trials before filtering, then selected the values for valid trials. This ensures the numbering reflects the animal's actual experience.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```

iii. The AI documented that choice in the raw data is encoded as -1 (left), 0 (no choice), 1 (right), matching IBL conventions.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice values are remapped: -1 (left) -> 0, 1 (right) -> 1. No-choice trials (choice == 0) were already excluded by the trial mask. The choice value is broadcast to all time bins as a per-trial constant.

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)

# In output construction:
np.full(N_BINS, int(choice[t]), dtype=np.int64),
```

iii. The AI followed the decoder task specification: "Choice, binary, per-trial, left = 0, right = 1". The reference code also extracts choice from `trials_df['choice']`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. The AI documented the mapping matches the decoder task specification: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2". The reference code stores `probabilityLeft` as `block`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to categorical indices (0, 1, 2) using `np.isclose` for floating point comparison. The prior value is broadcast to all time bins.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2

# In output:
np.full(N_BINS, int(prior[t]), dtype=np.int64),
```

iii. The AI used `np.isclose` to handle floating point comparison, which is a robust approach.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (timestamps).

ii.
```python
def load_wheel_data(alf_dir):
    pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
    ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
    position = np.load(pos_file).flatten()
    timestamps = np.load(ts_file).flatten()
```

iii. The AI documented loading wheel position and timestamps from numpy files, matching the IBL data architecture.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Processing: (1) Load wheel position and timestamps; (2) compute sampling frequency from median timestamp diff; (3) compute velocity using `velocity_filtered` from brainbox (Butterworth low-pass filter); (4) take absolute value to get speed; (5) interpolate to trial time bins using linear interpolation; (6) discretize into 3 equal-frequency bins using global quantile-based edges.

ii.
```python
from brainbox.behavior.wheel import velocity_filtered

def load_wheel_data(alf_dir):
    # ...
    fs = 1.0 / dt_median
    try:
        velocity, _ = velocity_filtered(position, fs)
    except Exception:
        # Fallback to simple diff
        velocity = np.zeros_like(position)
        velocity[1:] = np.diff(position) / dt
        velocity[0] = velocity[1]
    speed = np.abs(velocity)
    return timestamps, speed
```

iii. The AI initially used simple finite differences but later updated to use `velocity_filtered` from the brainbox library after discovering the reference code uses `SessionLoader.load_wheel()` which applies Butterworth filtering. However, the reference code's `SessionLoader.load_wheel()` uses Gaussian smoothing for velocity, not necessarily `velocity_filtered`. The `velocity_filtered` function uses a Butterworth filter, which is a different smoothing method.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency bins using global quantile-based thresholds (33.3rd and 66.7th percentiles computed across all sessions).

ii.
```python
# Global discretization
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])

# Per-session discretization
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. The AI used quantile-based binning to ensure approximately equal frequency in each bin. The decoder task specified "Wheel speed discretized into 3 bins" without specifying the binning method.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same time bins as neural data using `np.linspace(t_beg + binsize, t_end, n_bins)` for the interpolation points, matching the reference code's approach.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, interval_begs, interval_ends,
                                  binsize, n_bins):
    for trial_idx in range(n_trials):
        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
        interp_func = interp1d(local_times, local_vals, kind='linear',
                               fill_value='extrapolate')
        result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The AI matched the reference code's interpolation formula: `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_leftCamera.times.npy` (timestamps). Falls back to right camera if left is unavailable.

ii.
```python
def load_whisker_me(alf_dir):
    me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
    if me_file is None or times_file is None:
        me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
    me_values = np.load(me_file).flatten()
    me_times = np.load(times_file).flatten()
```

iii. The AI documented the left-first-then-right fallback, matching the reference code's `bin_behaviors` function which tries left camera first.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Processing: (1) Load motion energy and timestamps; (2) handle length mismatches by truncating to minimum length; (3) remove NaN values; (4) interpolate to trial time bins using linear interpolation; (5) discretize into 3 equal-frequency bins using global quantile-based edges.

ii.
```python
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. The AI handled edge cases of length mismatches and NaN values. The reference code loads via `SessionLoader.load_motion_energy()` which handles these internally.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker ME is discretized into 3 equal-frequency bins using global quantile-based thresholds (33.3rd and 66.7th percentiles computed across all sessions), same approach as wheel speed.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. Same quantile-based approach as wheel speed. The decoder task specified "Whisker motion energy discretized into 3 bins."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation method as wheel speed: linear interpolation to time bins using `np.linspace(t_beg + binsize, t_end, n_bins)`.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. Uses the same `interpolate_behavior_to_bins` function as wheel speed, matching the reference code's approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/problematic data: (1) sessions missing whisker motion energy are skipped entirely (15 sessions); (2) trials where behavioral interpolation fails are excluded; (3) NaN values in trial metadata trigger trial exclusion; (4) length mismatches between camera times and motion energy values are handled by truncation; (5) NaN values in behavioral data are removed before interpolation; (6) spike binning handles empty trials (no spikes) gracefully; (7) the `velocity_filtered` function has a fallback to simple differentiation.

ii.
```python
# Session skipped if no whisker ME
if me_times is None:
    print(f'  No whisker motion energy data found')
    return None

# Behavior validity mask
combined_valid = wheel_valid & me_valid
if n_final < 2:
    return None

# Length mismatch handling
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]

# NaN removal
valid = ~(np.isnan(me_values) | np.isnan(me_times))
```

iii. The AI documented in CONVERSION_NOTES.md that 15 sessions were skipped for missing whisker ME data and 3 trials had all-zero neural data.

## 10-a. What are the most time-consuming steps of the code?

i. Spike binning is the most time-consuming step. The conversion log shows binning times ranging from 0.2s to 1.8s per session (varying with number of spikes and trials). Total conversion time was ~37 minutes for 459 sessions.

ii.
```python
t0 = time.time()
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
print(f'  Spike binning: {time.time()-t0:.1f}s, shape={binned_spikes.shape}')
```

iii. The AI noted timing information in the conversion output and estimated ~2.5s per session on average.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over trials in `bin_spikes_fast` iterates trial-by-trial using `np.searchsorted` and `np.add.at`. While each iteration uses vectorized numpy operations within the trial, the outer trial loop could potentially be parallelized. The `interpolate_behavior_to_bins` function also loops over trials. The `compute_trial_number_in_block` function uses a simple Python loop but this is trivially fast.

ii.
```python
# bin_spikes_fast: trial loop
for trial_idx in range(n_trials):
    # ... vectorized within trial

# interpolate_behavior_to_bins: trial loop
for trial_idx in range(n_trials):
    # ... per-trial interpolation

# compute_trial_number_in_block: element loop
for i in range(len(prob_left)):
    # ... simple iteration
```

iii. The AI implemented `bin_spikes_fast` as an optimized version of `bin_spikes_vectorized`, using `np.searchsorted` for binary search and `np.add.at` for accumulation instead of per-spike Python loops.

## 10-c. What processing does the code repeat multiple times?

i. The code has two spike binning functions (`bin_spikes_vectorized` and `bin_spikes_fast`) but only uses `bin_spikes_fast`. The `bin_spikes_vectorized` function is dead code. The behavioral interpolation function is called twice (once for wheel, once for whisker ME) with the same trial intervals but different behavioral data — this is necessary, not redundant. The session loop recomputes `time_since_stim` implicitly (though it's actually computed once outside the loop in the final code).

ii.
```python
# Dead code - bin_spikes_vectorized is defined but never called
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters,
                          interval_begs, interval_ends, binsize, n_bins):
    # ... slower version with per-spike Python loop

# Used version
def bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                    interval_begs, interval_ends, binsize, n_bins):
    # ... optimized version
```

iii. The AI created `bin_spikes_fast` as a replacement for `bin_spikes_vectorized` but left the old function in the code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `bin_spikes_vectorized` function is dead code. (2) The code sorts merged spike times by time after concatenation (`np.argsort`), which is needed for `searchsorted` but adds overhead. (3) The code saves raw wheel and whisker ME data per session before global discretization, requiring two passes over the data. (4) Brain region mapping is done for all clusters including those in regions like "root" and "void" that may not be scientifically meaningful, but these are included per the reference code. (5) The `discretize_continuous` function is defined but never called (global discretization is done inline).

ii.
```python
# Dead code: discretize_continuous function
def discretize_continuous(values, n_bins=3):
    """Discretize continuous values into n_bins equal-frequency bins."""
    all_vals = values[~np.isnan(values)].flatten()
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_vals, percentiles)
    # ...

# Sorting merged spikes
sort_idx = np.argsort(merged_times)
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]
```

iii. The AI did not explicitly document these inefficiencies. The dead code and unused function are artifacts of iterative development.
