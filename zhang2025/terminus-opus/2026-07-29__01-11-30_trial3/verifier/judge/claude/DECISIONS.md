# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from a local ONE cache directory structure (`data/one_cache/<lab>/Subjects/<subject>/<date>/<session>/alf/`). It identifies which sessions to process using `bwm_release.csv` (from the reference code repository), which lists all 459 BWM sessions with their eids, subjects, dates, labs, and probe names. For each session, it loads trial data from `_ibl_trials.table.pqt`, spike data from `<probe>/pykilosort/<version>/spikes.times.npy` and `spikes.clusters.npy`, wheel data from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, and whisker motion energy from `leftCamera.ROIMotionEnergy.npy`. This is an offline adaptation of the reference code's ONE API approach.

ii.
```python
DATA_DIR = 'data/one_cache'
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'

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

iii. The AI documented this in CONVERSION_NOTES.md Step 0-2. The approach mirrors the reference code's use of `bwm_release.csv` and ONE API, adapted for offline file access. The AI noted 459 sessions matched from bwm_release.csv.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. Each unique subject name is tracked with a `subject_to_idx` dictionary, and a `subject_idx` array maps each session to its subject.

ii.
```python
subject_to_idx = {}
for sess_i, sess_info in enumerate(sessions_info):
    subject = sess_info['subject']
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(all_subjects)
        all_subjects.append(subject)
    all_subject_idx.append(subject_to_idx[subject])
```

iii. This follows the standard approach of using bwm_df metadata to identify subjects, consistent with the reference code's `bwm_df.groupby('subject')` pattern.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. Each eid corresponds to one recording session. Multiple probes within the same session are merged. The AI groups bwm_df by eid and processes each unique session.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({
            'eid': eid, 'subject': subject, 'date': date,
            'probe_names': probe_names, 'path': session_map[key]
        })
```

iii. This matches the reference code's approach where sessions are identified by eid, and the `bwm_release.csv` file is the authoritative session list.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` (parquet file) in each session's alf directory. Each row represents one trial. Neural and behavioral data are segmented into per-trial intervals aligned to `stimOn_times` with a window of (-0.5, 1.5) seconds.

ii.
```python
trials_df = pd.read_parquet(trials_file)
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]  # stim_on - 0.5
interval_ends = stim_on + TIME_WINDOW[1]  # stim_on + 1.5
```

iii. This matches the reference code's `bin_spiking_data` which computes intervals from `trials_df[align_time] + time_window`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the following trial exclusion criteria:
- Exclude trials with NaN in: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType
- Exclude trials with reaction time < 0.08s or > 2.0s (firstMovement_times - stimOn_times)
- Exclude trials with trial duration > 10.0s (feedback_times - goCue_times)
- Exclude no-choice trials (choice == 0)
- Keep unbiased block trials (probabilityLeft == 0.5) since prior is an output with 0.5 as a valid class
- Additionally exclude trials where wheel or whisker motion energy interpolation fails

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

iii. The AI documented the rationale in CONVERSION_NOTES.md Step 4: the reference code's `load_trials_and_mask` defaults are `min_rt=0.08, max_rt=2.0, exclude_unbiased=False, exclude_nochoice=True`, and `prepare_data` adds `max_trial_len=10.0`. The AI correctly identified that `exclude_unbiased=False` is needed since the task requires decoding prior probability with 0.5 as a class. The data paper also specifies the same NaN exclusion and RT range (0.08-2.00s).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) from the pykilosort directory for each probe. Cluster-to-brain-region mapping comes from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. The reference code uses `SpikeSortingLoader.load_spike_sorting()` which loads the same underlying files. The AI loads them directly as an offline adaptation.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (concatenated with cluster ID offset), sorted by time, then binned into 20ms time bins within the trial interval (-0.5s to +1.5s relative to stimOn_times). The result is spike counts per cluster per time bin, yielding shape (n_clusters, 100) per trial.

ii.
```python
def bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                    interval_begs, interval_ends, binsize, n_bins):
    for trial_idx in range(n_trials):
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        trial_times = spike_times[i_start:i_end]
        trial_clusters = spike_clusters[i_start:i_end]
        bin_idx = np.minimum(
            ((trial_times - t_beg) / binsize).astype(np.int32), n_bins - 1)
        linear_idx = trial_clusters * n_bins + bin_idx
        np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
    return binned
```

iii. The reference code uses `bincount2D` from `iblutil.numerical` or `get_spike_counts_in_bins` with multiprocessing. The AI implements equivalent binning logic using searchsorted and np.add.at. Both produce spike count matrices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No QC filtering is applied. All clusters from the spike sorting output are used, regardless of quality label.

ii.
```python
# No QC filtering applied - all clusters loaded directly
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
```

iii. The AI documented this in CONVERSION_NOTES.md Step 1: "No QC filtering: load_spiking_data called without qc parameter (defaults to None = all clusters)". This matches the reference code's `prepare_data` which calls `load_spiking_data` without a `qc` parameter, meaning `qc=None` (all clusters).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the time interval is [stimOn_times - 0.5, stimOn_times + 1.5] seconds. Spikes within this interval are binned into 100 time bins of 20ms each.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. Matches the reference code parameters: `'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)` from `0_data_caching.py`, and the methods paper: "align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 time bins per trial over the 2-second window. No temporal rebinning is applied; spike counts are directly computed at 20ms resolution.

ii.
```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. Matches the reference code (`binsize: 0.02` in params) and the methods paper ("divided into 20-ms bins, producing T = 100 time steps").

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not derived from any raw data variable. It is computed analytically from the time window configuration parameters. It represents the time offset of each bin center relative to stimulus onset.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. This is a decoder input specified in the instructions ("Time since stimulus onset, continuous, time-varying"). The AI computes bin centers ranging from -0.49s to +1.49s. This is not present in the reference code since it is a new requirement from the task instructions.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linear array of bin centers is computed using `np.linspace`, spanning from the start of the window + half bin width to the end of the window - half bin width, producing 100 evenly spaced values. The same array is used for every trial.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,   # -0.5 + 0.01 = -0.49
    TIME_WINDOW[1] - BINSIZE/2,   # 1.5 - 0.01 = 1.49
    N_BINS                         # 100
).astype(np.float32)
```

iii. The choice of bin centers (rather than bin edges) is a reasonable representation of the time associated with each bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset has the same number of time bins (100) as the neural data. Each time bin in the input corresponds to the same time bin in the neural data. It is broadcast into the input array as the first row of a (2, n_bins) matrix per trial.

ii.
```python
inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)  # (2, n_bins)
input_list.append(inp)
```

iii. Since both neural and input data use the same number of time bins aligned to the same trial window, they are inherently aligned.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table (`_ibl_trials.table.pqt`). Block boundaries are detected when `probabilityLeft` changes value.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. The reference code stores `block = trials_df['probabilityLeft'].to_numpy()` as a behavior variable, but "trial number in block" is a new input specified in the task instructions. The AI chose to derive it from probabilityLeft transitions.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block changes are detected by comparing consecutive `probabilityLeft` values. Within each block, trials are numbered starting from 1. The computation is performed on ALL trials first (before filtering), then valid trial indices are selected. The value is broadcast to all time bins as a per-trial constant.

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

# Applied to ALL trials, then indexed for valid ones
all_trial_nums = compute_trial_number_in_block(all_prob_left)
valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```

iii. Computing block numbers from all trials before filtering ensures that trial numbers reflect the true position within the block, not the position among surviving trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column of `_ibl_trials.table.pqt`.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. Matches the reference code's `choice = trials_df['choice'].to_numpy()`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are -1 (left), 0 (no choice), and 1 (right). No-choice trials (choice == 0) are excluded by the trial mask. Remaining values are remapped: -1 -> 0 (left), 1 -> 1 (right). The per-trial value is broadcast to all time bins.

ii.
```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
# Later broadcast:
np.full(N_BINS, int(choice[t]), dtype=np.int64)
```

iii. This matches the task specification ("Choice, binary, per-trial, left = 0, right = 1").

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column of `_ibl_trials.table.pqt`.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
```

iii. Matches the reference code's `block = trials_df['probabilityLeft'].to_numpy()`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability values (0.2, 0.5, 0.8) are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 using `np.isclose` for robust floating-point comparison. The per-trial value is broadcast to all time bins.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
# Later broadcast:
np.full(N_BINS, int(prior[t]), dtype=np.int64)
```

iii. This matches the task specification ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2").

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (timestamps).

ii.
```python
pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The reference code loads the same underlying data via `SessionLoader.load_wheel()`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Velocity is computed from wheel position using a Butterworth low-pass filter (`velocity_filtered` from brainbox). The sampling frequency is estimated from the median timestamp difference. Speed is the absolute value of velocity. The speed is then interpolated to trial time bins using linear interpolation.

ii.
```python
dt_median = np.median(np.diff(timestamps))
fs = 1.0 / dt_median
velocity, _ = velocity_filtered(position, fs)
speed = np.abs(velocity)
# Then interpolated:
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_begs, interval_ends, BINSIZE, N_BINS)
```

iii. The reference code's `SessionLoader.load_wheel()` first interpolates the wheel position to a uniform 1000 Hz sampling rate using `interpolate_position`, then applies `velocity_filtered`. The AI skips the interpolation-to-uniform-sampling step, applying `velocity_filtered` directly to the raw timestamps. This is a potentially meaningful difference since `velocity_filtered` expects uniformly sampled input. Both take the absolute value for speed.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency bins using global quantile-based thresholds computed across all sessions. The 33.33rd and 66.67th percentiles of all wheel speed values are used as bin edges. Values are then discretized using `np.digitize`.

ii.
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. The task instructions specify "Wheel speed discretized into 3 bins, time-varying." The reference code does not discretize wheel speed (it uses continuous values). The AI chose quantile-based (equal-frequency) discretization, which is a standard approach for creating balanced categorical bins.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset with the same time window (-0.5, 1.5) as neural data. It is interpolated to trial time bins using the reference code's interpolation formula: `np.linspace(t_beg + binsize, t_end, n_bins)`.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The interpolation formula matches the reference code's `get_behavior_per_interval`: `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`. Both use the same alignment event and window.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_leftCamera.times.npy` (timestamps). Falls back to right camera if left camera data is unavailable.

ii.
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The reference code uses `SessionLoader.load_motion_energy(views=['left'])` which loads `whiskerMotionEnergy`. The fallback to right camera matches the reference: `load_target_behavior(one, eid, 'left-whisker-motion-energy')` falls back to `'right-whisker-motion-energy'` on error.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The whisker motion energy values are loaded directly without additional processing (no filtering, smoothing, or normalization). NaN values and length mismatches between values and timestamps are handled. The data is then interpolated to trial time bins using linear interpolation.

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]
me_times = me_times[valid]
# Then interpolated to trial bins
```

iii. The reference code also loads motion energy values directly and interpolates them. No additional processing is applied, which is consistent.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: 3 equal-frequency bins using global quantile-based thresholds (33.33rd and 66.67th percentiles) computed across all sessions.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. Same rationale as wheel speed discretization. Equal-frequency bins ensure balanced class distributions.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: aligned to stimulus onset with (-0.5, 1.5) window, interpolated to trial time bins using the reference code's formula.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS)
```

iii. Uses the same `interpolate_behavior_to_bins` function as wheel speed, matching the reference code's `get_behavior_per_interval`.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Missing whisker ME**: Entire sessions are skipped (15 out of 459 sessions, leading to 444 processed).
- **Missing wheel or whisker interpolation**: Individual trials are excluded via `combined_valid` mask.
- **NaN trial events**: Excluded by the trial mask via `NAN_EXCLUDE` list.
- **Length mismatches**: Camera timestamps/values truncated to minimum length.
- **NaN in behavioral data**: Removed before interpolation.
- **Interpolation failures**: Caught via try/except, trial marked as invalid.
- **All-zero neural data**: Kept (3 trials in one session noted in verification).
- **Missing probes**: Skipped; session still processed if at least one probe has data.

ii.
```python
# Length mismatch handling
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]

# NaN removal
valid = ~(np.isnan(me_values) | np.isnan(me_times))

# Interpolation failure handling
try:
    interp_func = interp1d(...)
    result[trial_idx] = interp_func(x_interp)
except Exception:
    valid_mask[trial_idx] = False

# Combined validity
combined_valid = wheel_valid & me_valid
if n_final < 2:
    return None  # Skip session
```

iii. The AI documented handling in CONVERSION_NOTES.md Steps 10-12. Sessions without whisker ME are skipped entirely rather than excluded per-trial, which causes 15 sessions and 3 subjects to be lost from the dataset.

## 12-a. What are the most time-consuming steps of the code?

i. Based on the conversion output (total ~37 minutes for 459 sessions, ~2.5s/session average), spike binning is the primary bottleneck. The `bin_spikes_fast` function iterates over all trials per session, performing searchsorted and np.add.at operations for each. Behavioral interpolation adds additional per-trial loop overhead. The session loop itself is sequential (no session-level parallelism).

ii.
```python
# Spike binning - main bottleneck
t0 = time.time()
binned_spikes = bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                                interval_begs, interval_ends, BINSIZE, N_BINS)
print(f'  Spike binning: {time.time()-t0:.1f}s')
```

iii. The reference code uses multiprocessing (`multiprocessing.Pool`) for spike binning via `get_spike_data_per_interval`, which parallelizes across trials. The AI's sequential approach is simpler but slower.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized or parallelized:
1. **Spike binning trial loop** (`bin_spikes_fast`): Iterates over trials sequentially. Could use multiprocessing (as reference code does) or vectorized 3D histogram operations.
2. **Behavior interpolation trial loop** (`interpolate_behavior_to_bins`): Each trial's interpolation is independent and could be parallelized.
3. **Trial number in block** (`compute_trial_number_in_block`): Simple loop that could use `np.diff` and `np.cumsum` for vectorized block detection.
4. **Output construction loop**: Building per-trial output arrays with np.stack in a loop could be done as batch array operations.

ii.
```python
# Trial loop in bin_spikes_fast
for trial_idx in range(n_trials):
    ...

# Trial loop in interpolate_behavior_to_bins
for trial_idx in range(n_trials):
    ...

# Trial loop in compute_trial_number_in_block
for i in range(len(prob_left)):
    ...
```

iii. The reference code uses `multiprocessing.Pool` for both spike binning and behavior interpolation. The AI noted timing information but did not implement parallelism.

## 12-c. What processing does the code repeat multiple times?

i. The following processing is repeated:
1. **Brain region ID-to-acronym mapping**: `br.id2acronym` and `br.acronym2acronym` are called per session, though the mapping itself is deterministic and could be cached.
2. **Time since stimulus onset**: The same `time_since_stim` array is computed once but used identically for every trial across all sessions.
3. **File finding**: `find_file` uses glob for each data file per session, repeating the directory traversal pattern.

ii.
```python
# Per-session brain region mapping
acronyms = br.id2acronym(cluster_brain_ids)
beryl_regions = br.acronym2acronym(acronyms, mapping='Beryl')
```

iii. These repetitions have minimal performance impact compared to the spike binning bottleneck.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. **`bin_spikes_vectorized` function** (lines 181-221): Defined but never called. `bin_spikes_fast` is used instead. This is dead code.
2. **Sorting merged spikes by time**: After merging probes, spikes are sorted by time. This is necessary for `searchsorted` in `bin_spikes_fast`, so it's not truly unnecessary.
3. **Storing raw wheel/ME values**: Raw continuous values are stored per session (`all_wheel_raw`, `all_me_raw`) and then discretized globally. The raw values are not saved in the final output, so they are intermediate data. However, this global discretization step is necessary for consistent binning.
4. **Processing plots**: Only generated when `--show-processing` is used, so not unnecessary in default mode.

ii.
```python
# Dead code - unused function
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters,
                          interval_begs, interval_ends, binsize, n_bins):
    ...
    for spike_i in range(len(trial_times)):
        c = trial_clusters[spike_i]
        b = bin_idx[spike_i]
        if c < n_clusters:
            binned[trial_idx, c, b] += 1
    return binned
```

iii. The dead code (`bin_spikes_vectorized`) was likely an earlier implementation that was replaced by the faster version but not cleaned up.
