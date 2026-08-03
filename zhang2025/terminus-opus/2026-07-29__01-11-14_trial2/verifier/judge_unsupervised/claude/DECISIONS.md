# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `bwm_release.csv` (699 PIDs, 459 sessions, 139 subjects). It groups entries by `eid` (session ID) to identify which probes belong to each session. It then checks which sessions are locally available by looking for their directory path in the ONE cache (`data/one_cache/<lab>/Subjects/<subject>/<date>/<session>/alf/`). For each available session, it directly loads raw numpy/parquet files from the ONE cache directory structure rather than using the IBL `ONE` API or `SpikeSortingLoader`.

ii.
```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'
BASE_PATH = Path('data/one_cache')

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({
        'pid': row.pid, 'probe_name': row.probe_name,
        'subject': row.subject, 'lab': row.lab, 'date': row.date,
    })

available_sessions = {}
for eid, probes in sessions.items():
    sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
    if sess_path is not None:
        available_sessions[eid] = probes
```

iii. The AI chose to load directly from the ONE cache file system instead of using the `ONE` API because the data was already downloaded locally. This avoids the API dependency while loading the same underlying data files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` field in `bwm_release.csv`. Each session's subject is stored during processing, and a unique sorted list of all subjects is constructed after processing all sessions. A `subject_idx` array maps each session to its subject index.

ii.
```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# ...
subject_idx_list.append(subject_to_idx[sess['subject']])
```

iii. The AI uses the same subject identifiers as in `bwm_release.csv`, consistent with the reference code which also uses `bwm_df.subject`.

## 1-c. How are the data split into sessions?

i. Sessions are identified by `eid` (experiment ID) from `bwm_release.csv`. Multiple probes from the same session are grouped together. Each session is processed independently, and the final dataset contains a list of sessions.

ii.
```python
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({...})
```

iii. This matches the reference code approach, which also groups probes by session eid.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` (parquet file) in each session's alf directory. A quality mask is applied, and only valid trials are kept. Spike data is binned per trial using intervals defined by `stimOn_times + TIME_WINDOW`. Behavioral data is interpolated into the same trial intervals.

ii.
```python
trials_df = pd.read_parquet(trials_file)
# After masking:
valid_trials = trials_df[trials_mask].copy()
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. This matches the reference code, which also defines trial intervals from `trials_df[align_time] + time_window`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on: (1) reaction time between 0.08s and 2.0s (`firstMovement_times - stimOn_times`), (2) trial length <= 10.0s (`feedback_times - goCue_times`), (3) no NaN values in key events (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (4) no-choice trials excluded (choice != 0), (5) trials without valid wheel speed or whisker motion energy data are excluded.

ii.
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN)
for event in NAN_EXCLUDE:
    mask &= ~trials_df[event].isna()
mask &= (trials_df['choice'] != 0)
# Later:
combined_valid = wheel_valid & me_valid
```

iii. The filtering parameters match those used in `load_trials_and_mask` in the reference code (min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True, same NaN exclusion list). The additional behavioral data validity filtering matches `align_spike_behavior` in the reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times (`spikes.times.npy`) and spike cluster assignments (`spikes.clusters.npy`) loaded from each probe's pykilosort directory. Cluster-to-channel mappings (`clusters.channels.npy`) and channel brain region IDs (`channels.brainLocationIds_ccf_2017.npy`) are also loaded.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
```

iii. The AI loads the same underlying spike sorting outputs that `SpikeSortingLoader.load_spike_sorting()` would load in the reference code.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (cluster IDs offset, sorted by time). Spikes are binned into 20ms bins within trial intervals (-0.5s to +1.5s relative to stimulus onset, 100 bins total). Spike counts are accumulated per cluster per bin using `np.add.at`. No QC filtering is applied to clusters (all clusters used). Brain regions are mapped from Allen CCF IDs to Beryl-mapped acronyms.

ii.
```python
BINSIZE = 0.02  # 20ms bins
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins

# Merge probes:
spike_times = np.concatenate(all_times)
spike_clusters = np.concatenate(all_clusters)
sort_idx = np.argsort(spike_times, kind='stable')

# Bin spikes:
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

iii. The processing matches the reference code: same binsize (0.02s), same time window, same number of bins (100), no QC filtering (qc=None), Beryl mapping for brain regions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to neurons/clusters. All clusters from the spike sorting output are used, matching the reference code's `qc=None` parameter.

ii.
```python
# No QC filtering code - all clusters loaded and used
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
# No filtering applied to spike_clusters
```

iii. The CONVERSION_NOTES.md states: "No QC filtering: load_spiking_data called with qc=None in prepare_data". This matches the reference code where `prepare_data` calls `load_spiking_data` without specifying a `qc` parameter (defaulting to `None`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the interval is `[stimOn_times - 0.5, stimOn_times + 1.5]` seconds. Spikes within each interval are binned into 100 bins of 20ms each.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. This matches the reference code parameters: `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 bins per trial over a 2-second window. No temporal rebinning is applied after the initial binning.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. This matches the reference code: `binsize=0.02` in both the AI's code and `0_data_caching.py`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is derived from the time window parameters (TIME_WINDOW = (-0.5, 1.5)) and the bin size (BINSIZE = 0.02). It is not directly derived from any raw data variable but rather computed as the time axis for the binned data, matching the interpolation x-points used in the reference code.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The time points are computed as `linspace(-0.5 + 0.02, 1.5, 100)` = `linspace(-0.48, 1.5, 100)`, matching the interpolation x-points in `get_behavior_per_interval`: `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time since stimulus onset is a fixed array computed once: `np.linspace(-0.48, 1.5, 100)`. This represents the right edges of 100 time bins of 20ms spanning -0.5s to 1.5s relative to stimulus onset.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. This is a simple computation that matches the reference code's behavior interpolation time points. The same array is used for all trials and sessions.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time_since_stim array is the same for all trials and is broadcast to every trial. It represents the same time bins as the neural data (100 bins of 20ms). It is stored as a (1, n_bins) row in the input array for each trial.

ii.
```python
inp = np.stack([
    sess['time_since_stim'],  # (n_bins,) - same for all trials
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)  # (2, n_bins)
```

iii. The time array is aligned by construction since it uses the same bin structure as the neural data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials table (`_ibl_trials.table.pqt`). A "block" is a consecutive sequence of trials with the same `probabilityLeft` value.

ii.
```python
trial_num_in_block = compute_trial_number_in_block(
    valid_trials_final['probabilityLeft']
)
```

iii. The AI uses `probabilityLeft` to define block boundaries, which is the standard IBL block variable. This is a reasonable choice given the task instructions mention "Trial number in block" as an input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The algorithm iterates through filtered trials and counts consecutive trials with the same `probabilityLeft` value. The counter resets to 1 whenever `probabilityLeft` changes. The result is a per-trial integer starting at 1 for the first trial in each block. This value is broadcast to all time bins for each trial.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block = prob_left.iloc[0] if hasattr(prob_left, 'iloc') else prob_left[0]
    count = 0
    for i in range(len(prob_left)):
        val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
        if val == current_block:
            count += 1
        else:
            current_block = val
            count = 1
        trial_nums[i] = count
    return trial_nums
```

iii. The AI computes trial number in block on the **filtered** trials (after quality mask applied), not on the full trials table. This means if a trial is filtered out within a block, the trial count may not accurately reflect the true trial position within the original block. This is a potential issue -- the block structure should be computed on all trials, then filtered trials selected.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table, where the IBL convention is: -1 = left, 1 = right, 0 = no choice.

ii.
```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

iii. The mapping follows the instruction specification: left = 0, right = 1. No-choice trials (choice == 0) are excluded by the trial filter.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL choice values (-1 for left, 1 for right) are remapped to binary (0 for left, 1 for right). The value is broadcast to all time bins in the output array, making it a per-trial constant repeated across time.

ii.
```python
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
# In output construction:
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
```

iii. This matches the instructions: "Choice, binary, per-trial, left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. This matches the instructions: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to categorical values (0, 1, 2). The result is broadcast to all time bins.

ii.
```python
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
# In output:
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. The mapping is straightforward and matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (wheel timestamps) in each session's alf directory.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. These are the standard IBL wheel data files, matching what `SessionLoader.load_wheel()` loads in the reference code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is: (1) interpolated to uniform 1000Hz sampling, (2) low-pass Butterworth filtered (corner=20Hz, order=8), (3) differentiated to get velocity, (4) absolute value taken to get speed. The speed is then interpolated to match neural time bins using `interp1d(kind='linear', fill_value='extrapolate')` at bin right edges.

ii.
```python
# Step 1: Interpolate to 1000Hz
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
# Step 2: Butterworth filter + velocity
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
# Step 3: Speed = abs(velocity)
speed = np.abs(vel).astype(np.float32)
```

iii. This matches the reference code's `interpolate_position` + `velocity_filtered` pipeline, and the `load_target_behavior` function which takes `abs()` of velocity for wheel-speed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 categories using global quantile boundaries (terciles computed across all sessions). All wheel speed values across all trials and time bins are pooled, then 1/3 and 2/3 quantiles are computed. Values are binned using `np.digitize`.

ii.
```python
# Global quantiles:
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
# Per session:
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. The instructions say "Wheel speed discretized into 3 bins, time-varying" but don't specify the discretization method. Using quantile-based terciles ensures roughly equal distribution across bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same time bins as neural data using `interp1d` with x-points `np.linspace(t_start + binsize, t_end, n_bins)`, matching the reference code's `get_behavior_per_interval`.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. This matches the reference code's behavior interpolation approach, ensuring temporal alignment between neural and behavioral data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is loaded from `leftCamera.ROIMotionEnergy.npy` (left camera first, falling back to right camera) and `_ibl_leftCamera.times.npy`.

ii.
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. The reference code also tries left camera first, then right camera for whisker motion energy, matching this approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Whisker motion energy values are loaded directly (no additional processing beyond what was already computed for the camera). NaN values are removed. The data is then interpolated to match neural time bins.

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. The reference code loads whisker motion energy similarly. The AI adds NaN removal as a data cleaning step.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into 3 categories using global quantile boundaries (terciles), the same approach as wheel speed.

ii.
```python
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. Same quantile-based approach as wheel speed, producing 3 categories.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned identically to wheel speed: interpolated to the same time bin structure using `interp1d` at right-edge time points.

ii.
```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. Uses the same `interpolate_behavior` function as wheel speed, ensuring consistent alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing data are handled: (1) Sessions with missing wheel data or whisker ME data are skipped entirely. (2) Individual trials with insufficient behavioral data coverage (data doesn't span the trial interval within one binsize tolerance) are marked invalid and excluded. (3) NaN values in key trial events exclude those trials. (4) Sessions with fewer than 2 valid trials are skipped. (5) Missing or unavailable sessions from the cache are skipped. (6) Versioned file directories are searched with a fallback mechanism.

ii.
```python
# Behavioral coverage check:
if np.abs(t_start - seg_times[0]) > binsize:
    valid_mask[trial_idx] = False
if np.abs(t_end - seg_times[-1]) > binsize:
    valid_mask[trial_idx] = False

# Combined validity:
combined_valid = wheel_valid & me_valid
if np.sum(combined_valid) < 2:
    return None

# Missing sessions:
if sess_path is None:
    return None
```

iii. The handling is generally robust and matches the reference code's approach of excluding trials with insufficient behavioral data coverage.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the timing output in the code, spike binning is the most time-consuming step, followed by wheel speed computation (which involves 1000Hz interpolation and Butterworth filtering) and whisker motion energy interpolation.

ii.
```python
t1 = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike = time.time() - t1

t1 = time.time()
wheel_times, wheel_speed = compute_wheel_speed(sess_path)
wheel_binned, wheel_valid = interpolate_behavior(...)
t_wheel = time.time() - t1
```

iii. The AI added timing instrumentation and estimated ~4s per session, ~30-35 minutes for the full dataset.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the trial loop in `bin_spikes_fast`, which iterates over trials sequentially. While it uses `searchsorted` for fast spike selection, the per-trial loop with `np.add.at` could potentially be parallelized or replaced with a fully vectorized approach. The `compute_trial_number_in_block` function also uses a sequential loop. The `interpolate_behavior` function loops over trials.

ii.
```python
# bin_spikes_fast: trial loop
for trial_idx in range(n_trials):
    idx_start = np.searchsorted(spike_times, t_start, side='left')
    idx_end = np.searchsorted(spike_times, t_end, side='left')
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)

# interpolate_behavior: trial loop
for trial_idx in range(n_trials):
    f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
    result[trial_idx] = f_interp(x_interp)
```

iii. The reference code uses multiprocessing (`ProcessPoolExecutor`) for spike binning and behavior interpolation to parallelize across trials. The AI's code processes trials sequentially within each session but does not parallelize.

## 10-c. What processing does the code repeat multiple times?

i. The code has a `bin_spikes_vectorized` function that is defined but never used (superseded by `bin_spikes_fast`). The `merge_probes` function is defined but the inline merging in `process_session` is used instead. The `discretize_to_bins` function is defined but not used (inline discretization in `build_final_dataset` is used instead).

ii.
```python
# Defined but unused:
def bin_spikes_vectorized(spike_times, spike_clusters, ...):
    ...

def merge_probes(spikes_list, clusters_info_list):
    ...

def discretize_to_bins(values, n_bins=3, quantiles=None):
    ...
```

iii. These are dead code artifacts from development. No processing is actually repeated during execution.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code broadcasts per-trial variables (choice, prior probability of left, trial number in block) to all time bins by creating `np.full(N_BINS, value)` arrays. While choice and prior are per-trial values that don't need to be repeated across time bins, the decoder framework requires consistent array shapes. The code also loads `stimOnTrigger_times` and `goCueTrigger_times` files that are not used in the final output. All clusters from all brain regions are included even though some regions may not be relevant to the decoder task.

ii.
```python
# Per-trial values broadcast to time bins:
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32)

# Unused loaded files:
stim_trig_file = find_versioned_file(sess_path, '_ibl_trials.stimOnTrigger_times.npy')
go_trig_file = find_versioned_file(sess_path, '_ibl_trials.goCueTrigger_times.npy')
```

iii. Broadcasting per-trial values to time bins is required by the data format specification, which expects (d_output, n_timepoints) shape arrays. The loaded but unused timing files add minor I/O overhead but don't affect correctness.
