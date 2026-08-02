# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv` to identify the full set of sessions (459 sessions, 699 probes). It groups by `eid` (session ID) to get per-session probe lists. For each session, it constructs the file path from lab/subject/date/number and directly loads files from the ONE cache on disk (`/app/data/one_cache/`). Neural data is loaded from `spikes.times.npy` and `spikes.clusters.npy` in pykilosort directories. Trials are loaded from `_ibl_trials.table.pqt`. Wheel data from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. Whisker motion energy from `leftCamera.ROIMotionEnergy.npy` (or right camera fallback) with corresponding camera timestamps. Files in revision directories (e.g., `#2024-05-06#/`) are searched with a `find_file` helper.

ii.
```python
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATA_ROOT = '/app/data/one_cache'
# ...
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
# ...
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
# Load trials
trials_df = load_trials(session_path)  # reads _ibl_trials.table.pqt
# Load spikes
spikes = { 'times': np.load(times_file).flatten(), 'clusters': np.load(clusters_file).flatten() }
# Load wheel
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
# Load whisker ME
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
```

iii. The AI chose to bypass the ONE API and load files directly from disk because the data was already cached locally in the ONE cache format. This is documented in CONVERSION_NOTES.md Step 6: "Direct disk loading (bypasses ONE API since cache is read-only)". The reference code uses the ONE API and SpikeSortingLoader/SessionLoader, but the underlying data files are the same.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. As sessions are processed, unique subjects are collected into a `subject_set` list. Each session's subject is looked up and assigned a `subject_idx` pointing into this list.

ii.
```python
subject = result['subject']  # from bwm_df row
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

iii. This follows the same approach as the reference code, which uses `bwm_df.subject` to identify subjects. The AI reports 135 subjects (vs 139 in BWM release) due to 4 subjects whose sessions were all skipped.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`, grouped so that all probes for a session are collected together. Each session is processed independently via `process_session()`. Sessions that fail (missing data, no valid trials) are skipped.

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

iii. The reference code similarly iterates over `include_eids` from the BWM release. The AI processed 438 of 459 sessions (21 skipped due to missing data), which is close to the reference's 433 sessions.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file, where each row is one trial. For each trial, the time window is computed as `stimOn_times + [-0.5, 1.5]` seconds to define interval boundaries for neural and behavioral data extraction.

ii.
```python
trials_df = pd.read_parquet(trials_file)
stim_times = trials_df[ALIGN_TIME].values  # ALIGN_TIME = 'stimOn_times'
interval_begs = stim_times + TIME_WINDOW[0]  # -0.5
interval_ends = stim_times + TIME_WINDOW[1]  # +1.5
```

iii. This matches the reference code's `bin_spiking_data()` which computes intervals as `trials_df[align_time] + time_window[0]` and `trials_df[align_time] + time_window[1]`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a multi-stage trial filtering mask:
1. **NaN exclusion**: Trials with NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, or feedbackType are excluded.
2. **Reaction time filter**: RT (firstMovement_times - stimOn_times) must be in [0.08, 2.0] seconds.
3. **Trial length filter**: feedback_times - goCue_times must be <= 10 seconds.
4. **No-choice exclusion**: Trials where choice == 0 are excluded.
5. **Behavior coverage**: Trials must have valid wheel speed AND whisker motion energy data (behavior interpolation must succeed for both).

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
# ...
def create_trials_mask(trials_df):
    mask = np.ones(n_trials, dtype=bool)
    for event in NAN_EXCLUDE:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)  # 0.08
    mask &= (rt <= MAX_RT)  # 2.0
    trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()  # 10.0
    mask &= (trials_df['choice'] != 0)
    return mask
# ...
combined_mask = mask & wheel_mask & me_mask
```

iii. Steps 1-4 match the reference `load_trials_and_mask()` exactly (same defaults: min_rt=0.08, max_rt=2.0, nan_exclude='default', max_trial_len=10.0, exclude_nochoice=True). Step 5 matches the reference `align_spike_behavior()` which combines trial mask with behavior masks. Documented in CONVERSION_NOTES.md Steps 3-4.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps in seconds) and `spikes.clusters.npy` (cluster/neuron ID for each spike), loaded from the pykilosort spike sorting output directories. When multiple probes are present, spikes from all probes are merged with re-indexed cluster IDs.

ii.
```python
times_file = os.path.join(spike_dir, 'spikes.times.npy')
clusters_file = os.path.join(spike_dir, 'spikes.clusters.npy')
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
# Merge probes:
merged_spikes, cluster_brain_ids = merge_probes(spikes_list, clusters_list)
```

iii. The reference code loads the same data via `SpikeSortingLoader.load_spike_sorting()` which returns spikes['times'] and spikes['clusters']. Documented in CONVERSION_NOTES.md Step 1.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms time bins within a 2-second window ([-0.5, 1.5]s relative to stimOn_times), producing spike count matrices of shape (n_neurons, 100) per trial. The AI uses a fast binning approach: for each trial, spikes are found via binary search, bin indices are computed as `floor((time - interval_start) / binsize)`, and `np.bincount` with linear indexing accumulates counts. Multiple probes are merged by re-indexing cluster IDs and sorting by time.

ii.
```python
BINSIZE = 0.02          # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
# ...
def bin_spikes_fast(spike_times, spike_clusters, interval_begs, interval_ends, n_clusters_total):
    starts = np.searchsorted(spike_times, interval_begs[valid_idx], side='left')
    ends = np.searchsorted(spike_times, interval_ends[valid_idx], side='right')
    for i, trial_idx in enumerate(valid_idx):
        t = spike_times[s:e]
        c = spike_clusters[s:e]
        b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
        lin_idx = c * N_BINS + b
        counts = np.bincount(lin_idx, minlength=minlength)
        binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
    return binned
```

iii. The reference uses `bincount2D` for the same binning operation. The AI's approach is equivalent but uses a different implementation (np.bincount with linear indexing). Documented in CONVERSION_NOTES.md Step 6: "Spike binning uses np.bincount with linear indexing (4x faster than np.add.at)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron quality control filtering is applied. All clusters from Kilosort 2.5 are used, regardless of their quality label. This follows the reference code which calls `load_spiking_data()` with `qc=None` (the default).

ii.
```python
# No filtering in load_spike_data - all clusters loaded:
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
# n_clusters includes ALL cluster IDs:
n_clusters = int(merged_spikes['clusters'].max()) + 1
```

iii. From CONVERSION_NOTES.md Step 1: "No neuron quality filtering: load_spiking_data() called with default qc=None -> ALL clusters used". Step 4: "Follow code: use ALL neurons. Methods paper says 'all neurons, sorted by Kilosort 2.5'".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the interval is defined as `[stimOn_times - 0.5, stimOn_times + 1.5]` seconds. Spikes within this window are binned into 20ms bins.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
# ...
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. The instructions specify "Temporally align based on stimulus onset". The reference code uses `align_time='stimOn_times'` and `time_window=(-.5, 1.5)`. Match confirmed in CONVERSION_NOTES.md Step 3.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 time bins per 2-second trial. No temporal rebinning is applied; this is the native binning resolution.

ii.
```python
BINSIZE = 0.02          # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. The methods paper states "divided into 20-ms bins, producing T = 100" and the reference code uses `binsize=0.02`. The data paper mentions 50ms bins for choice/prior decoding and 20ms for dynamic behaviors, but the code uses a unified 20ms binning. Documented in CONVERSION_NOTES.md Step 4.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from a raw data variable per se. It is constructed from the time window parameters (`TIME_WINDOW` and `BINSIZE`) as a regularly-spaced time vector representing the center/right-edge of each bin relative to stimOn_times.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
# Produces values from -0.48 to 1.5 in 100 steps
```

iii. The AI chose to compute this as `np.linspace(-0.5 + 0.02, 1.5, 100)`, matching the interpolation grid used in the reference code's `get_behavior_per_interval()`: `np.linspace(interval_beg + binsize, interval_end, n_bins)`. Documented in CONVERSION_NOTES.md Step 5.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time vector is computed as 100 linearly-spaced values from -0.48 to 1.5 seconds. This corresponds to bin right-edges (or equivalently, interpolation target times) for the 20ms bins spanning [-0.5, 1.5]s. The same time vector is used for every trial.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
# For each trial:
input_trial[0, :] = time_since_onset
```

iii. The processing is minimal -- just constructing a regular time grid. This is the same grid the reference code uses for behavior interpolation targets.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset vector is defined on the same time grid as the neural data bins. Both use 100 bins covering [-0.5, 1.5]s relative to stimOn_times. The time values represent the right edge of each bin, matching the convention in the reference behavior interpolation code.

ii.
```python
# Neural bins: floor((spike_time - interval_beg) / BINSIZE), where interval_beg = stimOn - 0.5
# Time input: np.linspace(-0.5 + 0.02, 1.5, 100)
# Both have 100 elements aligned to the same trial window
input_trial = np.zeros((2, N_BINS), dtype=np.float32)
input_trial[0, :] = time_since_onset
```

iii. Since both neural and time input share the same 100-bin structure within the same time window, they are inherently aligned.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials table. Block boundaries are detected where `probabilityLeft` changes value (transitions between 0.2, 0.5, and 0.8).

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. The reference code stores `block = trials_df['probabilityLeft']` as a behavior variable but does not explicitly compute trial number in block. The AI inferred this variable from the task instructions which specify "Trial number in block" as a decoder input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through all trials (before filtering), tracking the current `probabilityLeft` value. When the value changes, a new block starts. The trial number within the current block is the index offset from the block start (0-indexed). The computation is done on the full (unfiltered) trial sequence, then indexed by the good trial indices.

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
# Applied to full trials, then filtered:
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices]
```

iii. The AI correctly computes this on the full trial sequence before applying the quality filter, so block boundaries are correctly identified. The value is stored as a per-trial scalar broadcast to all time bins.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`), which contains values -1 (left), 0 (no choice), and 1 (right).

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]  # -1 or 1
```

iii. The reference code similarly uses `trials_df['choice']`. No-choice trials (choice==0) are already excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL choice values (-1 for left, 1 for right) are mapped to binary values (0 for left, 1 for right) using the formula `(choice + 1) / 2`. The result is stored as a per-trial integer broadcast to all 100 time bins.

ii.
```python
choice = ((choice_raw + 1) / 2).astype(np.int32)  # -1->0, 1->1
# In output formatting:
output_trial[0, :] = choice[trial_idx]  # broadcast to all time bins
```

iii. The task instructions specify "Choice, binary, per-trial, left = 0, right = 1". In IBL convention, choice=-1 is left and choice=1 is right, so the mapping -1->0, 1->1 is correct.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in the trials table, which contains values 0.2, 0.5, or 0.8 representing the prior probability of the stimulus appearing on the left.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]  # 0.2, 0.5, 0.8
```

iii. The reference code stores this as `block = trials_df['probabilityLeft']`. The AI correctly identifies this as the source for the prior probability output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability values are mapped to categorical integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The result is a per-trial integer broadcast to all 100 time bins.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
# In output formatting:
output_trial[1, :] = prior[trial_idx]
```

iii. The task instructions specify "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2". The AI implements this mapping exactly.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position in radians) and `_ibl_wheel.timestamps.npy` (corresponding timestamps). These are the raw rotary encoder readings from the wheel.

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. The reference code uses `SessionLoader.load_wheel()` which internally loads the same position/timestamps files and processes them. Documented in CONVERSION_NOTES.md Step 1.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel speed is computed through a multi-step pipeline matching the brainbox SessionLoader:
1. **Interpolation**: Raw position is interpolated to uniform 1000 Hz sampling using linear interpolation.
2. **Butterworth filter**: A low-pass Butterworth filter (order=8, corner frequency=20 Hz) is applied using `sosfiltfilt`.
3. **Velocity**: Computed as `diff(filtered_position) * sampling_rate`, with a zero prepended.
4. **Speed**: Absolute value of velocity.
5. **Per-trial interpolation**: Speed is interpolated to the 100 trial time bins using linear interpolation with gap checking.

ii.
```python
# Interpolate to 1000 Hz
fs = 1000
t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
# Butterworth low-pass filter
sos = scipy.signal.butter(N=8, Wn=20 / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
# Speed = abs(velocity)
speed = np.abs(vel).astype(np.float32)
# Then interpolated to trial bins via interpolate_behavior()
```

iii. The reference code's `load_target_behavior(one, eid, 'wheel-speed')` calls `sess_loader.load_wheel()` which performs the same interpolation, filtering, and differentiation internally. The AI reimplements this to avoid the ONE API dependency. Documented in CONVERSION_NOTES.md Step 6.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency (quantile) bins per session. All wheel speed values across all valid trials and time bins within a session are flattened, and the 33.3rd and 66.7th percentiles are used as bin boundaries. Values are assigned to bins 0 (low), 1 (medium), or 2 (high).

ii.
```python
def discretize_to_bins(values, n_bins=3):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)  # [0, 33.3, 66.7, 100]
    thresholds = np.percentile(valid, quantiles)
    result = np.digitize(values, thresholds[1:-1], right=False)
    result = np.clip(result, 0, n_bins - 1)
    return result
# Applied per session:
wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
wheel_discrete_2d = wheel_discrete.reshape(wheel_data.shape).astype(np.int32)
```

iii. The task instructions say "Wheel speed discretized into 3 bins, time-varying" without specifying the discretization method. The AI chose equal-frequency quantile bins, which ensures balanced class distributions (verified as 0.333/0.333/0.333 in CONVERSION_NOTES.md Step 7). The discretization is computed per-session.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same time window as neural data: [-0.5, 1.5]s relative to stimOn_times. The continuous wheel speed signal is interpolated to 100 time bins matching the neural data bins, using linear interpolation with the target times at `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

ii.
```python
binned_wheel, wheel_mask = interpolate_behavior(
    wheel_times, wheel_speed, interval_begs, interval_ends)
# In interpolate_behavior:
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. The reference code's `get_behavior_per_interval()` uses the same interpolation approach: `np.linspace(interval_beg + binsize, interval_end, n_bins)`. Both use the same gap-checking logic (skip trial if behavior data starts too late or ends too early). Documented in CONVERSION_NOTES.md Step 1.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (motion energy values) and `_ibl_leftCamera.times.npy` (camera timestamps). If left camera data is unavailable or has a length mismatch, the right camera (`rightCamera.ROIMotionEnergy.npy` and `_ibl_rightCamera.times.npy`) is used as a fallback.

ii.
```python
def load_whisker_motion_energy(session_path):
    me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
    if me_file is not None and times_file is not None:
        me = np.load(me_file).flatten()
        times = np.load(times_file).flatten()
        if len(me) == len(times):
            return times, me
    # Fall back to right camera
    me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
    # ...
```

iii. The reference code's `bin_behaviors()` handles whisker-motion-energy by first trying `load_target_behavior(one, eid, 'left-whisker-motion-energy')` and falling back to the right camera if it fails. Documented in CONVERSION_NOTES.md Step 1.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy values are loaded directly (no additional processing like filtering or normalization). They are then interpolated to the 100 trial time bins using linear interpolation, the same way as wheel speed.

ii.
```python
me_times, me_values = load_whisker_motion_energy(session_path)
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

iii. The reference code similarly loads the raw motion energy and interpolates it per trial. The motion energy is already computed in the raw data as "the mean across pixels of the absolute value of the difference between adjacent frames" (data paper).

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into 3 equal-frequency (quantile) bins per session, using the same method as wheel speed. All values across valid trials and time bins within a session are pooled, percentile thresholds are computed, and values are assigned to bins 0 (low), 1 (medium), or 2 (high).

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. Same approach and justification as wheel speed discretization. Verified to produce equal-frequency bins (0.333/0.333/0.333) in CONVERSION_NOTES.md Steps 7 and 9.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to the same time window as neural data: [-0.5, 1.5]s relative to stimOn_times, interpolated to the same 100 time bins.

ii.
```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
# Same interpolation grid as wheel and neural data
```

iii. Same alignment approach as wheel speed. The reference code also uses unified stimOn alignment for all variables in the cached dataset.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data at multiple levels:
1. **Missing files**: If session path, trials table, spike data, or behavior data files are not found, the session is skipped entirely.
2. **NaN values**: Trials with NaN in key event times (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType) are excluded via the trial mask.
3. **Behavior coverage gaps**: The `interpolate_behavior` function checks that behavior data covers the trial interval (start and end within one bin size). Trials failing this check are marked as bad and excluded.
4. **Combined masking**: The final mask requires trial quality AND wheel AND whisker ME coverage to all pass. Only trials passing all criteria are included.
5. **Minimum trial count**: Sessions with fewer than 2 valid trials after filtering are skipped.
6. **Probe failures**: If some probes fail to load but others succeed, the session proceeds with available probes.

ii.
```python
# File-level handling:
if not os.path.exists(session_path): return None
if trials_file is None: return None
# NaN and quality:
mask = create_trials_mask(trials_df)
# Behavior coverage:
if np.abs(t_beg - trial_times[0]) > BINSIZE: continue  # gap check
# Combined mask:
combined_mask = mask & wheel_mask & me_mask
if n_good_trials < 2: return None
# Session-level error handling:
except Exception as e:
    print(f"  ERROR processing {eid}: {type(e).__name__}: {e}")
    n_skipped += 1
```

iii. The reference code handles missing data similarly: `load_trials_and_mask()` excludes NaN trials, `get_behavior_per_interval()` has the same gap checks, and `align_spike_behavior()` removes trials with missing behavior. Session-level errors are caught and the session is skipped.

## 12-a. What are the most time-consuming steps of the code?

i. Based on timing information in CONVERSION_NOTES.md Step 7:
1. **Loading spike data** (~0.9s/session): Reading spikes.times.npy and spikes.clusters.npy from disk.
2. **Loading/processing wheel data** (~0.5s/session): Reading position/timestamps, interpolating to 1000Hz, Butterworth filtering, computing velocity.
3. **Spike binning** (~0.4s/session): Binning spikes into 20ms trial bins.
Total: ~1.9s/session, ~15 minutes for 459 sessions (actual: ~36 minutes for 438 sessions at ~4.6s/session average).

ii.
```python
# Timing in process_session:
t0 = time.time()
# ... load data ...
t_spike = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike_done = time.time()
# ... load/process behaviors ...
t_beh_done = time.time()
print(f"  spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s, total={t_format_done-t0:.1f}s")
```

iii. Documented in CONVERSION_NOTES.md Step 7 Run Time Estimates.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain that could be further vectorized:
1. **`bin_spikes_fast` outer loop over trials**: Each trial is processed sequentially. Could potentially be vectorized by assigning all spikes to trials first, then using grouped bincount.
2. **`interpolate_behavior` loop over trials**: Each trial's behavior interpolation is sequential. Could use vectorized interpolation for all trials simultaneously.
3. **`compute_trial_number_in_block` loop**: Sequential scan could be vectorized using `np.diff` and `np.cumsum`.
4. **Output formatting loop**: The per-trial loop creating neural_list, input_list, output_list is sequential.

ii.
```python
# Trial loop in bin_spikes_fast:
for i, trial_idx in enumerate(valid_idx):
    # per-trial processing...

# Trial loop in interpolate_behavior:
for trial_idx in range(n_trials):
    # per-trial interpolation...

# Block number loop:
for i in range(len(prob_left)):
    if prob_left[i] != current_val:
        current_block_start = i
        current_val = prob_left[i]
    trial_numbers[i] = i - current_block_start
```

iii. The AI acknowledged some optimization in CONVERSION_NOTES.md Step 6: "np.bincount linear indexing for spike binning: 1.6s -> 0.4s per session" and "Vectorized searchsorted for trial boundaries". However, the inner loops remain.

## 12-c. What processing does the code repeat multiple times?

i. A few operations are computed redundantly:
1. **Reaction time**: Computed twice in `create_trials_mask` -- once for min_rt check and once for max_rt check.
2. **Brain region mapping**: `BrainRegions()` is instantiated for each session rather than once globally.
3. **File searching**: `find_file()` scans revision directories for each file type; some directories are listed multiple times.

ii.
```python
# RT computed twice:
if MIN_RT is not None:
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt <= MAX_RT)

# BrainRegions instantiated per session:
def process_session(...):
    br = BrainRegions()  # created fresh each time
```

iii. These are minor inefficiencies. The main processing (spike binning, behavior interpolation) is not repeated.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of data are loaded or computed but not used in the final output:
1. **Cluster depths and channels**: Loaded from disk (`clusters.depths.npy`, `clusters.channels.npy`) but only used for brain region mapping; the depths themselves are not in the output.
2. **Cluster metrics**: Loaded from `clusters.metrics.pqt` but never used (since no QC filtering is applied).
3. **Processing plots** (`--show-processing`): Generates plots that are only for verification, not part of the output.
4. **Wheel velocity filtering at 1000 Hz**: The full session-length wheel signal is processed at 1000 Hz before being downsampled to 100 trial bins at 50 Hz. This is necessary for the Butterworth filter but produces a very large intermediate signal.
5. **Binning ALL trials including masked ones**: Spikes and behaviors are binned for all trials before the mask is applied. Masked trials are then discarded.

ii.
```python
# Cluster metrics loaded but unused:
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)

# Cluster depths loaded but only brain_ids used:
if os.path.exists(depths_file):
    clusters['depths'] = np.load(depths_file).flatten()

# All trials binned before masking:
binned_spikes = bin_spikes_fast(..., interval_begs, interval_ends, ...)  # all trials
# Then: neural_data = binned_spikes[good_indices]  # only good trials kept
```

iii. The cluster metrics loading is entirely unnecessary since no QC is applied. Binning all trials before filtering is a design choice that simplifies the code but wastes computation on trials that will be discarded.
