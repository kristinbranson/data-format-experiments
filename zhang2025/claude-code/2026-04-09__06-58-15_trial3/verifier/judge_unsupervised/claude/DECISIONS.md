# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from a local ONE cache on disk (`/app/data/one_cache/`). It reads a CSV file (`bwm_release.csv`) from the reference code directory to get the list of sessions, then groups by `eid` to find all probes per session. For each session, it constructs the file path from the CSV metadata (lab, subject, date, session_number) and loads spike data (`.npy` files for spikes.times and spikes.clusters), trials table (`.pqt` parquet file), wheel data (position + timestamps `.npy`), and whisker motion energy (`.npy`). It handles revision directories (e.g., `#2024-05-06#/`) to find the latest versions of files.

ii.
```python
DATA_ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

# In main():
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
# ...
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)

# Loading trials:
trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
return pd.read_parquet(trials_file)

# Loading spikes:
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```

iii. The AI's CONVERSION_NOTES.md documents that it uses "Direct disk loading (bypasses ONE API since cache is read-only)" and reads the BWM release CSV containing 459 sessions across 139 subjects.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` field in the BWM release CSV. Each session's subject name is tracked in a list (`subject_set`), and a `subject_idx` array maps each session to its subject index. 135 subjects are retained after processing (4 lost because all their sessions were skipped).

ii.
```python
# In main():
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
# ...
'subjects': subject_set,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md states: "Subjects: 135 (97.1% of 139)" — 4 subjects lost because all their sessions were among the 21 skipped.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the BWM release CSV. The AI groups by eid and processes each session independently. Multiple probes within a session are merged. 438 of 459 sessions were successfully processed; 21 were skipped due to having 0 valid trials after filtering.

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

iii. CONVERSION_NOTES.md: "Sessions processed: 438, Sessions skipped: 21... All due to 'Too few valid trials (0)'"

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file for each session. Each trial corresponds to a row in this table. The trial time window is defined relative to stimulus onset: [-0.5, 1.5] seconds, producing 100 time bins at 20ms resolution.

ii.
```python
trials_df = load_trials(session_path)
# ...
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. CONVERSION_NOTES.md: "Time window: -0.5 to 1.5s... 100 time steps per trial"

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a mask that excludes: (1) trials with NaN in key event times (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (2) reaction time outside [0.08, 2.0]s, (3) trial length (feedback_times - goCue_times) > 10s, (4) no-choice trials (choice == 0). Additionally, trials where wheel speed or whisker motion energy could not be properly interpolated are excluded via a combined mask.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

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

# Combined with behavior masks:
combined_mask = mask & wheel_mask & me_mask
```

iii. CONVERSION_NOTES.md: "Trial mask: excludes RT < 0.08s or > 2.0s, NaN in key events, no-choice (choice==0), max_trial_len=10s." The behavior masks ensure wheel and whisker data are available for each trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster/neuron identity for each spike), loaded from each probe's pykilosort directory. Multiple probes are merged with cluster IDs re-indexed.

ii.
```python
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```

iii. CONVERSION_NOTES.md: "Load spikes+clusters for a probe... No QC filtering by default (qc=None)"

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms bins within the trial window [-0.5, 1.5]s relative to stimulus onset. The binning uses a fast vectorized approach with `np.bincount` and linear indexing. The result is spike counts per neuron per time bin. Data is stored as uint8 (since counts in 20ms bins rarely exceed 255).

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100

def bin_spikes_fast(spike_times, spike_clusters, interval_begs, interval_ends, n_clusters_total):
    b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
    lin_idx = c * N_BINS + b
    counts = np.bincount(lin_idx, minlength=minlength)
    binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
    # ...
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. CONVERSION_NOTES.md: "Spike binning uses np.bincount with linear indexing (4x faster than np.add.at)" and "20ms bins... T=100 time steps per trial."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All clusters from the spike sorting output are used, matching the Zhang et al. reference code which uses qc=None (no quality control filtering).

ii.
```python
# No cluster filtering code - all clusters used
n_clusters = int(merged_spikes['clusters'].max()) + 1
# In metadata:
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
```

iii. CONVERSION_NOTES.md: "No neuron quality filtering: load_spiking_data() called with default qc=None → ALL clusters used" and "We follow Zhang code: use ALL neurons (cluster label filtering is NOT applied)"

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the time window is [stimOn_times - 0.5, stimOn_times + 1.5] seconds. Spikes within this window are binned into 100 time bins of 20ms each.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. CONVERSION_NOTES.md: "Alignment event: stimOn_times" and "align trials to the stimulus onset"

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins over the 2-second trial window. No temporal rebinning is applied — the spike data is binned directly from raw spike times into 20ms bins.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
# In metadata:
'time_bin_size': BINSIZE * 1000,  # in ms → 20.0
```

iii. CONVERSION_NOTES.md: "Neural data time bin: 20ms... divided into 20-ms bins, producing T = 100"

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from a raw data variable per se. It is computed as a fixed array of time values representing the bin centers within the trial window [-0.5, 1.5]s. The time axis is the same for all trials.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
# Result: [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. CONVERSION_NOTES.md: "np.linspace(-0.48, 1.5, 100)" — this represents time bin centers starting at -0.48s (first bin center) through 1.5s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed linspace array is created from (TIME_WINDOW[0] + BINSIZE) to TIME_WINDOW[1] with N_BINS points. This represents the right edges / centers of the time bins, matching the reference code's behavior interpolation target times (`np.linspace(t_beg + binsize, t_end, N_BINS)`).

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. CONVERSION_NOTES.md: "Matching reference code: bin centers from interval_beg + binsize to interval_end"

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time_since_onset array has 100 elements matching the 100 neural time bins. Both use the same window [-0.5, 1.5]s relative to stimulus onset. The time values represent the same time points as the neural bins, so they are inherently aligned. The time array is broadcast across all trials as input[0].

ii.
```python
input_trial = np.zeros((2, N_BINS), dtype=np.float32)
input_trial[0, :] = time_since_onset  # same N_BINS as neural
```

iii. No explicit justification in notes; alignment is implicit from using the same N_BINS.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the trials table. Block transitions are detected when probabilityLeft changes value.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. CONVERSION_NOTES.md: "Compute from probabilityLeft transitions"

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The algorithm iterates through all trials in the session. When `probabilityLeft` changes value, a new block begins. The trial number is the 0-based index within the current block (i.e., trial_index - block_start_index). The computation is done on all trials first, then the filtered trial indices are used to extract the correct values.

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
```

iii. CONVERSION_NOTES.md: "Trial number in block: Compute from transitions in probabilityLeft values." The value is 0-indexed (first trial in block = 0). Per-trial scalar broadcast across all time bins.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table (`_ibl_trials.table.pqt`), where IBL convention is -1 for left and 1 for right.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]  # -1 or 1
```

iii. CONVERSION_NOTES.md: "IBL choice -1 (left) → 0, choice 1 (right) → 1"

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw IBL choice values (-1 for left, 1 for right) are mapped to binary values: -1 → 0 (left), 1 → 1 (right). The transformation is `(choice + 1) / 2`. This is then broadcast across all time bins as a per-trial value.

ii.
```python
choice = ((choice_raw + 1) / 2).astype(np.int32)  # 0 or 1
# ...
output_trial[0, :] = choice[trial_idx]  # broadcast per-trial
```

iii. CONVERSION_NOTES.md: "Choice mapping: IBL choice -1 (left) → 0, choice 1 (right) → 1"

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. CONVERSION_NOTES.md: "trials.probabilityLeft → output[1]: 'prior_probability_left' → Map: 0.2→0, 0.5→1, 0.8→2"

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probabilityLeft values are mapped to categorical indices: 0.2 → 0, 0.5 → 1, 0.8 → 2, matching the task specification. This is a per-trial value broadcast across all time bins.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
# ...
output_trial[1, :] = prior[trial_idx]  # broadcast per-trial
```

iii. CONVERSION_NOTES.md: "Prior probability: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (corresponding timestamps).

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. CONVERSION_NOTES.md: "Wheel velocity computed matching brainbox: interpolate to 1000Hz uniform, Butterworth LP filter (order=8, corner=20Hz), diff * fs"

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel speed is computed through multiple steps matching the brainbox library: (1) Interpolate raw position to 1000 Hz uniform sampling, (2) Apply Butterworth low-pass filter (order=8, corner frequency=20 Hz) using `sosfiltfilt`, (3) Compute velocity as filtered_diff * sampling_frequency, (4) Take absolute value for speed. The speed is then interpolated to the 20ms trial bins using linear interpolation.

ii.
```python
def load_wheel_speed(session_path):
    fs = 1000
    t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
    position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
    sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2,
                               btype='lowpass', output='sos')
    vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
    speed = np.abs(vel).astype(np.float32)
    return t.astype(np.float64), speed
```

iii. CONVERSION_NOTES.md: "Wheel speed = abs(velocity) following load_target_behavior() in reference code"

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency (quantile) bins across all valid time points within a session. The quantile thresholds (33rd and 67th percentiles) are computed from the flattened wheel speed data for all valid trials. Values are assigned to bins 0 (low), 1 (medium), or 2 (high).

ii.
```python
def discretize_to_bins(values, n_bins=3):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(valid, quantiles)
    result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
    result = np.clip(result, 0, n_bins - 1)
    return result

wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
```

iii. CONVERSION_NOTES.md: "Discretize into 3 equal-frequency (quantile) bins... Wheel speed discretization: 0.333 per bin (perfect equal-frequency)"

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same time bins as the neural data. The interpolation target times are `np.linspace(t_beg + BINSIZE, t_end, N_BINS)` — the same grid used for the time_since_onset input. The `interpolate_behavior` function checks that the behavior data covers the trial interval (within one bin size tolerance) and uses linear interpolation.

ii.
```python
def interpolate_behavior(beh_times, beh_values, interval_begs, interval_ends):
    x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
    y_interp = interp1d(trial_times, trial_vals, kind='linear',
                         fill_value='extrapolate')(x_interp)
```

iii. CONVERSION_NOTES.md: "Behavior interpolation: linear interpolation to bin centers at interval_beg + binsize, ..., interval_end"

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
def load_whisker_motion_energy(session_path):
    me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
    # Fall back to right camera if not found
    me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
```

iii. CONVERSION_NOTES.md: "Whisker ME: tries left camera first, falls back to right (matching reference code)"

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy values are loaded directly (no additional processing like filtering). They are then interpolated to the 20ms trial time bins using the same `interpolate_behavior` function as wheel speed.

ii.
```python
me_times, me_values = load_whisker_motion_energy(session_path)
if me_times is not None:
    binned_me, me_mask = interpolate_behavior(
        me_times, me_values, interval_begs, interval_ends)
```

iii. CONVERSION_NOTES.md: "Whisker ME... linear interpolation to bin centers"

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized identically to wheel speed: into 3 equal-frequency (quantile) bins across all valid time points within the session, producing categories 0 (low), 1 (medium), 2 (high).

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. CONVERSION_NOTES.md: "Whisker ME discretization: 0.333 per bin (perfect equal-frequency)"

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed — interpolated to the same time grid as neural data using `interpolate_behavior`, with target times `np.linspace(t_beg + BINSIZE, t_end, N_BINS)`.

ii.
```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

iii. Same alignment approach as wheel speed, using the reference code's `get_behavior_per_interval()` logic.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms handle missing/problematic data:
- Sessions with no spike data, no trials table, or no probes are skipped entirely.
- Trials with NaN in key events are excluded via the trial mask.
- Trials where wheel speed or whisker ME data doesn't cover the trial interval (gap > BINSIZE) are excluded via behavior masks.
- Sessions with fewer than 2 valid trials after filtering are skipped.
- When brain region IDs are missing, clusters are assigned to 'void'.
- The `find_file` function checks multiple revision directories to find files.
- Channel IDs are clipped to valid ranges to prevent index errors.
- Errors during session processing are caught and the session is skipped.

ii.
```python
# Skip session if too few trials:
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None

# Missing brain region:
cluster_beryl = np.array(['void'] * n_clusters)

# Clip channel IDs:
chan_ids = np.clip(chan_ids, 0, len(brain_ids) - 1)

# Try/except around session processing:
try:
    result = process_session(session_info, show_processing=show)
except Exception as e:
    print(f"  ERROR processing {eid}: {type(e).__name__}: {e}")
    n_skipped += 1
    continue
```

iii. CONVERSION_NOTES.md: "All 21 skipped sessions had 'Too few valid trials (0)' after applying the trial filtering mask AND behavior interpolation mask."

## 10-a. What are the most time-consuming steps of the code?

i. Based on the timing output, the most time-consuming steps are: (1) Spike binning (~0.4s/session avg), (2) Behavior loading and interpolation (~0.5-1.0s/session), and (3) Spike data loading from disk. Total processing averages ~4.6s/session, with the full conversion taking ~36 minutes for 459 sessions.

ii.
```python
# Timing in process_session:
t_spike_done = time.time()  # After spike binning
t_beh_done = time.time()    # After behavior processing
print(f"  spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s, total={t_format_done-t0:.1f}s")
```

iii. CONVERSION_NOTES.md: "Processing time: 2,149s (~36 min)... ~4.6s/session average"

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be further vectorized:
1. The spike binning inner loop in `bin_spikes_vectorized` iterates per-spike (though the faster `bin_spikes_fast` uses `np.bincount`, it still loops over trials).
2. The trial loop in `interpolate_behavior` processes each trial sequentially.
3. The brain region tracking loop in `main()` iterates per neuron.
4. The `compute_trial_number_in_block` uses a sequential loop over all trials.
5. The final formatting loop creating per-trial lists iterates over all trials.

ii.
```python
# Per-trial loop in bin_spikes_fast:
for i, trial_idx in enumerate(valid_idx):
    # ... per trial processing

# Per-trial loop in interpolate_behavior:
for trial_idx in range(n_trials):
    # ... per trial interp

# Per-neuron loop for brain regions:
for neuron_idx in range(result['n_neurons']):
    region = result['cluster_beryl'][neuron_idx]
```

iii. CONVERSION_NOTES.md mentions: "np.bincount linear indexing for spike binning: 1.6s → 0.4s per session" as a speedup that was implemented, but several other loops remain.

## 10-c. What processing does the code repeat multiple times?

i. The code computes reaction time (`firstMovement_times - stimOn_times`) twice in `create_trials_mask` (once for MIN_RT check, once for MAX_RT check). The `find_file` function is called separately for each file type, potentially scanning the same revision directories multiple times. The BrainRegions object is instantiated once per session rather than once globally.

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
br = BrainRegions()  # inside process_session()
```

iii. No explicit justification for this; the RT double-computation is minor and the BrainRegions instantiation per session was likely not identified as inefficient.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes several elements not strictly needed:
1. It loads `clusters.depths.npy` and `clusters.metrics.pqt` but never uses them in the final output.
2. It computes and stores `merged_depths` and `merged_channels` in `merge_probes` but doesn't use them.
3. The `bin_spikes_vectorized` function is defined but never called (the faster `bin_spikes_fast` is used instead).
4. The `reward` behavior (feedbackType) is mentioned in the reference code parameters but is not included as a decoder output — this is correct per the task spec.
5. The code saves `decoder_stats.json` which is not part of the required outputs.

ii.
```python
# Unused data loading:
if os.path.exists(depths_file):
    clusters['depths'] = np.load(depths_file).flatten()
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)

# Unused function:
def bin_spikes_vectorized(...):  # Never called

# In merge_probes, merged_channels and merged_depths are computed but not returned
```

iii. No explicit justification for loading unused data; likely included for potential future use or debugging.
