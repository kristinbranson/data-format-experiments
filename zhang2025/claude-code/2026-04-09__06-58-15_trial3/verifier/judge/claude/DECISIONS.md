# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly reading files from disk rather than using the ONE API. It reads a `bwm_release.csv` file from the reference code directory to get the list of sessions and their metadata (eid, lab, subject, date, session_number, probe_names). For each session, it constructs file paths manually to load trials, spikes, wheel data, and whisker motion energy from the ONE cache directory structure.

ii.
```python
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATA_ROOT = '/app/data/one_cache'
# ...
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
# ...
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
```

iii. The AI chose direct disk access because the ONE cache is read-only and the API requires network connectivity. The CONVERSION_NOTES.md states "Direct disk loading (bypasses ONE API since cache is read-only)."

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the `bwm_release.csv` file. As sessions are processed sequentially, subjects are appended to a list in the order they are first encountered. The `subject_idx` maps each session to its subject's index in this list.

ii.
```python
row = group.iloc[0]
# ...
session_list.append((eid, row['lab'], row['subject'], ...))
# ...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

iii. The subject identity comes from the BWM release CSV which lists the subject per session.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their `eid` in the BWM release CSV. The CSV is grouped by `eid`, and each group represents one session with one or more probes.

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

iii. The BWM release CSV naturally organizes data by session eid, so each unique eid is one session.

## 1-d. How are the data split into trials?

i. Trials come from the `_ibl_trials.table.pqt` parquet file loaded per session. Each row in the trials table is one trial.

ii.
```python
def load_trials(session_path):
    alf_path = os.path.join(session_path, 'alf')
    trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    return pd.read_parquet(trials_file)
```

iii. The trials table is already one row per trial in the IBL data format.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) NaN exclusion for stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType; (2) Reaction time between 0.08s and 2.0s; (3) Trial length (feedback_times - goCue_times) <= 10s; (4) No-choice exclusion (choice != 0); (5) Behavior coverage — wheel speed and whisker ME must have valid interpolation over the trial window.

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
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    if MAX_TRIAL_LEN is not None:
        if 'goCue_times' in trials_df.columns:
            trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
            mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    if EXCLUDE_NOCHOICE:
        mask &= (trials_df['choice'] != 0)
    return mask
```

iii. The CONVERSION_NOTES state these filters match `load_trials_and_mask()` from the reference code. The NaN exclusion and RT filtering are from the reference. The max_trial_len=10s filter is an additional filter from the reference code's defaults.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from each probe's pykilosort directory. Brain region information comes from `channels.brainLocationIds_ccf_2017.npy` mapped through cluster channel assignments.

ii.
```python
def load_spike_data(session_path, probe_name):
    times_file = os.path.join(spike_dir, 'spikes.times.npy')
    clusters_file = os.path.join(spike_dir, 'spikes.clusters.npy')
    spikes = {
        'times': np.load(times_file).flatten(),
        'clusters': np.load(clusters_file).flatten(),
    }
```

iii. These are the standard spike sorting outputs from pykilosort.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over a 2s window (-0.5 to 1.5s relative to stimulus onset), producing 100 time bins. The binning uses `np.bincount` with linear indexing for speed. The result is stored as **spike counts** (not firing rates) in uint8 format. When a session has multiple probes, clusters are merged by offsetting cluster indices and sorting by spike time.

ii.
```python
bin_idx = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
lin_idx = c * N_BINS + b
counts = np.bincount(lin_idx, minlength=minlength)
binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
# ...
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. The CONVERSION_NOTES state this matches `bin_spiking_data()` from the reference code with 20ms bins. However, the data is stored as spike counts (uint8) rather than firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron quality filtering is applied.** All clusters from the spike sorting are used, regardless of their quality label. The AI explicitly chose to follow the Zhang et al. reference code which calls `load_spiking_data()` with `qc=None` (no quality control filtering). Additionally, `void` brain regions (channels placed outside the brain) are NOT filtered out.

ii.
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0
# No filtering by cluster label or brain region
```

The metadata explicitly documents this:
```python
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
```

iii. The CONVERSION_NOTES state: "No neuron quality filtering: load_spiking_data() called with default qc=None -> ALL clusters used." The AI identified that the Zhang code uses no QC filtering and followed that choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial intervals are computed as stimulus onset time +/- the time window. Spikes within each trial interval are binned relative to the interval start (stimulus onset - 0.5s).

ii.
```python
stim_times = trials_df[ALIGN_TIME].values   # ALIGN_TIME = 'stimOn_times'
interval_begs = stim_times + TIME_WINDOW[0]  # stimOn - 0.5
interval_ends = stim_times + TIME_WINDOW[1]  # stimOn + 1.5
# ...
bin_idx = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
```

iii. All variables are aligned to stimulus onset, matching the reference code's unified approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (BINSIZE = 0.02s), producing 100 time bins over a 2s window. No rebinning is applied — spikes are directly binned at this resolution.

ii.
```python
BINSIZE = 0.02          # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. This matches the reference code parameter `'binsize': 0.02` and the methods paper description "divided into 20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the time window parameters and bin size. It does not come from a raw data variable — it is a constructed time axis.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The time axis is defined by the binning parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as `np.linspace(-0.48, 1.5, 100)`, which produces 100 evenly spaced points from -0.48 to 1.5. This corresponds to the right edges of the 20ms bins (or equivalently, `interval_beg + binsize` to `interval_end`), matching the reference code's `get_behavior_per_interval()` interpolation targets.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES describe this as "Matching reference code: bin centers from interval_beg + binsize to interval_end." However, these are actually the right bin edges, not bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values use the same binning grid as the neural data — both use 100 bins over the same 2s window. However, the neural bins are counted starting from the interval start, while the time values are computed as `linspace(-0.48, 1.5, 100)` which are the right bin edges, creating a systematic 10ms offset from the bin centers used for neural binning.

ii.
```python
# Neural: bins from interval_beg = stimOn - 0.5
bin_idx = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
# Input time: linspace from -0.48 to 1.5
time_since_onset = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The AI intended these to match the reference code's behavior interpolation approach.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where `probabilityLeft` changes value.

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

iii. The trials table carries no explicit block identifier, so blocks must be recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected by checking where `probabilityLeft` changes value. Within each block, the trial number is the offset from the block start (0-indexed). The computation is performed on ALL trials (before filtering), so filtered-out trials still advance the count, preserving the animal's true position in the block.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. Computing on all trials before filtering ensures the block count reflects the animal's actual experience.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column from the trials table, which takes values -1, 0, or 1 in the IBL convention.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]       # -1 or 1
```

iii. Choice is directly available in the trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice values using the formula `((choice_raw + 1) / 2)`. The AI's code comment states "Choice: -1 (left) -> 0, 1 (right) -> 1", but in the IBL convention, choice = +1 is left and choice = -1 is right. So the actual mapping is: left(+1) -> 1, right(-1) -> 0, which is **reversed** from the instructions (left=0, right=1).

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = ((choice_raw + 1) / 2).astype(np.int32)  # 0 or 1
```

iii. The AI's comment incorrectly describes the IBL convention. In IBL, +1 = left, -1 = right. The formula maps +1 -> 1 and -1 -> 0, which is the reverse of the required mapping (left=0, right=1).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column from the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. This is directly available in the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The mapping is 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, implemented with boolean indexing.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
```

iii. This follows the instructions exactly: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2."

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel position (`_ibl_wheel.position.npy`) and timestamps (`_ibl_wheel.timestamps.npy`), loaded and processed into speed (absolute velocity).

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. These are the raw wheel recording files from the IBL data.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI reimplements the brainbox wheel processing: (1) Interpolate position to 1000 Hz uniform sampling; (2) Apply Butterworth low-pass filter (order=8, corner=20Hz); (3) Compute velocity as filtered diff * fs; (4) Speed = abs(velocity). Then the speed is interpolated to trial bin times using scipy.interpolate.interp1d and discretized into 3 equal-frequency (quantile) bins.

ii.
```python
fs = 1000
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

iii. The CONVERSION_NOTES state this matches brainbox `interpolate_position()` + `velocity_filtered()`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel speed values (all trials in a session flattened) are discretized into 3 equal-frequency (quantile) bins using percentile thresholds at 33.3% and 66.7%. The bins are labeled 0 (low), 1 (medium), 2 (high).

ii.
```python
def discretize_to_bins(values, n_bins=3):
    quantiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(valid, quantiles)
    result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
    result = np.clip(result, 0, n_bins - 1)
    return result

wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
```

iii. Equal-frequency binning ensures approximately equal class sizes, matching the reference approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to trial-level time points using `np.linspace(t_beg + BINSIZE, t_end, N_BINS)`, which gives the right bin edges from interval_beg + 0.02 to interval_end. This is offset by 10ms from the bin centers used for neural data.

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                     fill_value='extrapolate')(x_interp)
```

iii. The AI describes this as "matching reference code get_behavior_per_interval()" interpolation targets.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy of a side camera: `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding timestamps `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

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
    # ...
```

iii. The AI tries the left camera first and falls back to right, matching the reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no additional filtering or normalization). It is interpolated to trial bin times using the same approach as wheel speed, then discretized into 3 equal-frequency bins.

ii.
```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: 3 equal-frequency (quantile) bins computed from all trials in a session, using percentile thresholds at 33.3% and 66.7%.

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. Same discretization approach as wheel speed, producing balanced bin distributions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated to `np.linspace(t_beg + BINSIZE, t_end, N_BINS)` (right bin edges), which is offset by 10ms from the neural data bin centers.

ii.
```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
# Uses same x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. Same interpolation approach as all behavioral variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms: (1) Sessions with missing trial tables, spike data, or no clusters are skipped; (2) The NaN exclusion filter drops trials where key events are NaN; (3) The behavior interpolation coverage check drops trials where wheel or whisker data doesn't span the trial window; (4) Sessions with fewer than 2 valid trials are skipped; (5) File loading checks for existence before attempting to read.

ii.
```python
if result is None:
    n_skipped += 1
    continue
# ...
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None
# ...
if np.abs(t_beg - trial_times[0]) > BINSIZE:
    continue  # behavior doesn't cover trial window
```

iii. The CONVERSION_NOTES report 21 sessions skipped (all with 0 valid trials after filtering), and 4 subjects lost because all their sessions were skipped.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the timing information in the AI's output, spike binning is the most expensive step (~0.4s/session after optimization), followed by behavior loading/interpolation. The CONVERSION_NOTES estimate ~1.9s total per session, with the full conversion taking ~36 minutes for 459 sessions.

ii.
```python
t_spike = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike_done = time.time()
# ...
print(f"  spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s")
```

iii. The CONVERSION_NOTES note that spike binning was optimized from 1.6s to 0.4s per session using np.bincount with linear indexing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes_fast` function loops over trials. The `interpolate_behavior` function loops over trials. The `compute_trial_number_in_block` function loops over all trials one by one. The per-trial formatting loop at the end of `process_session` that creates input/output arrays could also be vectorized.

ii.
```python
# Trial loop in bin_spikes_fast
for i, trial_idx in enumerate(valid_idx):
    # ... per-trial spike binning

# Trial loop in interpolate_behavior
for trial_idx in range(n_trials):
    # ... per-trial interpolation

# Per-element loop in compute_trial_number_in_block
for i in range(len(prob_left)):
    if prob_left[i] != current_val:
        # ...

# Per-trial formatting loop
for trial_idx in range(n_trials):
    neural_trial = neural_data[trial_idx].astype(np.uint8)
    # ...
```

iii. The AI optimized spike binning with np.bincount but left other loops unvectorized.

## 10-c. What processing does the code repeat multiple times?

i. The `BrainRegions()` object is instantiated once per session in `process_session()` rather than being shared. The file revision search logic (`find_file`) is called independently for each file type.

ii.
```python
def process_session(session_info, show_processing=False):
    # ...
    br = BrainRegions()   # instantiated per session
```

iii. The BrainRegions instantiation per session is redundant since it's a static lookup table.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and bins spikes for ALL clusters (including low-quality and void clusters), resulting in ~596k neurons vs ~76k well-isolated neurons. This means roughly 87% of the computed neural data is from low-quality clusters that could have been filtered out. The code also computes brain region mappings for all clusters including void ones.

ii.
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1
# All clusters used - no quality filtering
binned_spikes = bin_spikes_fast(..., n_clusters_total=n_clusters)
```

iii. The AI's decision to include all clusters means substantially more computation than necessary if a quality filter were applied.
