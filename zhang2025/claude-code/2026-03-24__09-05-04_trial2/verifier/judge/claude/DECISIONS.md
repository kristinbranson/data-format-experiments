# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the local filesystem under `data/one_cache/` using glob patterns to find session directories matching the pattern `*/Subjects/*/*/001`. It validates each session has spike data, trial data, wheel data, and motion energy data before including it. Sessions are processed sequentially in a loop, each producing a result dict that gets appended to lists.

ii.
```python
def find_session_dirs():
    """Find all session directories with required data."""
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        has_me = (len(glob.glob(os.path.join(sdir, 'alf/*/leftCamera.ROIMotionEnergy.npy'))) > 0 or
                  len(glob.glob(os.path.join(sdir, 'alf/*/rightCamera.ROIMotionEnergy.npy'))) > 0)
        # ...
        if has_spikes and has_trials and has_wheel and has_me:
            valid.append(sdir)
    return valid
```

iii. The AI noted that the reference code uses ONE API with `bwm_df` to list sessions by eid, but since the data is available locally on disk, it scans the filesystem directly. The AI documented finding 393 sessions with complete data on disk, compared to 433 sessions mentioned in the methods paper.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the directory path structure. The path format is `data/one_cache/<lab>/Subjects/<subject>/<date>/001`. The subject name is the directory name after `Subjects/`. A list of unique subjects is maintained and each session is assigned a subject index.

ii.
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    subject = parts[sub_idx + 1]
    return lab, subject, date

# In main():
if subject not in all_subjects:
    all_subjects.append(subject)
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. The AI used the filesystem directory structure to identify subjects, which mirrors how the IBL data is organized on disk.

## 1-c. How are the data split into sessions?

i. Each directory at the `<subject>/<date>/001` level represents one session. Each session is processed independently by `process_session()`, producing separate neural, input, and output arrays that are stored as separate list elements.

ii.
```python
for i, sdir in enumerate(session_dirs):
    result = process_session(sdir, br, ...)
    if result is not None:
        neural.append(result['neural'])
        input_data.append(result['input'])
        output_data.append(result['output'])
```

iii. The AI followed the standard IBL data organization where each session is identified by subject+date, matching the reference code's per-eid processing loop.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file for each session. Each row is one trial. After filtering, valid trials are indexed and per-trial data arrays are created.

ii.
```python
def load_trials(sdir):
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    trials = pd.read_parquet(trial_files[-1])
    return trials

# In process_session():
good_indices = np.where(combined_mask)[0]
neural_trials = binned_spikes[good_indices]
trials_good = trials.iloc[good_indices]
```

iii. The AI loads the parquet trials table, which contains one row per trial with all trial metadata. This matches the reference code's approach through `SessionLoader.load_trials()`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) reaction time between 0.08 and 2.0 seconds, (2) trial length from goCue to feedback <= 10 seconds, (3) exclude no-choice trials (choice==0), (4) exclude trials with NaN in key columns. Additionally, behavior availability masks (wheel and whisker) are combined with the trial quality mask.

ii.
```python
def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    if 'goCue_times' in trials.columns and 'feedback_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    mask &= (trials['choice'] != 0)
    for col in NAN_EXCLUDE:
        if col in trials.columns:
            mask &= ~trials[col].isna()
    return mask

# Combined with behavior masks:
combined_mask = mask.values & wheel_mask & whisker_mask
```

iii. The AI documented matching the reference code's `load_trials_and_mask` function with the same parameters: min_rt=0.08, max_rt=2.0, max_trial_len=10.0, NaN exclusion list, and no-choice exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from each probe's pykilosort directory, along with brain region mapping from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
def load_spikes(sdir):
    probe_dirs = sorted(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*')))
    for pdir in probe_dirs:
        spike_times = np.load(os.path.join(pdir, 'spikes.times.npy')).flatten()
        spike_clusters = np.load(os.path.join(pdir, 'spikes.clusters.npy')).flatten()
        cluster_channels = np.load(os.path.join(pdir, 'clusters.channels.npy')).flatten()
        channel_brain_ids = np.load(os.path.join(pdir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. The AI identified the same source variables as the reference code (spike times and cluster IDs), loading them directly from numpy files rather than through the SpikeSortingLoader API.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (with cluster ID offsets), sorted by time, and then binned into 20ms time bins within each trial's time window. The binning uses `searchsorted` for trial-wise spike selection and flat-index bincount for fast counting. The result is stored as uint8 (clipped to 255).

ii.
```python
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_begs, interval_ends):
    binned = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    for trial_idx in range(n_trials):
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        bin_idx = np.minimum(((times_trial - t_beg) / BINSIZE).astype(np.int32), N_BINS - 1)
        flat_idx = clusters_trial * N_BINS + bin_idx
        counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
        binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)
    return binned

# Stored as uint8:
neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. The AI documented using searchsorted+bincount as a vectorized alternative to the reference code's bincount2D with multiprocessing. The uint8 storage was a memory optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron quality filtering is applied. All neurons from all probes are included regardless of quality metrics.

ii.
```python
# No QC filtering in load_spikes - all clusters are included
# From docstring: "Following reference code: no QC filtering (qc=None)."
```

iii. The AI noted that the reference code's `prepare_data` calls `load_spiking_data` with default `qc=None`, meaning all neurons are used. The CONVERSION_NOTES.md explicitly states: "Use ALL neurons (qc=None), matching reference code."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `stimOn_times` (stimulus onset). For each trial, the time window is [stimOn_times - 0.5, stimOn_times + 1.5], giving a 2-second window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

# In process_session():
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The AI followed both the reference code (which uses `stimOn_times` with `(-0.5, 1.5)` window) and the decoder task instructions (which say "Temporally align based on stimulus onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02 seconds), resulting in 100 bins per 2-second trial. No temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The AI matched the reference code's 20ms bin size. The methods paper mentions 50ms bins for choice/prior decoding and 20ms for wheel/whisker, but the reference code uniformly uses 20ms, and the AI followed the code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from raw data variables per se. It is a computed time axis based on the alignment window parameters (TIME_WINDOW and BINSIZE).

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The AI constructed a linearly spaced time vector representing the center of each 20ms bin, from -0.49s to 1.49s (100 values). This represents the time relative to stimulus onset for each time bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linspace creates 100 evenly spaced values from -0.49s (center of first bin: -0.5 + 0.01) to 1.49s (center of last bin: 1.5 - 0.01). This time vector is the same for all trials.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,  # -0.5 + 0.01 = -0.49
    TIME_WINDOW[1] - BINSIZE / 2,  # 1.5 - 0.01 = 1.49
    N_BINS  # 100
).astype(np.float32)
```

iii. The AI chose bin centers as the time representation. This is a reasonable choice for representing the time of each neural bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is inherently aligned with neural data since both use the same time window (-0.5 to 1.5s relative to stimOn_times) and the same number of bins (100). The time input provides the temporal coordinate for each neural time bin.

ii.
```python
# Same time_input vector used for all trials:
inp = np.stack([
    time_input,
    np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
], axis=0)  # (2, 100)
```

iii. The alignment is implicit through the shared binning structure.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

iii. The AI identified that blocks are defined by changes in `probabilityLeft`, which represents the prior probability of the stimulus appearing on the left.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code iterates through trials, incrementing a counter when `probabilityLeft` stays the same, and resetting to 1 when it changes. This counts the position of each trial within its probability block.

ii.
```python
def compute_trial_num_in_block(prob_left):
    trial_nums = np.ones(len(prob_left), dtype=np.int32)
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            trial_nums[i] = trial_nums[i - 1] + 1
        else:
            trial_nums[i] = 1
    return trial_nums
```

iii. The AI noted this is computed from `probabilityLeft` changes, consistent with the IBL task structure where blocks are defined by changes in stimulus probability.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The AI identified `choice` as the source variable from the trials table, matching the reference code.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are mapped from the IBL convention (-1=left, 1=right, 0=no-choice) to binary (0=left, 1=right). No-choice trials are excluded during trial filtering. The choice value is broadcast to all 100 time bins (per-trial but stored as time-varying).

ii.
```python
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
# In output construction:
np.full(N_BINS, choice_binary[i], dtype=np.int64)
```

iii. The AI followed the decoder task instructions: "Choice, binary, per-trial, left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
prior = np.full(len(prob_left), 1, dtype=np.int32)  # default 0.5->1
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The AI mapped probabilityLeft values to the categorical encoding specified in the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to categorical integers: 0.2→0, 0.5→1, 0.8→2, using `np.isclose` with tolerance 0.05. The prior value is broadcast to all 100 time bins per trial.

ii.
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
# In output:
np.full(N_BINS, prior[i], dtype=np.int64)
```

iii. The AI followed the decoder task instructions: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel_speed(sdir):
    wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
    wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The AI identified the wheel position and timestamp files as the source data, matching the reference code's approach through SessionLoader.load_wheel().

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to a uniform 1kHz sampling rate, velocity is computed via finite differences (np.gradient), and speed is the absolute value of velocity.

ii.
```python
dt = 0.001  # 1kHz
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
```

iii. The AI noted it was replicating the SessionLoader.load_wheel() processing. However, the reference code uses Gaussian smoothing for velocity computation, while the AI uses simple finite differences (np.gradient).

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using quantile-based thresholds (terciles). Non-NaN values across all trials in a session are used to compute the 33rd and 67th percentile boundaries, then values are assigned to bins 0, 1, or 2 using `np.digitize`.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [33.33, 66.67]
    boundaries = np.percentile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int32)
    return result

wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
```

iii. The AI chose session-wide tercile-based discretization, producing balanced categories (~33% each).

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same trial time bins as neural data. The interpolation points match the reference code: `linspace(interval_beg + binsize, interval_end, n_bins)`. The same time window (-0.5 to 1.5s relative to stimOn_times) is used for both neural and wheel data.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_vals, interval_begs, interval_ends):
    x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
    interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
    values[trial_idx] = interp_func(x_interp).astype(np.float32)

# Called with same intervals as neural:
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
```

iii. The AI followed the reference code's `get_behavior_per_interval` interpolation logic, using the same interpolation points and the same trial time windows as neural data.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and corresponding camera timestamp files (`_ibl_leftCamera.times.npy`).

ii.
```python
def load_whisker_me(sdir):
    left_me_files, left_time_files = _find_files(
        sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
    if left_me_files and left_time_files:
        me = np.load(left_me_files[-1]).flatten()
        times = np.load(left_time_files[-1]).flatten()
        return times[:min_len], me[:min_len]
    # Fall back to right camera
    # ...
```

iii. The AI's left-first-then-right fallback strategy matches the reference code's `bin_behaviors` function which tries `left-whisker-motion-energy` first and falls back to `right-whisker-motion-energy`.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional processing) and interpolated to trial time bins. The only processing is the temporal interpolation to match neural data bins.

ii.
```python
me_times, me_vals_raw = load_whisker_me(sdir)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. The AI loads the pre-computed motion energy from the camera files, matching the reference code which loads via `SessionLoader.load_motion_energy()`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same quantile-based discretization as wheel speed: terciles computed per session, 3 bins (0, 1, 2).

ii.
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. The AI used the same discretization approach as wheel speed, consistent with the decoder task instructions specifying "3 bins" for both.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation approach as wheel speed: interpolated to the same trial time bins using the same time window (-0.5 to 1.5s relative to stimOn_times).

ii.
```python
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. Aligned to stimOn_times with (-0.5, 1.5) window, matching neural data alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Sessions missing required data files (spikes, trials, wheel, ME) are skipped entirely during discovery. (2) Sessions that fail processing raise exceptions caught by try/except, logged, and skipped. (3) Trials with NaN alignment times are skipped during spike binning. (4) Behavior interpolation marks trials without sufficient data coverage as bad (mask=False). (5) Combined masks ensure only trials with valid neural AND behavioral data are included. (6) Sessions with fewer than 2 valid trials are skipped. (7) Camera times/ME length mismatches handled with `min(len(me), len(times))`.

ii.
```python
# Session-level: skip if missing data
if has_spikes and has_trials and has_wheel and has_me:
    valid.append(sdir)

# Trial-level: NaN handling in binning
if np.isnan(t_beg) or np.isnan(t_end):
    continue

# Behavior coverage check
if np.abs(t_beg - beh_t[0]) > BINSIZE:
    mask[trial_idx] = False
if np.abs(t_end - beh_t[-1]) > BINSIZE:
    mask[trial_idx] = False

# Combined mask
combined_mask = mask.values & wheel_mask & whisker_mask
```

iii. The AI documented handling 57 sessions that were missing wheel data (PL050/hausserlab sessions) and 1 session with too few valid trials.

## 12-a. What are the most time-consuming steps of the code?

i. Spike binning is the primary bottleneck, processing each trial in a loop with searchsorted and bincount operations. Wheel speed and whisker ME loading and interpolation are the next most time-consuming. The AI reported ~0.2-0.3s per session for spike binning and ~0.5-1s for behavior processing.

ii.
```python
# Spike binning loop over all trials:
for trial_idx in range(n_trials):
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    # ... binning logic
```

iii. The AI documented timing information in CONVERSION_NOTES.md showing ~2.5s per session total, with ~16 minutes estimated for all sessions.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loops that could be vectorized are: (1) the trial-level spike binning loop in `bin_spikes_vectorized` which processes one trial at a time, (2) the trial-level behavior interpolation loop in `interpolate_behavior_to_bins`, and (3) the `compute_trial_num_in_block` loop.

ii.
```python
# Trial loop in spike binning:
for trial_idx in range(n_trials):
    # Could potentially batch multiple trials

# Trial loop in behavior interpolation:
for trial_idx in range(n_trials):
    # Each trial interpolated independently

# Block computation loop:
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        trial_nums[i] = trial_nums[i - 1] + 1
```

iii. The AI noted it used searchsorted for efficient spike selection within the loop, which is partially vectorized. The reference code used multiprocessing instead of vectorization for these operations.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats wheel and whisker ME loading/interpolation using the same `interpolate_behavior_to_bins` function with the same structure. The trial mask creation and the behavior interpolation share similar interval construction logic. Brain region mapping (id2acronym, acronym2acronym) is done per-session rather than cached.

ii.
```python
# Same function called for both behaviors:
wheel_vals, wheel_mask = interpolate_behavior_to_bins(wh_times, wh_speed, interval_begs, interval_ends)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(me_times, me_vals_raw, interval_begs, interval_ends)

# BrainRegions() instantiated per session in load_spikes AND once in main:
br = BrainRegions()  # in load_spikes
br = BrainRegions()  # in main
```

iii. The repeated BrainRegions instantiation is a minor inefficiency noted by the AI, who instantiated it once in main and passed it to process_session, though load_spikes still creates its own instance.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Spike counts are first computed as float32, then clipped and stored as uint8 for memory, but the downstream decoder converts back to float32. (2) All spikes are binned for ALL trials before the trial mask is applied, meaning binning is done for trials that are later discarded. (3) The code computes interval_begs/interval_ends for all trials including filtered-out ones. (4) Brain region mapping is done for all clusters including those in 'root' and 'void' regions.

ii.
```python
# Binning ALL trials before filtering:
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)
# Then filtering:
neural_trials = binned_spikes[good_indices]

# uint8 conversion that gets undone:
neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. The AI documented the uint8 storage as a deliberate memory optimization, noting the decoder converts to float32 when loading. Binning all trials before filtering was noted as following the reference code's approach.
