# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the local disk cache at `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/` using glob patterns to find session directories with all required files (spikes, trials, wheel, whisker motion energy). It does not use the ONE API or a `bwm_release.csv` freeze file to identify sessions; instead it discovers sessions by filesystem traversal. Each session's data is loaded from parquet and numpy files directly.

ii.
```python
DATA_ROOT = Path('data/one_cache')

def find_session_dirs():
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        has_me = (len(glob.glob(os.path.join(sdir, 'alf/*/leftCamera.ROIMotionEnergy.npy'))) > 0 or
                  len(glob.glob(os.path.join(sdir, 'alf/*/rightCamera.ROIMotionEnergy.npy'))) > 0)
        ...
        if has_spikes and has_trials and has_wheel and has_me:
            valid.append(sdir)
    return valid
```

iii. The AI justified this approach because the ONE API was not available offline — the data was pre-cached on disk. The reference code used `ONE()` with `bwm_df` from `bwm_release.csv` to list sessions and then loaded via `prepare_data`, `SpikeSortingLoader`, and `SessionLoader`. The AI replicated the loading by directly reading the numpy/parquet files that those loaders would produce.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is parsed from the directory path (`<lab>/Subjects/<subject>/<date>/001`). A list of unique subjects is built as sessions are processed. `subject_idx` maps each session to its subject.

ii.
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    subject = parts[sub_idx + 1]
    ...
    return lab, subject, date

# In main():
if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. The subject name is embedded in the IBL data path convention. The AI correctly extracted it from the path structure.

## 1-c. How are the data split into sessions?

i. Each directory under `<subject>/<date>/001/` represents one session. The AI processes each valid session directory independently, producing one entry per session in the `neural`, `input`, and `output` lists.

ii.
```python
for i, sdir in enumerate(session_dirs):
    lab, subject, date = parse_session_info(sdir)
    result = process_session(sdir, br, ...)
    if result is not None:
        neural.append(result['neural'])
        input_data.append(result['input'])
        output_data.append(result['output'])
```

iii. This matches the reference code's per-eid processing loop in `0_data_caching.py`.

## 1-d. How are the data split into trials?

i. Trials are loaded from the parquet file `_ibl_trials.table.pqt`. Each row is one trial. After applying trial quality filters, the remaining valid trials each become one element in the session's trial list. Neural data is binned per-trial using the stimulus onset time to define trial boundaries.

ii.
```python
def load_trials(sdir):
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    trials = pd.read_parquet(trial_files[-1])
    return trials

# Trial intervals aligned to stimulus onset:
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]  # stimOn - 0.5s
interval_ends = stim_on + TIME_WINDOW[1]  # stimOn + 1.5s
```

iii. The reference code similarly loads trials via `SessionLoader.load_trials()` and defines trial intervals relative to an alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) reaction time between 0.08-2.0s, (2) no-choice exclusion (choice == 0), (3) NaN exclusion on key events, (4) trial length <= 10s, and (5) behavior data availability (wheel and whisker masks). This is combined into a single mask.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType']

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

# Combined with behavior availability:
combined_mask = mask.values & wheel_mask & whisker_mask
```

iii. The AI noted this matches `load_trials_and_mask` from the reference code with parameters `min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True, nan_exclude='default'`. The reference code also combines with behavior masks in `align_spike_behavior`. One difference: the reference `load_trials_and_mask` uses strict `<` for min_rt and strict `>` for max_rt, while the AI uses `>=` and `<=`. Additionally, the reference code applies `mask` before `bin_behaviors` (line 76-77 in 0_data_caching.py: `trials_df=trials_data['trials_df']` without mask, but then `align_spike_behavior` applies `trials_mask`), whereas the AI applies the mask after binning all trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files from each probe's pykilosort directory.

ii.
```python
def load_spikes(sdir):
    probe_dirs = sorted(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*')))
    for pdir in probe_dirs:
        spike_times = np.load(os.path.join(pdir, 'spikes.times.npy')).flatten()
        spike_clusters = np.load(os.path.join(pdir, 'spikes.clusters.npy')).flatten()
        ...
```

iii. The reference code uses `SpikeSortingLoader.load_spike_sorting()` which loads the same underlying files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms bins within a 2s trial window (-0.5s to 1.5s relative to stimulus onset), producing spike counts of shape (n_clusters, 100) per trial. Multiple probes are merged by offsetting cluster IDs and sorting by time. The data is stored as uint8 (clipped to 255).

ii.
```python
BINSIZE = 0.02  # 20ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100

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

iii. The reference code uses `bincount2D` from `iblutil.numerical` for binning, which is functionally equivalent. The AI's approach is a custom vectorized implementation. The uint8 storage is a memory optimization not in the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to neurons. All clusters (including multi-unit activity) are used, matching the reference code's `qc=None` default.

ii.
```python
# From docstring and code: no QC filtering
def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.
    Following reference code: no QC filtering (qc=None)."""
    # No label filtering applied - all clusters used
```

iii. The AI noted that the reference code's `prepare_data` calls `load_spiking_data(one, pid, eid=eid, pname=probe_name)` without specifying `qc`, so `qc=None` (default), meaning all neurons are used. The methods paper describes quality filtering for "well-isolated neurons" but the reference code does not apply this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Each trial's time window is `[stimOn - 0.5s, stimOn + 1.5s]`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. This matches the reference code's `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` and the decoder task instruction "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms, producing 100 bins per 2s trial. No rebinning is applied — spikes are binned directly into 20ms bins.

ii.
```python
BINSIZE = 0.02  # 20ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The reference code uses `'binsize': 0.02`. The methods paper describes 50ms bins for choice/prior and 20ms bins for wheel/whisker, but the reference code uniformly uses 20ms. The AI followed the code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from any raw data variable. It is computed analytically as the center of each time bin within the trial window.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The time values represent bin centers from -0.49s to 1.49s. This is a synthetic variable derived from the alignment parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced array of 100 values from -0.49 to 1.49 (bin centers) is generated. This is identical for every trial.

ii. Same as 3-a.

iii. The decoder task specifies "Time since stimulus onset, continuous, time-varying" as a decoder input.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Since trials are aligned to stimulus onset and the time input represents time relative to stimulus onset, they are inherently aligned — both use the same 100-bin temporal grid. Time=0 corresponds to stimulus onset.

ii.
```python
# Same time grid: both neural bins and time_input span TIME_WINDOW with BINSIZE steps
time_input = np.linspace(TIME_WINDOW[0] + BINSIZE/2, TIME_WINDOW[1] - BINSIZE/2, N_BINS)
```

iii. No explicit alignment step needed because both are defined on the same temporal grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table.

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

# Called with:
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

iii. Block boundaries are inferred from changes in `probabilityLeft`. The AI noted this is not directly available in the reference code but is a reasonable derivation from block structure.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A sequential scan compares each trial's `probabilityLeft` with the previous trial. When the value changes, the counter resets to 1. The counter is computed on already-filtered trials (after applying the trial mask), so filtered-out trials can cause apparent block boundary changes.

ii. Same as 4-a.

iii. Computing trial number on filtered trials means that if trials are removed by quality filtering, block boundaries may appear where there are none (e.g., if a filtered trial was the last in a block). However, `probabilityLeft` is typically constant within a block, so filtering shouldn't cause spurious boundary changes often. Note: the computation is done on `trials_good` (filtered trials) rather than all trials and then selecting, which could affect the count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column of the trials table.

ii.
```python
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. In the IBL convention, `choice` is -1 (left) or 1 (right). No-choice trials (choice=0) are already excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values (-1 for left, 1 for right) are mapped to binary: left=0, right=1. This is broadcast to all 100 time bins as a per-trial constant.

ii.
```python
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
# Then in output construction:
np.full(N_BINS, choice_binary[i], dtype=np.int64)
```

iii. The decoder task specifies "Choice, binary, per-trial, left = 0, right = 1". The reference code stores choice as raw values (-1/1); the mapping to 0/1 is specific to the decoder format requirement.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
prior = np.full(len(prob_left), 1, dtype=np.int32)  # default 0.5->1
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The reference code stores `probabilityLeft` as `block` in `bin_behaviors`. The decoder task specifies the mapping: 0.2->0, 0.5->1, 0.8->2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. `probabilityLeft` values are mapped to categorical integers using approximate matching (atol=0.05): 0.2->0, 0.5->1, 0.8->2. The result is broadcast to all time bins.

ii. Same as 6-a, plus:
```python
np.full(N_BINS, prior[i], dtype=np.int64)
```

iii. The decoder task specifies "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel_speed(sdir):
    wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
    wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The reference code uses `SessionLoader.load_wheel()` which loads the same raw files but applies Gaussian smoothing to compute velocity. The AI instead uses `np.gradient` for finite differences.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is interpolated to 1kHz uniform sampling, velocity is computed via finite differences (`np.gradient`), and speed is taken as absolute value. This is then interpolated to the trial time bins.

ii.
```python
dt = 0.001  # 1kHz
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
```

iii. The reference code uses `SessionLoader.load_wheel()` which "contains wheel times and position interpolated to a uniform sampling rate, velocity and acceleration computed using Gaussian smoothing." The AI's approach uses simple finite differences instead of Gaussian smoothing, which will produce noisier velocity estimates.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using session-wide quantile boundaries (terciles).

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

iii. The decoder task specifies "Wheel speed discretized into 3 bins, time-varying." The AI chose session-wide terciles, which produces balanced bins (~33% each). The reference code does not discretize wheel speed (it uses continuous R^2 regression), so this is a format-specific decision.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset using the same trial windows as neural data: `[stimOn - 0.5s, stimOn + 1.5s]`. The continuous wheel speed signal is interpolated to the same 100 time bins.

ii.
```python
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)

# interpolation uses: x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. The methods paper states wheel speed should be aligned to `firstMovement_times` with window (0, 1s). However, the reference code (`0_data_caching.py`) aligns everything to `stimOn_times` with (-0.5, 1.5) window. The AI followed the code, which also matches the decoder task instruction to align to stimulus onset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding camera timestamps (`_ibl_leftCamera.times.npy`).

ii.
```python
def load_whisker_me(sdir):
    left_me_files, left_time_files = _find_files(
        sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
    if left_me_files and left_time_files:
        me = np.load(left_me_files[-1]).flatten()
        times = np.load(left_time_files[-1]).flatten()
        ...
    # Fall back to right camera
```

iii. The reference code uses `SessionLoader.load_motion_energy(views=['left'])` which loads the same underlying files. The left-first-then-right fallback matches `bin_behaviors` logic for `whisker-motion-energy`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional processing beyond what's pre-computed). They are then interpolated to the trial time bins using linear interpolation with extrapolation.

ii.
```python
me_times, me_vals_raw = load_whisker_me(sdir)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. Motion energy is pre-computed in the IBL pipeline. The reference code similarly loads it directly via `SessionLoader.load_motion_energy()`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 bins using session-wide quantile boundaries (terciles).

ii.
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. The decoder task specifies "Whisker motion energy discretized into 3 bins, time-varying."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same alignment as neural and wheel data: aligned to stimulus onset, window (-0.5, 1.5)s, interpolated to 100 bins.

ii.
```python
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. Same reasoning as wheel speed alignment (7-d).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions missing required data files (spikes, trials, wheel, whisker ME) are skipped entirely. (2) Trials with NaN in key event columns are masked out. (3) Trials where behavioral data doesn't cover the time window are masked out (coverage check). (4) If fewer than 2 valid trials remain, the session is skipped. (5) Cluster channel indices are clipped to valid range. (6) Camera times and ME arrays are truncated to the shorter length.

ii.
```python
# Coverage check in interpolate_behavior_to_bins:
if np.abs(t_beg - beh_t[0]) > BINSIZE:
    mask[trial_idx] = False
if np.abs(t_end - beh_t[-1]) > BINSIZE:
    mask[trial_idx] = False

# Clip channels:
valid_channels = np.clip(cluster_channels, 0, len(channel_brain_ids) - 1)

# Truncate to shorter:
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]

# Skip sessions with too few trials:
if len(good_indices) < 2:
    return None
```

iii. The reference code has similar handling: `get_behavior_per_interval` skips intervals with missing data, `align_spike_behavior` removes trials with None behavior data. Session-level errors are caught and skipped.

## 10-a. What are the most time-consuming steps of the code?

i. Spike binning is the most compute-intensive step per session (~0.2-0.3s per session). Wheel speed computation (interpolation to 1kHz then gradient) and behavioral interpolation also take time. File I/O (loading numpy files) adds overhead. The AI reported ~2.5s per session total, ~16 minutes for full dataset.

ii.
```python
# Timing in process_session:
t_bin = time.time()
binned_spikes = bin_spikes_vectorized(...)
print(f"  Spike binning: {time.time() - t_bin:.1f}s")
```

iii. The CONVERSION_NOTES.md reports spike binning at 0.2-0.3s per session and total ~2.5s per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning function has a per-trial loop that could be partially vectorized. The behavioral interpolation (`interpolate_behavior_to_bins`) also loops over trials. The `compute_trial_num_in_block` function uses a sequential loop. The reference code uses `multiprocessing.Pool` for both spike binning and behavior interpolation, while the AI uses single-threaded loops.

ii.
```python
# Per-trial loop in bin_spikes_vectorized:
for trial_idx in range(n_trials):
    ...

# Per-trial loop in interpolate_behavior_to_bins:
for trial_idx in range(n_trials):
    ...
```

iii. The AI used `np.searchsorted` and `np.bincount` within each trial iteration for efficiency but didn't parallelize across trials. The reference code uses `multiprocessing.Pool` for parallelism.

## 10-c. What processing does the code repeat multiple times?

i. The AI loads wheel data and whisker ME data separately but they go through the same `interpolate_behavior_to_bins` function. Brain region mapping (`BrainRegions()`) is initialized both in `load_spikes` (per session) and in `main()`. The `BrainRegions()` constructor is called inside `load_spikes` for every session.

ii.
```python
# BrainRegions created per session in load_spikes:
def load_spikes(sdir):
    br = BrainRegions()
    ...

# And also in main:
br = BrainRegions()
```

iii. Creating `BrainRegions()` involves loading atlas data from disk each time, which is wasteful. The `br` instance from `main()` is passed to `process_session` but not used by `load_spikes`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The neural data is stored as uint8, then later a separate `reduce_data.py` script subsamples neurons to max 500 per session because the decoder uses PCA to 100 components. The full neuron binning for all ~1400 neurons per session is partially wasted. (2) Spikes are binned for ALL trials (including those that will be filtered out), before the trial mask is applied. (3) The wheel position is interpolated to 1kHz (creating large arrays) just to compute velocity, when direct differentiation of the raw timestamps would suffice. (4) The code computes and stores spike counts but the decoder converts them to PCA components.

ii.
```python
# Binning ALL trials before filtering:
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends)
# Then filter:
neural_trials = binned_spikes[good_indices]

# Wheel: 1kHz interpolation
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)  # potentially millions of points
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
```

iii. The AI noted in CONVERSION_NOTES.md that spike binning was done before masking for consistency with the reference code approach, though it wastes computation on trials that are later discarded. The neuron reduction step was added as a post-hoc fix for memory issues.
