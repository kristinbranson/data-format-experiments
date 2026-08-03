# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly scanning the filesystem for session directories matching the pattern `data/one_cache/*/Subjects/*/*/001`. It uses `glob.glob` to find directories containing the required files (spikes, trials, wheel, motion energy). It does NOT use the ONE API or the Brainwidemap release tag to discover sessions. Instead, it manually navigates the directory structure and loads files with `np.load` and `pd.read_parquet`.

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
```

iii. The AI chose to directly scan the filesystem rather than using the ONE API. The CONVERSION_NOTES state: "Processes sessions from `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`". The AI found 393 sessions with complete data on disk.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject name from the directory path by parsing the path components relative to the "Subjects" directory. Subjects are collected in encounter order (not sorted) as sessions are processed.

ii.
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    subject = parts[sub_idx + 1]
    return lab, subject, date
```

```python
if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
```

iii. The directory structure organizes data by `<lab>/Subjects/<subject>/<date>/001/`, so the subject name is extracted from the path.

## 1-c. How are the data split into sessions?

i. Each directory at the path `<lab>/Subjects/<subject>/<date>/001/` represents one session. The session identifier is constructed as `<subject>_<date>`.

ii.
```python
session_id = f"{subject}_{date}"
```

iii. Sessions correspond to unique directory entries on disk. No splitting is required.

## 1-d. How are the data split into trials?

i. Trials correspond to rows in the `_ibl_trials.table.pqt` parquet file. Each row is one trial.

ii.
```python
def load_trials(sdir):
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    trials = pd.read_parquet(trial_files[-1])
    return trials
```

iii. The trials table naturally has one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) reaction time between 0.08 and 2.0 seconds, (2) exclude no-choice trials (choice == 0), (3) exclude trials with NaN in key columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (4) trial length (goCue to feedback) <= 10 seconds, (5) behavioral coverage masks for wheel and whisker. This is broader than the reference, which does not filter on trial length or NaN in feedback_times/feedbackType.

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
```

iii. The AI's CONVERSION_NOTES say: "Trial filtering: RT 0.08-2.0s, exclude no-choice, NaN exclusion" and references the `load_trials_and_mask` function from the reference code. The AI also adds a trial length filter (`MAX_TRIAL_LEN = 10.0`) and NaN checks on `feedback_times` and `feedbackType`, which go beyond the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` loaded from each probe directory. Brain region mapping uses `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
cluster_channels = np.load(cc_file).flatten()
channel_brain_ids = np.load(cb_file).flatten()
```

iii. These are the standard IBL spike sorting outputs.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over a 2s trial window (-0.5 to 1.5s relative to stimulus onset), producing spike counts per cluster per bin. The counts are stored as uint8 (clipped to 255), NOT converted to firing rates (Hz). The reference divides by the bin width to produce Hz. When a session has multiple probes, clusters are merged with offset cluster IDs.

ii.
```python
bin_idx = np.minimum(
    ((times_trial - t_beg) / BINSIZE).astype(np.int32),
    N_BINS - 1
)
flat_idx = clusters_trial * N_BINS + bin_idx
counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)
```

```python
neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. The AI stores raw spike counts as uint8 to save memory (21 GB vs potentially 84 GB for float32). The CONVERSION_NOTES state: "Neural data stored as uint8 (spike counts rarely exceed 255) to reduce memory." This differs from the reference which converts to firing rates (Hz) by dividing by bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT apply any quality control filtering on neurons. ALL clusters are used regardless of their quality label. The reference filters to keep only clusters with `label >= 1` (well-isolated neurons).

ii.
```python
def load_spikes(sdir):
    # No QC filtering - ALL clusters used
    spike_times = np.load(st_file).flatten()
    spike_clusters = np.load(sc_file).flatten()
    # ... no label filtering
```

iii. The AI's CONVERSION_NOTES explicitly state: "No QC filtering applied in `prepare_data` (qc=None by default). ALL neurons used, not just good ones (label >= 1 not required)." The AI justified this by citing the reference method paper's code (`0_data_caching.py`) which uses `qc=None`. However, the data paper describes `label >= 1` as the standard quality control, and the reference solution applies this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to stimulus onset (`stimOn_times`). The trial window is computed as `stimOn_times + TIME_WINDOW[0]` to `stimOn_times + TIME_WINDOW[1]`, i.e., -0.5s to +1.5s relative to stimulus onset.

ii.
```python
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. Aligns to stimOn_times as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial (2s window). No rebinning is applied. This matches the reference.

ii.
```python
BINSIZE = 0.02  # 20ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize: 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the bin parameters (window start, bin size, number of bins), not from any raw data variable. It represents the center of each time bin.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The time input is a synthetic variable defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Computed as evenly spaced bin centers from -0.49 to 1.49 using `np.linspace`. The reference uses `EDGES[:-1] + BIN/2` which produces the same values.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. This is equivalent to the reference approach. Both produce 100 bin centers from -0.49 to 1.49.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input represents the centers of the same bins used for spike binning, so it is inherently aligned with the neural data.

ii. The same `BINSIZE` and `TIME_WINDOW` constants are used for both neural binning and time input generation.

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. A change in `probabilityLeft` marks a new block boundary.

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

iii. Blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts the trial's position within a block starting from 1 (the first trial in a block is 1). The reference uses `cumcount()` which starts from 0. The AI also computes this on the filtered trials only (after masking), whereas the reference computes it on ALL trials before filtering, preserving the animal's true position in the block.

ii.
```python
# AI code: computed on filtered trials only
trials_good = trials.iloc[good_indices]
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

Reference code:
```python
# Reference: computed on all trials before filtering
block = (trials.probabilityLeft != trials.probabilityLeft.shift()).cumsum()
trial_variables = pd.DataFrame({...
    'trial_in_block': trials.groupby(block).cumcount()})
variables = variables[keep]  # filtered after
```

iii. The AI's approach starts from 1 instead of 0 and computes block position on filtered trials. This means if trial 3 of a block is dropped, the AI would renumber subsequent trials, changing their block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The choice column from the IBL trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == 1` to 1 and everything else (including -1) to 0. In IBL convention, `choice = 1` means LEFT and `choice = -1` means RIGHT. The instructions say "left = 0, right = 1". So the AI maps LEFT -> 1 and RIGHT -> 0, which is INVERTED from the instructions. The reference correctly maps `+1 -> 0` (left=0) and `-1 -> 1` (right=1).

ii.
```python
# AI: choice==1 (LEFT in IBL) -> 1, choice==-1 (RIGHT in IBL) -> 0
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

Reference:
```python
# Reference: +1 (left) -> 0, -1 (right) -> 1
CHOICE = {1.0: 0, -1.0: 1}
```

iii. The AI comment says "Choice: -1 (left) -> 0, 1 (right) -> 1" but this is wrong about the IBL convention. In IBL, choice=1 is left and choice=-1 is right. The code maps left to 1 and right to 0, inverting the instruction specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The `probabilityLeft` column holds the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI uses `np.isclose` with `atol=0.05` to map values to categories: 0.2->0, 0.5->1, 0.8->2. The default is 1 (0.5). The reference uses an exact dictionary mapping: `{0.2: 0, 0.5: 1, 0.8: 2}`. The AI's approach is more tolerant of floating-point imprecision but could potentially miscategorize values near boundaries.

ii.
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. Uses approximate matching rather than exact values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The raw wheel position and timestamps.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates position to 1kHz, computes velocity via `np.gradient` (finite differences), and takes absolute value. The reference uses `SessionLoader.load_wheel()` which uses `brainbox.behavior.wheel.velocity_filtered` — a 3rd-order Butterworth low-pass filter at 20 Hz — producing a much smoother velocity trace. The AI then interpolates to trial bins using `interp1d` with `linspace(t_beg + BINSIZE, t_end, N_BINS)` as interpolation points, while the reference uses `np.interp` at bin centers (`onset + TIME`).

ii.
```python
# AI: finite differences velocity
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
```

```python
# AI: different interpolation points
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
values[trial_idx] = interp_func(x_interp)
```

iii. The AI's CONVERSION_NOTES say "reference code uses SessionLoader.load_wheel() which interpolates to 1kHz and computes velocity via Gaussian smoothing." This is actually incorrect — the SessionLoader uses a Butterworth filter, not Gaussian smoothing. The AI's implementation uses finite differences instead.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using session-wide tercile percentiles (33rd and 67th percentiles), matching the reference approach.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [33.33, 66.67]
    boundaries = np.percentile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int32)
    return result
```

iii. Same tercile approach as reference: `np.digitize` with percentile boundaries.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to trial time bins using `np.linspace(t_beg + BINSIZE, t_end, N_BINS)` where `t_beg = stimOn + TIME_WINDOW[0]` and `t_end = stimOn + TIME_WINDOW[1]`. This produces interpolation points offset by one bin width from the reference's bin centers. The reference evaluates at `onset + TIME` where `TIME = EDGES[:-1] + BIN/2`, which are the true bin centers.

ii.
```python
# AI interpolation points: linspace(beg + 0.02, end, 100)
# = linspace(stimOn - 0.48, stimOn + 1.5, 100)
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

```python
# Reference bin centers: stimOn + [-0.49, -0.47, ..., 1.49]
TIME = EDGES[:-1] + BIN / 2
traces.append(np.interp(onset + TIME, ...))
```

iii. The AI's interpolation grid is shifted ~10ms later than the reference's bin centers. The first point is at `stimOn - 0.48` vs reference `stimOn - 0.49`, and the last is at `stimOn + 1.5` vs reference `stimOn + 1.49`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` with corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
left_me_files, left_time_files = _find_files(
    sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
if left_me_files and left_time_files:
    me = np.load(left_me_files[-1]).flatten()
    times = np.load(left_time_files[-1]).flatten()
```

iii. Left camera preferred, right as fallback, matching reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is loaded and interpolated to trial time bins using the same `interp1d` approach as the wheel speed. No additional filtering or normalization. Same interpolation grid offset issue as wheel speed.

ii.
```python
me_times, me_vals_raw = load_whisker_me(sdir)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. Uses the same behavioral interpolation function as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 bins using session-wide tercile percentiles.

ii.
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation approach as wheel speed, with the same ~10ms offset from the reference bin centers.

ii. Uses the same `interpolate_behavior_to_bins` function.

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by: (1) skipping sessions that lack required files (spikes, trials, wheel, motion energy), (2) skipping sessions with fewer than 2 valid trials, (3) masking out trials where behavioral coverage is insufficient, (4) excluding NaN values in key trial columns, (5) clipping channel indices to valid range, (6) truncating motion energy and time arrays to the shorter length. Failed sessions are caught with a try/except and skipped.

ii.
```python
# Skip missing files
if not all(os.path.exists(f) for f in [st_file, sc_file, cc_file, cb_file]):
    continue

# Handle length mismatch in motion energy
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]

# Skip sessions with too few trials
if len(good_indices) < 2:
    return None
```

iii. The AI's approach is generally robust, catching exceptions and skipping problematic sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk (`np.load` for spike times and clusters) and spike binning are the most time-consuming steps. The AI processes sessions sequentially (no parallel processing), which makes the overall runtime slower.

ii.
```python
spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
```

```python
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)
```

iii. The CONVERSION_NOTES estimate ~2.5s per session, ~16 minutes total.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning function loops over trials individually, which could be vectorized by offsetting spike indices globally. The behavioral interpolation also loops per trial. The trial number in block computation uses a Python for loop.

ii.
```python
# Trial loop in spike binning
for trial_idx in range(n_trials):
    ...

# Trial loop in behavior interpolation
for trial_idx in range(n_trials):
    ...

# Python loop for trial number
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        trial_nums[i] = trial_nums[i - 1] + 1
```

iii. The per-trial loops are similar to the reference code's structure.

## 10-c. What processing does the code repeat multiple times?

i. The behavioral interpolation function (`interpolate_behavior_to_bins`) is called twice with the same `interval_begs` and `interval_ends` — once for wheel speed and once for whisker motion energy. The `searchsorted` calls inside could share computation. Also, `np.searchsorted` on spike times is done separately in `bin_spikes_vectorized` despite the same sorted array.

ii.
```python
wheel_vals, wheel_mask = interpolate_behavior_to_bins(wh_times, wh_speed, interval_begs, interval_ends)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(me_times, me_vals_raw, interval_begs, interval_ends)
```

iii. The repeated coverage checking and searchsorted calls add minor overhead.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins ALL neurons (including poor quality ones) which results in a massive 21 GB file, then requires a separate `reduce_data.py` script to subsample neurons to max 500 per session because the full data exceeds memory during training. The reference filters to good neurons upfront (~142/session median), avoiding this issue entirely. Additionally, the AI bins spikes for ALL trials before applying the trial mask, doing unnecessary computation for trials that will be discarded.

ii.
```python
# Bins spikes for ALL trials first, then filters
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)
# ... later:
neural_trials = binned_spikes[good_indices]
```

iii. The CONVERSION_NOTES document the memory issue: "The 21 GB uint8 pickle converts to ~84 GB float32... Solution: created `reduce_data.py` to subsample neurons to max 500 per session."
