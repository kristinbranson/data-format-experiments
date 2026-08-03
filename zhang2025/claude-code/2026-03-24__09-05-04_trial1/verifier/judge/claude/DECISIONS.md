# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading a BWM release CSV file (`bwm_release.csv`) from the reference code directory to get a list of sessions. It then navigates the ONE cache directory structure directly (lab/Subjects/subject/date/number/alf/) to find and load files, rather than using the ONE API. Files are loaded with `np.load()` and `pd.read_parquet()` directly from the filesystem, with a custom `find_latest_revision()` function to handle revision directories.

ii.
```python
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)
```
```python
def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
```

iii. The AI chose to bypass the ONE API and load files directly from the cache filesystem because it works offline without needing a database connection. The session list is derived from the BWM release CSV file found in the reference code directory.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the BWM CSV file's `subject` column. Each row in the CSV has a subject field, and sessions are grouped by subject. At assembly, unique subjects are sorted and each session is assigned a `subject_idx`.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The subject identity comes directly from the BWM CSV metadata, no path parsing needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by the unique `eid` values in the BWM release CSV. Each unique eid corresponds to one session, and data files are located by constructing file paths from the lab, subject, date, and session number fields.

ii.
```python
sessions = bwm_df.groupby('eid').first().reset_index()
```

iii. The BWM CSV has one row per probe insertion, so grouping by eid gives one row per session.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) is loaded per session, with one row per trial. Trials are split by iterating over rows of this table.

ii.
```python
trials = pd.read_parquet(trials_file)
```

iii. The trials table is already structured with one row per trial, no splitting needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) exclude trials with NaN in key event columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (2) reaction time must be between 0.08s and 2.0s, (3) exclude no-choice trials (choice == 0), (4) max trial length filter (feedback_times - goCue_times <= 10s), and (5) a combined mask requiring valid wheel and whisker motion energy data for each trial's time window.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    mask &= (trials['choice'] != 0)
    if 'goCue_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    return mask
```

iii. The AI documented these filters as matching the reference code's trial mask criteria. The NaN exclusion list and RT bounds match the reference code's `load_trials_and_mask()`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from each probe's pykilosort directory. Channel-to-brain-region mapping comes from `channels.brainLocationIds_ccf_2017.npy` and `clusters.channels.npy`.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

iii. These are the standard spike sorting outputs from pykilosort.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms time windows aligned to stimulus onset, producing spike counts per cluster per bin. The binned data is stored as raw spike counts (uint8), NOT divided by the bin width to produce firing rates. Multi-probe sessions have their spikes merged with cluster ID offsets. The bin grid spans -0.5s to 1.5s (100 bins).

ii.
```python
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_starts, interval_ends, binsize, n_bins):
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    for trial_idx in range(n_trials):
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
        flat_idx = c_sel[valid] * n_bins + bin_idx[valid]
        np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
    return binned
```
```python
neural_trials.append(final_spikes[t].astype(np.uint8))  # (n_clusters, n_bins)
```

iii. The AI stored raw spike counts rather than firing rates (Hz) to save memory with uint8. The reference divides by bin width to get Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by quality. All clusters are included regardless of their quality label. The AI explicitly noted this decision: "Follow reference code: use all units (qc=None)" based on the reference Python code in `code_zhang2025` using `qc=None`.

ii.
```python
# No quality filtering - all clusters from all probes are used
n_clusters = len(clusters['channels'])
```

iii. The AI's CONVERSION_NOTES document this as: "Reference code uses qc=None: ALL clusters used (not filtered by quality label)". The AI followed the reference code's `qc=None` parameter rather than the data paper's well-isolated neuron criterion (label >= 1).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are binned relative to its stimulus onset time (`stimOn_times`). The interval for each trial runs from `stimOn_times + (-0.5)` to `stimOn_times + (1.5)`.

ii.
```python
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. This matches the instruction to align to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins (BINSIZE = 0.02), producing 100 time bins over the 2s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20 ms time bins
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's bin size of 20ms.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the time window parameters and bin size, not from any raw data variable directly. It is a synthetic time axis constructed from the window bounds.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. This creates a time axis representing time since stimulus onset for each bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time axis is computed as `np.linspace(-0.48, 1.5, 100)`, which produces evenly spaced values from -0.48 to 1.5. This differs from the reference which uses bin centers: `EDGES[:-1] + BIN/2`, producing values from -0.49 to 1.49.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# Produces: [-0.48, -0.46, ..., 1.48, 1.5]
```

Reference uses:
```python
TIME = EDGES[:-1] + BIN / 2  # bin centres
# Produces: [-0.49, -0.47, ..., 1.47, 1.49]
```

iii. The AI intended to compute bin centers but used a slightly different formula that produces values shifted by ~0.01s from true bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time grid is used for both neural binning and the time input, though the neural bins use `(t_sel - t_start) / binsize` for bin assignment while the time input uses `np.linspace`. The alignment is approximate but not exact due to the different grid computation.

ii.
```python
# Neural binning:
bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
# Time input:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. Both share 100 time bins over the same 2s window, but the time labels don't correspond exactly to bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

iii. The trials table has no explicit block identifier, so blocks must be inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A counter starts at 1 for the first trial of each block and increments by 1 for each subsequent trial. Block boundaries are detected where `probabilityLeft` changes. The count is computed on ALL trials before filtering, so filtered-out trials still advance the counter. The count starts from 1, not 0.

ii.
```python
def compute_trial_in_block(prob_left):
    trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
    return trial_in_block
```

iii. The AI computes trial number before filtering (preserving real position in block), but starts counting from 1 rather than 0 as the reference does with `cumcount()`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which takes values -1, 0, or +1 in the IBL convention.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. The choice column is a standard IBL trials table field.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice values using `(choice + 1) // 2`, with the comment "IBL: -1=left, 1=right -> convert to 0=left, 1=right". However, the IBL convention is actually +1=left, -1=right. This means the AI's mapping produces LEFT=1 and RIGHT=0, which is inverted from the instruction's "left = 0, right = 1".

ii.
```python
# IBL: -1=left, 1=right -> convert to 0=left, 1=right
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
```

The reference uses:
```python
CHOICE = {1.0: 0, -1.0: 1}  # +1 is a leftward choice, -1 rightward
```

iii. The AI misunderstood the IBL convention (stated -1=left when it is actually -1=right), resulting in an inverted choice encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
```

iii. This is a standard IBL trials table field.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three probability values are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, matching the instructions.

ii.
```python
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

iii. Matches the instruction mapping exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (timestamps).

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

iii. These are the raw wheel recordings from which velocity/speed must be derived.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI reimplements the wheel processing: (1) interpolate position onto a 1000 Hz grid, (2) apply an 8th-order 20Hz Butterworth low-pass filter (sosfiltfilt), (3) differentiate to get velocity, (4) take absolute value for speed. This matches the ibllib `SessionLoader` processing. The speed is then interpolated to trial-aligned bins using `interp1d` with `fill_value='extrapolate'`.

ii.
```python
def interpolate_wheel(timestamps, position, fs=WHEEL_FS, corner_freq=WHEEL_CORNER_FREQ, order=WHEEL_FILTER_ORDER):
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t)
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    return t, np.abs(vel).astype(np.float32)
```

iii. The AI reimplemented the ibllib wheel processing to avoid API dependency.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 equal-frequency bins using quantile boundaries (33rd and 67th percentiles). The quantiles are computed across all timepoints of all trials within the session.

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(int), boundaries
```

iii. Equal-frequency bins produce roughly 33/33/33% distribution.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to trial-aligned bins using `interp1d` with `np.linspace(t_start + binsize, t_end, n_bins)` as the query points. This grid differs from the neural data's bin centers by ~0.01s.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The AI uses the same grid as the time input, but this differs slightly from the bin centers used in the reference.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and corresponding `_ibl_leftCamera.times.npy` (or right).

ii.
```python
def load_motion_energy(alf_path, side='left'):
    me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    me = np.load(me_file).flatten()
    times = np.load(times_file).flatten()
```

iii. Left camera is preferred, with fallback to right camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are used as-is (no filtering or normalization). They are interpolated to trial-aligned bins using `interp1d`, then discretized into 3 equal-frequency bins.

ii.
```python
me_times, me_values = load_motion_energy(alf_path, side='left')
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: quantile-based discretization into 3 equal-frequency bins using percentiles computed across all timepoints of all trials within the session.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same method as wheel speed: interpolated using `interp1d` to `np.linspace(t_start + binsize, t_end, n_bins)`, which differs slightly from neural bin centers.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Sessions with missing paths are skipped, (2) probes that fail to load are skipped, (3) sessions with fewer than 2 valid trials are skipped, (4) trials where wheel or whisker ME data has NaN are excluded via a combined mask, (5) length mismatches between motion energy values and timestamps are handled by truncating to the shorter length.

ii.
```python
# Handle length mismatch (common in IBL data)
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```
```python
if n_valid < 2:
    print(f"  Session {eid}: fewer than 2 trials with all data, skipping")
    return None
```

iii. The AI handled missing data by skipping/dropping rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (reading large npy files) and the wheel interpolation/filtering (1000 Hz resampling and Butterworth filter). The AI reports ~3-6s per session, processing 459 sessions sequentially in ~29 minutes total.

ii.
```python
spk, clu = load_spike_sorting(alf_path, probe_name)  # expensive I/O
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)  # expensive computation
```

iii. File I/O dominates processing time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per trial. The behavior interpolation loop also iterates per trial. Both could potentially be vectorized. The `compute_trial_in_block` function uses a Python for-loop over all trials that could use vectorized cumsum/diff operations.

ii.
```python
for trial_idx in range(n_trials):  # spike binning loop
    ...
for trial_idx in range(n_trials):  # behavior interpolation loop
    ...
def compute_trial_in_block(prob_left):  # Python loop
    for i in range(len(prob_left)):
        ...
```

iii. The per-trial loops are the main vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. The behavior interpolation function `interpolate_behavior_to_bins` is called twice (once for wheel, once for whisker ME), repeating the same trial-window computation each time. The `find_latest_revision` function is called multiple times per session to locate different files.

ii.
```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
me_binned, me_good = interpolate_behavior_to_bins(...)
```

iii. The repeated computation of trial windows could be factored out.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `clusters.depths.npy` for each probe, which is stored in the merged clusters dict but never used in the final output. The `metrics` dataframe (quality labels) is loaded but never used for filtering since all clusters are kept. The cluster-to-channel mapping and depths are carried through merging but only the brain region mapping matters.

ii.
```python
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
metrics_file = rev_dir / 'clusters.metrics.pqt'
if metrics_file.exists():
    metrics = pd.read_parquet(metrics_file)
```

iii. Loading unused data adds I/O overhead without contributing to the output.
