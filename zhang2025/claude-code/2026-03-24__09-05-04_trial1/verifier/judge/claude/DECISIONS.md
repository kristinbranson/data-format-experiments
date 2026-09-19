# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a BWM release CSV file (`/app/code/code_zhang2025/data/bwm_release.csv`) to get the list of sessions, then constructs file paths directly into the ONE cache directory structure (`data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/alf/`). It loads data files (trials parquet, spike npy files, wheel npy files, camera npy files) by navigating the directory hierarchy and finding the latest revision. It does NOT use the ONE API.

ii.
```python
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)

def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
    if sess_path.exists():
        return sess_path
    return None
```

iii. The AI chose direct file loading to avoid needing the ONE API and to work offline from the cache. This resulted in 392 of 459 sessions being found (67 skipped due to missing paths).

## 1-b. How are the data split into subjects?

i. Subject names come from the BWM CSV column. Sessions are processed sequentially, and at the end all unique subject names are collected and sorted.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The CSV provides the subject name per session, so no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Each row in the BWM CSV (grouped by eid) is a session. The AI groups by eid to get one row per session.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

iii. The BWM CSV has one row per probe insertion, so grouping by eid gives unique sessions.

## 1-d. How are the data split into trials?

i. Each session's trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI reads this parquet file and each row is one trial.

ii.
```python
def load_trials(alf_path):
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
    trials = pd.read_parquet(trials_file)
    return trials
```

iii. No decision to make; the trials table already has one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) exclude NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType; (2) reaction time between 0.08s and 2.0s; (3) exclude no-choice trials (choice==0); (4) max trial length filter (feedback_times - goCue_times <= 10s). Additionally, trials without complete wheel or motion energy data are excluded via a combined mask.

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

iii. The AI states these filters match the reference code's `load_trials_and_mask` function, which excludes NaN in key events, applies RT bounds, and excludes no-choice trials. The max trial length filter is from the reference code as well.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster IDs). Also `clusters.channels.npy`, `clusters.depths.npy`, `clusters.metrics.pqt`, and `channels.brainLocationIds_ccf_2017.npy` for metadata.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over the trial window (-0.5s to 1.5s from stimulus onset). The raw spike counts are stored as uint8. Notably, the AI does NOT convert spike counts to firing rates (Hz) by dividing by bin size. Multiple probes are merged by offsetting cluster IDs.

ii.
```python
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters, interval_starts, interval_ends, binsize, n_bins):
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    ...
    bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
    flat_idx = c_sel[valid] * n_bins + bin_idx[valid]
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
    return binned

# Later stored as uint8:
neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. The CONVERSION_NOTES.md notes the uint8 storage was done to fit in 64GB container memory. The lack of conversion to firing rates is not justified anywhere.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by quality. All clusters are used regardless of their quality label. There is no filtering to remove `void` brain regions (channels placed outside the brain).

ii.
```python
# No quality filtering code exists. All clusters are used:
n_clusters = len(clusters['channels'])
cluster_regions = get_brain_regions(clusters, br)
```

iii. The CONVERSION_NOTES.md explicitly states: "Follow reference code: use all units (qc=None)" based on the observation that the reference code's `prepare_data` calls `load_spiking_data` without a `qc` argument.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike bin window starts at `stimOn_times + TIME_WINDOW[0]` (-0.5s) and ends at `stimOn_times + TIME_WINDOW[1]` (1.5s). Spikes within this absolute time window are binned.

ii.
```python
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
binned_spikes = bin_spikes_vectorized(
    spikes['times'], spikes['clusters'], n_clusters,
    interval_starts, interval_ends, BINSIZE, N_BINS)
```

iii. Alignment to stimulus onset matches the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, 100 bins total over the 2s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20 ms time bins
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived purely from the time window constants and bin size, not from any raw data variable. It is a synthetic time axis.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. This is the same conceptual approach as the reference (a deterministic time axis), though the specific values differ.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A `np.linspace` from -0.48 to 1.5 with 100 points. This produces values at the right edges of time bins, NOT at bin centres.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# = np.linspace(-0.48, 1.5, 100) -> values at -0.48, -0.46, ..., 1.5
```

iii. The reference uses bin centres: `EDGES[:-1] + BIN/2` = [-0.49, -0.47, ..., 1.49]. The AI's values are shifted by +0.01s (10ms) relative to the reference.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input array has the same number of bins (100) as the neural data, so they correspond element-wise. However, the time values represent right bin edges rather than bin centres, so there is a 10ms conceptual misalignment: bin 0 of the neural data covers [-0.5, -0.48) but the time value for that bin is -0.48 (right edge) rather than -0.49 (centre).

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. No explicit justification for using right edges instead of bin centres.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where probabilityLeft changes.

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

iii. Since the trials table has no explicit block ID, the AI infers blocks from changes in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop detects block boundaries where `probabilityLeft` changes value. The counter resets to 1 at each boundary and increments for each trial. The count is computed on ALL trials before filtering, so filtered-out trials still advance the counter. The count is 1-indexed (first trial in block = 1).

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
```

iii. Computing on all trials before filtering preserves the animal's real position in the block. The 1-indexing is a choice that differs from the reference's 0-indexed cumcount.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values -1, 0, or 1 in IBL convention.

ii.
```python
choice = final_trials['choice'].values.copy()
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
```

iii. The AI states: "IBL: -1=left, 1=right" but this is INCORRECT. The actual IBL convention is choice=+1 for LEFT, choice=-1 for RIGHT.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI applies the formula `((choice + 1) // 2)` which maps -1->0 and 1->1. Due to the incorrect understanding of IBL conventions (the AI believes -1=left, 1=right when it is actually +1=left, -1=right), the encoding is REVERSED: left is encoded as 1 and right as 0, opposite to the instruction's specification of left=0, right=1.

ii.
```python
# IBL: -1=left, 1=right -> convert to 0=left, 1=right  <-- WRONG COMMENT
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
# Actual effect: left(+1)->1, right(-1)->0  <-- REVERSED
```

iii. The AI's comment documents the wrong IBL convention, leading to reversed choice encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

iii. Directly maps the three probability values to categories as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Simple categorical encoding: 0.2->0, 0.5->1, 0.8->2. No additional processing.

ii. Same as 6-a.

iii. Matches the instruction specification exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, the raw wheel position and its timestamps.

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

iii. Standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000Hz, a Butterworth low-pass filter (20Hz corner, order 8) is applied, velocity is computed by differentiation, and speed is taken as the absolute value. The AI reimplements this processing manually (matching ibllib's `interpolate_position` and `velocity_filtered`). The speed is then interpolated onto trial-aligned time points using scipy's interp1d with linear interpolation and extrapolation.

ii.
```python
def interpolate_wheel(timestamps, position, fs=WHEEL_FS, corner_freq=WHEEL_CORNER_FREQ, order=WHEEL_FILTER_ORDER):
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t)
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    return t, np.abs(vel).astype(np.float32)
```

iii. The AI notes this matches ibllib's wheel processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 equal-frequency bins using quantile boundaries computed across all time points of all trials within the session. Uses `np.percentile` at [0, 33.33, 66.67, 100] and `np.digitize` with the inner boundaries, then clips to 0..2.

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(int), boundaries

all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
```

iii. Per-session quantile-based discretization into 3 equal-frequency bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to `np.linspace(t_start + binsize, t_end, n_bins)` which are the right bin edges, same as the time input. This matches the neural data element-wise (100 bins) but evaluates at right edges rather than bin centres.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The interpolation targets are consistent with the time input but use right edges rather than bin centres.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding camera timestamps `_ibl_leftCamera.times.npy`.

ii.
```python
def load_motion_energy(alf_path, side='left'):
    me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    me = np.load(me_file).flatten()
    times = np.load(times_file).flatten()
    min_len = min(len(me), len(times))
    return times[:min_len], me[:min_len].astype(np.float32)
```

iii. Uses left camera with right camera fallback, same approach as reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded and used as-is (no filtering or normalization). They are interpolated onto trial-aligned time points using scipy's interp1d, then discretized into 3 equal-frequency bins using the same quantile approach as wheel speed.

ii.
```python
me_times, me_values = load_motion_energy(alf_path, side='left')
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS)
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: 3 equal-frequency bins using session-wide quantile boundaries.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. Per-session quantile-based discretization, same as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated to right bin edges using `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. Consistent with wheel and time input within the AI's code, but offset by 10ms from reference bin centres.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled at multiple levels: (1) Sessions with missing file paths are skipped; (2) Probes that fail to load are skipped; (3) Sessions with <2 valid trials after filtering are skipped; (4) Trials without complete wheel or camera data coverage are excluded via a combined mask; (5) Length mismatch between camera timestamps and motion energy is handled by truncating to minimum length.

ii.
```python
# Length mismatch handling
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)

# Combined mask for behavior coverage
combined_mask = np.ones(len(valid_trials), dtype=bool)
if wheel_speed_binned is not None:
    combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
else:
    combined_mask[:] = False
```

iii. The AI handles missing data by dropping affected trials/sessions rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (reading large npy files) and the spike binning computation. The AI reports ~3-6s per session, with the full run taking ~29 minutes for 459 sessions.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

iii. The AI reports timing per session in the output log.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized: (1) the per-trial spike binning loop in `bin_spikes_vectorized`; (2) the per-trial behavior interpolation loop in `interpolate_behavior_to_bins`. Both iterate over trials.

ii.
```python
# Per-trial spike binning loop
for trial_idx in range(n_trials):
    ...
    bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
    flat_idx = c_sel[valid] * n_bins + bin_idx[valid]
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)

# Per-trial behavior interpolation loop
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. Both loops are essentially the same structure as the reference code's per-trial loops.

## 10-c. What processing does the code repeat multiple times?

i. The wheel processing (interpolation, filtering, velocity computation) is done once per session, which is appropriate. The behavior interpolation function `interpolate_behavior_to_bins` is called twice per session (once for wheel, once for whisker ME) with similar logic. No significant redundant processing.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads cluster depths, cluster channels, and cluster metrics for all clusters even though no quality filtering is applied. The metrics data is loaded but never used for filtering. Additionally, `merged_channels` and `merged_depths` are computed during probe merging but not used in the final output.

ii.
```python
clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
metrics_file = rev_dir / 'clusters.metrics.pqt'
if metrics_file.exists():
    metrics = pd.read_parquet(metrics_file)
```

iii. The AI loads these for potential use but doesn't filter on them.
