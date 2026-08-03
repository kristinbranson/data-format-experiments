# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv` (the BWM freeze file) to identify all 459 sessions (by eid). For each session, it constructs filesystem paths to the ONE cache directory (`/app/data/one_cache/<lab>/Subjects/<subject>/<date>/001`) and loads data directly from numpy/parquet files on disk, rather than using the ONE API. It iterates over all sessions in sorted eid order.

ii.
```python
def find_session_paths(cache_dir):
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
    sessions = {}
    for _, row in bwm_df.iterrows():
        eid = row['eid']
        lab = row['lab']
        subject = row['subject']
        date = row['date']
        probe_name = row['probe_name']
        session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
        ...
        sessions[eid]['probes'].append(probe_name)
    return sessions
```

iii. The AI justified this approach by noting the reference code uses `bwm_release.csv` as the freeze file and ONE API. Since the ONE API was not available in the offline environment, the AI constructed equivalent file paths from the CSV metadata. This is documented in CONVERSION_NOTES.md.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. Each session is associated with a subject, and unique subjects are tracked in order of first appearance. A `subject_idx` array maps each session to its subject.

ii.
```python
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
```

iii. This follows from the BWM CSV structure where each row has a subject field. The AI identified 135 unique subjects.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. Multiple probes within the same eid are merged into a single session. Each session produces one entry in the neural/input/output lists.

ii.
```python
for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)
```

iii. The AI correctly identified that the reference code merges probes within a session (`merge_probes()`), treating each eid as one session. 459 sessions were found, 439 processed, 20 skipped.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` parquet files. Each row in the trials table represents one trial. A quality mask is applied (see 1-e), and valid trials are iterated individually to construct per-trial neural and behavioral arrays.

ii.
```python
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
trials = pd.read_parquet(trials_files[0])
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    stim_on = trial[ALIGN_TIME]
    t_start = stim_on + TIME_WINDOW[0]
    t_end = stim_on + TIME_WINDOW[1]
```

iii. Trials are split using the stimOn_times alignment event, with a 2-second window per trial. This matches the reference code's approach in `bin_spiking_data()`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) any of the key event columns contain NaN (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`); (2) reaction time (firstMovement_times - stimOn_times) is < 0.08s or > 2.0s; (3) choice == 0 (no response). Additionally, trials are dropped if wheel speed or whisker ME interpolation fails or produces NaN values.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType']
...
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

iii. The AI states this matches `load_trials_and_mask()` with default parameters. The reference code also passes `max_trial_len=10.0` in `prepare_data()`, which the AI did not implement. The reference `exclude_nochoice` defaults to `True`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) located in the pykilosort directory for each probe.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. This matches the reference code which loads spikes via `SpikeSortingLoader.load_spike_sorting()` returning spikes with 'times' and 'clusters' keys.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over a 2-second window (-0.5s to 1.5s relative to stimulus onset), producing a (n_clusters, 100) matrix per trial. For sessions with multiple probes, spike data is merged with cluster ID offsets. Cluster IDs are remapped to sequential indices.

ii.
```python
def bin_spikes_trial(spike_times, spike_clusters, n_clusters, t_start, t_end, binsize, n_bins):
    i_start = np.searchsorted(spike_times, t_start, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    ...
    bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
    return binned
```

iii. The binning approach is functionally equivalent to the reference code's `bincount2D` and `get_spike_data_per_interval`, though implemented differently. The reference uses `bincount2D` from iblutil.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ALL clusters are loaded regardless of quality labels. No quality-based filtering of neurons is applied.

ii.
```python
# In load_spikes: loads all spike data, cluster_labels loaded but not used for filtering
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. The AI notes this matches the reference code where `load_spiking_data()` is called without the `qc` parameter (defaults to None), meaning all clusters are loaded. The CONVERSION_NOTES.md states: "Load ALL clusters (not filtered by quality metric)...matches the ML approach where the decoder can learn from all available neural signals."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data window is computed as `[stimOn_times - 0.5, stimOn_times + 1.5]` (2-second window centered on stimulus onset with offset). Spikes are binned within this absolute time window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
neural = bin_spikes_trial(spike_times, spike_clusters_remapped, n_clusters,
    t_start, t_end, BINSIZE, N_BINS)
```

iii. This matches the reference code params `'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)` and the instruction "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms (0.02s), resulting in 100 time bins per 2-second trial. No rebinning is applied - this is the native bin size used throughout.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. This matches the reference code parameter `'binsize': 0.02`. The CONVERSION_NOTES mention the methods paper uses both 50ms and 20ms but the caching code uses 20ms uniformly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from a raw data variable. It is computed analytically as a linearly spaced array from -0.48s to 1.5s (bin centers within the time window).

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. Since the trial is aligned to stimulus onset and the time window is fixed, the time-since-stimulus-onset values are deterministic for every trial. This is a reasonable approach.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linear space of 100 values from -0.48s to 1.5s is generated using `np.linspace`. These represent bin centers within the [-0.5, 1.5] window.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# Results in: [-0.48, -0.4602, ..., 1.48, 1.5]
```

iii. The bin centers match the interpolation grid used for behavioral signals (`np.linspace(interval_beg + binsize, interval_end, n_bins)`), ensuring temporal alignment.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time-since-stimulus array uses the same bin centers as the behavioral interpolation grid and the same number of bins (100) as the neural data. Since both neural binning and the time array use the same time window and bin size, they are inherently aligned.

ii.
```python
# Neural bins: t_start to t_end in BINSIZE steps
# Time input: linspace(TIME_WINDOW[0]+BINSIZE, TIME_WINDOW[1], N_BINS)
# Both produce 100 values spanning the same 2-second interval
```

iii. No explicit alignment step is needed since both are constructed from the same parameters.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    trials_masked = trials_df[mask].copy()
    prob_left = trials_masked['probabilityLeft'].values
    block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
    trial_in_block = np.zeros(len(prob_left), dtype=int)
    for i in range(len(block_changes)):
        start = block_changes[i]
        end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
        trial_in_block[start:end] = np.arange(end - start)
    return trial_in_block
```

iii. The reference code stores `block` as raw `probabilityLeft` values. Trial number in block is a task-specific requirement from the decoder instructions, not directly present in the reference code.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected by finding indices where `probabilityLeft` changes. Within each block, trials are numbered sequentially starting from 0. This is computed on the masked (quality-filtered) trials only. The trial number is broadcast to all time bins as a constant per-trial value.

ii.
```python
trial_num = np.float32(trial_in_block[trial_idx])
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. Computing trial number in block on masked trials (after filtering) means the numbering reflects only valid trials, which may differ from the actual trial position within the block if some trials were filtered out.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table. In the IBL convention, choice=1 means left and choice=-1 means right.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1  # IBL: 1=left, -1=right -> map: left=0, right=1
```

iii. The mapping follows the decoder task specification: left=0, right=1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL choice values (1=left, -1=right) are remapped to decoder format (0=left, 1=right). The choice value is broadcast to all time bins as a constant per-trial value.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1
...
output = np.stack([
    np.full(N_BINS, choice_val, dtype=np.int64),
    ...
], axis=0)
```

iii. This matches the decoder task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = trial['probabilityLeft']
if prob_left == 0.2:
    prior_val = 0
elif prob_left == 0.5:
    prior_val = 1
elif prob_left == 0.8:
    prior_val = 2
else:
    prior_val = 1  # fallback
```

iii. The `probabilityLeft` values (0.2, 0.5, 0.8) represent the block-specific prior probability of the stimulus appearing on the left.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability values are mapped to categorical indices: 0.2->0, 0.5->1, 0.8->2. A fallback to 1 is used for unexpected values. The prior value is broadcast to all time bins.

ii.
```python
if prob_left == 0.2:
    prior_val = 0
elif prob_left == 0.5:
    prior_val = 1
elif prob_left == 0.8:
    prior_val = 2
else:
    prior_val = 1  # fallback
```

iii. This matches the decoder task specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` files.

ii.
```python
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. The AI loads raw wheel position and timestamps, then computes speed from these.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Velocity is computed as simple finite differences of wheel position divided by time intervals. Speed is the absolute value of velocity. The speed is then interpolated to bin centers and discretized into 3 quantile-based bins per session.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

iii. The CONVERSION_NOTES acknowledge: "The reference code uses `SessionLoader.load_wheel()` which applies Gaussian smoothing. Our implementation uses simple finite differences, which is noisier but preserves the same information." The reference code uses `np.abs(sess_loader.wheel['velocity'].to_numpy())` which provides Gaussian-smoothed velocity.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using quantile-based edges computed per-session. Quantiles at 0%, 33.3%, 66.7%, and 100% are computed across all time points within a session, producing 3 equal-count bins labeled 0 (low), 1 (medium), 2 (high).

ii.
```python
def discretize_to_bins(values_list, n_bins, compute_edges=True, edges=None):
    all_vals = np.concatenate([v[~np.isnan(v)] for v in values_list if v is not None])
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_vals, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ...
    d = np.digitize(v, edges[1:-1])
```

iii. The discretization is done per-session using quantiles. The CONVERSION_NOTES acknowledge this as a known limitation: "bin edges vary across sessions."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same bin centers used for neural data using linear interpolation with extrapolation.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, t_start, t_end, binsize, n_bins):
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. This matches the reference code's `get_behavior_per_interval()` which uses `np.linspace(interval_begs + binsize, interval_ends, n_bins)` and `interp1d(..., kind='linear', fill_value='extrapolate')`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and `_ibl_leftCamera.times.npy`.

ii.
```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
me = np.load(left_me_files[0])
times = np.load(left_times_files[0])
```

iii. The reference code loads `whiskerMotionEnergy` via `SessionLoader.load_motion_energy()`, which is a specific subset of motion energy targeting the whisker region. The AI loads `ROIMotionEnergy`, which is the full ROI motion energy and may differ from the whisker-specific signal.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy signal is loaded and interpolated to bin centers matching the neural data, then discretized into 3 quantile-based bins per session. No additional processing (smoothing, normalization) is applied.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
...
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. The processing is minimal: load, interpolate, discretize.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: per-session quantile-based discretization into 3 bins (low/medium/high).

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. Uses the same `discretize_to_bins` function as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: linear interpolation to bin centers matching neural data.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```

iii. Uses the same `interpolate_behavior_to_bins` function as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions without trials data, spike data, or wheel data are skipped entirely. (2) Trials where behavioral interpolation fails or produces NaN are skipped. (3) Sessions with fewer than 2 valid trials are skipped. (4) A divide-by-zero warning in wheel velocity computation (when timestamps have zero difference) is not explicitly handled but propagates as inf values which are later filtered out during interpolation/NaN checks. (5) Brain region IDs that can't be mapped default to root (0). (6) If whisker ME length doesn't match camera times, the shorter array is used.

ii.
```python
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
...
if len(me) == len(times) - 1:
    return times[:-1], me
```

iii. The AI's approach is pragmatic: skip problematic data rather than impute. This resulted in 20 sessions being skipped entirely and additional trials being dropped within sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading spike data from disk (multiple numpy files per probe). (2) Binning spikes for each trial (iterating over all trials with searchsorted and np.add.at). (3) Interpolating behavioral signals for each trial. (4) The remapping of spike clusters to sequential indices using a Python list comprehension over potentially millions of spikes.

ii.
```python
# Slow: Python-level remapping of millions of spike cluster IDs
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```

iii. No justification provided for the slow remapping approach. A vectorized approach using np.searchsorted or a lookup array would be much faster.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The spike cluster remapping loop (`[cluster_id_to_idx[c] for c in spike_clusters]`) could use `np.searchsorted` or a pre-allocated lookup array. (2) The per-trial spike binning loop could potentially be vectorized across trials. (3) The brain region ID mapping loop (`for cid in cluster_ids`) could be vectorized with array indexing.

ii.
```python
# Could be vectorized:
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
# And:
for cid in cluster_ids:
    if cid < len(all_channels):
        ch = int(all_channels[cid])
```

iii. No justification for not vectorizing these operations.

## 10-c. What processing does the code repeat multiple times?

i. (1) The `interp1d` interpolation object is created fresh for each trial for both wheel speed and whisker ME, rather than being created once per session. (2) The `np.linspace` for bin centers is recomputed identically for every trial within a session. (3) The time_since_stim array is computed identically for every trial.

ii.
```python
# Repeated per trial, but identical values:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# Repeated per trial in interpolate_behavior_to_bins:
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. No justification provided. These are minor inefficiencies.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Cluster labels/metrics are loaded (`clusters.metrics.pqt`) but never used for filtering since all clusters are kept. (2) Cluster depths are loaded but not used in the final output. (3) Wheel edges and whisker edges are stored per-session in the result dict but not included in the final saved data. (4) The `cluster_labels` variable is passed through merge_probes but never used.

ii.
```python
# Loaded but never used for filtering:
metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
cluster_labels = None
if metrics_files:
    metrics = pd.read_parquet(metrics_files[0])
    cluster_labels = metrics['label'].values
# Also loaded but unused:
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
```

iii. No justification for loading unused data.
