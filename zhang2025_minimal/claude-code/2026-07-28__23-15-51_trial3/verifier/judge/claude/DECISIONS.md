# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`/app/code/code_zhang2025/data/bwm_release.csv`) to get the list of sessions with their EIDs, lab, subject, date, and probe names. It then constructs file paths directly from these metadata fields (e.g., `cache_dir / lab / 'Subjects' / subject / date / '001'`) and loads data files (trials, spikes, wheel, whisker) by globbing for them on disk (e.g., `session_path.rglob('_ibl_trials.table.pqt')`). It does NOT use the ONE API to load data.

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
```

iii. The agent initially tried to use the ONE API but encountered issues with the dataset registry only listing a subset of files. It then decided: "The ONE cache has files on disk but the dataset registry only has a subset. Let me work directly with the files." (Step 56)

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the `bwm_release.csv` file. Each session is associated with a subject, and unique subjects are tracked in an ordered list as sessions are processed. Subject indices are assigned based on insertion order.

ii.
```python
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
```

iii. The subject identity comes directly from the CSV metadata. No special parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their unique `eid` in the `bwm_release.csv` file. Each unique EID corresponds to one session. Multiple probes from the same session are grouped together under the same EID.

ii.
```python
for _, row in bwm_df.iterrows():
    eid = row['eid']
    ...
    if eid not in sessions:
        sessions[eid] = {
            'path': session_path,
            'probes': [],
            'subject': subject,
            ...
        }
    sessions[eid]['probes'].append(probe_name)
```

iii. The CSV has one row per probe insertion, so sessions with multiple probes have multiple rows. The code groups by EID to recover sessions.

## 1-d. How are the data split into trials?

i. Trials come from the `_ibl_trials.table.pqt` parquet file, which has one row per trial. The code iterates over valid (masked) trials one at a time.

ii.
```python
trials = pd.read_parquet(trials_files[0])
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    stim_on = trial[ALIGN_TIME]
```

iii. The trials table already has one row per trial; no splitting decision is needed.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) Exclude trials with NaN in key columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`). (2) Exclude trials with reaction time outside [0.08, 2.0] seconds. (3) Exclude no-choice trials (`choice == 0`). (4) Trials where wheel or whisker motion energy cannot be interpolated to the trial window are also dropped.

ii.
```python
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

And later in the trial loop:
```python
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. The agent referenced the reference code's `load_trials_and_mask` function for the NaN exclusion and RT bounds (Step 67). The behavioral coverage check is done implicitly via interpolation failure.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike times) and `spikes.clusters.npy` (cluster assignments). Additionally, `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`, and optionally `clusters.metrics.pqt` are loaded for region mapping.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
```

iii. The agent identified these files by exploring the data directory structure on disk.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over the [-0.5, 1.5] s trial window, producing a (n_clusters, 100) matrix per trial. The spike counts are stored as float32 but are NOT divided by the bin width (i.e., they remain as counts, not firing rates). When a session has multiple probes, clusters are merged with offset cluster IDs and sorted by time.

ii.
```python
def bin_spikes_trial(spike_times, spike_clusters, n_clusters, t_start, t_end, binsize, n_bins):
    binned = np.zeros((n_clusters, n_bins), dtype=np.float32)
    bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
    return binned
```

iii. The agent noted from the reference code that 20 ms bins are used. The code does not convert counts to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads ALL clusters without any quality filtering. Cluster metrics are loaded but never used for filtering in the final code. The `cluster_labels` variable is loaded but not applied as a filter.

ii.
```python
# Labels are loaded:
metrics = pd.read_parquet(metrics_files[0])
cluster_labels = metrics['label'].values
# But never used for filtering. All clusters are kept:
cluster_ids = np.unique(spike_clusters)
n_clusters = len(cluster_ids)
```

iii. The agent explicitly reasoned about this: "The reference code loads ALL clusters (not just good ones - `qc=None` in `prepare_data`)" and "loading all clusters is correct - the decoder can learn from all available neural data" (Steps 67, 83). The agent also noted that `void` regions are not filtered out.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are extracted from the window [stimOn_times - 0.5, stimOn_times + 1.5] using searchsorted, and bin indices are computed relative to the window start. This aligns the neural data to stimulus onset.

ii.
```python
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
neural = bin_spikes_trial(spike_times, spike_clusters_remapped, n_clusters, t_start, t_end, BINSIZE, N_BINS)
```

iii. The agent identified `stimOn_times` as the alignment event from the reference code configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or resampling is applied to the neural data.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. The agent confirmed this matches the reference code's `binsize: 0.02` parameter.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` in the trials table, which defines the alignment event. The input values are computed as evenly-spaced bin centers from the time window.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The agent used the reference code's time window parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as 100 evenly spaced points from (-0.5 + 0.02) = -0.48 to 1.5, using `np.linspace`. These represent the right edges of each bin rather than the centers.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. No special processing; the variable is defined by the binning grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values are computed to correspond to the neural bins. However, the AI uses `np.linspace(-0.48, 1.5, 100)` which gives right-edge-aligned bin times rather than bin centers. The reference uses `EDGES[:-1] + BIN/2` which gives true bin centers at -0.49, -0.47, ..., 1.49.

ii.
```python
# AI:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
# This gives: -0.48, -0.46, ..., 1.48, 1.5
```

iii. The agent did not explicitly discuss the distinction between bin centers and bin right edges.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block changes are detected where `probabilityLeft` changes value.

ii.
```python
prob_left = trials_masked['probabilityLeft'].values
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
```

iii. The agent recognized that blocks must be derived from changes in `probabilityLeft`, consistent with the reference approach.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected from changes in `probabilityLeft`. Within each block, trials are numbered sequentially from 0. However, the AI computes this AFTER filtering trials (on masked trials only), so the block count reflects the position among surviving trials, not the original position.

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

iii. The agent did not discuss whether the block count should use original or filtered trial indices.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, where IBL convention is +1 = left, -1 = right.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1  # IBL: 1=left, -1=right -> map: left=0, right=1
```

iii. The agent correctly identified the IBL choice convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple remapping: +1 (left) -> 0, -1 (right) -> 1. No-choice trials (choice == 0) are already excluded by the trial mask.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. Matches the instruction specification of left=0, right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

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

iii. The agent followed the instruction specification for the mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping of 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, with a fallback to 1 for unexpected values. The reference does not include a fallback; it filters out non-standard values instead.

ii.
```python
else:
    prior_val = 1  # fallback
```

iii. The fallback is a defensive choice, though in practice all valid `probabilityLeft` values should be 0.2, 0.5, or 0.8 (and others are excluded by the NaN filter in the reference approach).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, loaded directly from disk.

ii.
```python
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. The agent identified the correct wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes velocity as a simple finite difference of position divided by time intervals (`dp/dt`), then takes the absolute value for speed. This does NOT use the `SessionLoader`'s built-in interpolation to 1000 Hz and Butterworth filtering that the reference uses. The velocity timestamps are taken as the midpoints of the original timestamps.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

iii. The agent noted looking at the reference code's wheel-speed computation but implemented a simpler finite-difference approach rather than using `SessionLoader`'s interpolated and filtered velocity.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using quantile-based edges. Quantiles are computed across all trials of the session (0%, 33.3%, 66.7%, 100%), and `np.digitize` assigns each value to a bin.

ii.
```python
def discretize_to_bins(values_list, n_bins, compute_edges=True, edges=None):
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_vals, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    d = np.digitize(v, edges[1:-1])  # Returns 0 to n_bins-1
```

iii. The agent mentioned "discretize wheel speed into 3 bins" in the docstring. The percentile-based approach is similar to the reference but the edge handling differs slightly (using -inf/inf vs. the reference's simpler `np.percentile(trace, SPLIT)`).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to bin centers of each trial using `scipy.interpolate.interp1d` with linear interpolation and extrapolation. The bin centers are computed as `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, t_start, t_end, binsize, n_bins):
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. The interpolation approach aligns the behavioral data to the same time grid as the neural data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), along with the corresponding camera timestamps `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
```

iii. The agent correctly prioritized the left camera, falling back to right camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly and interpolated to the trial bin centers using the same `interpolate_behavior_to_bins` function as the wheel speed. Then discretized into 3 bins using session-level quantiles.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. The processing follows the same pipeline as wheel speed: interpolate, then discretize.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 bins using session-level quantile edges (33.3%, 66.7%).

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. Consistent approach with wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to the trial bin centers using `scipy.interpolate.interp1d` with linear interpolation.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several checks handle missing data: (1) Sessions without trials data are skipped. (2) Probes without spike files are skipped. (3) Sessions with fewer than 2 valid trials are skipped. (4) Trials where wheel or whisker interpolation fails (returns None) are dropped. (5) Trials with NaN in behavioral data are dropped. (6) NaN values in key trial columns are excluded via the trial mask.

ii.
```python
if trials is None:
    return None
if n_valid < 2:
    return None
if spike_times is None:
    continue
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. The agent handled missing data by skipping problematic sessions/trials, resulting in 439 of 459 sessions being processed.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk (`np.load` for `spikes.times.npy` and `spikes.clusters.npy` which are large arrays) and the per-trial spike binning loop.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. The agent noted the conversion was slow and attempted to optimize the spike binning (Step 85).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops could be vectorized: (1) The per-trial spike binning in `bin_spikes_trial` called in a loop over trials. (2) The per-trial behavioral interpolation in the main trial loop. Additionally, the cluster remapping uses a Python list comprehension over all spikes: `[cluster_id_to_idx[c] for c in spike_clusters]`.

ii.
```python
# Slow cluster remapping:
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])

# Per-trial loop:
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    neural = bin_spikes_trial(...)
    ws = interpolate_behavior_to_bins(...)
    wm = interpolate_behavior_to_bins(...)
```

iii. The agent recognized the per-spike loop was slow and attempted optimization (Step 85).

## 10-c. What processing does the code repeat multiple times?

i. The `interpolate_behavior_to_bins` function creates a new `interp1d` interpolation object for each trial, recalculating the selection mask on the full behavioral time series each time. The `np.linspace` call for `time_since_stim` and `bin_centers` is also repeated for every trial despite always producing the same values.

ii.
```python
# Called once per trial:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# And inside interpolate_behavior_to_bins:
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The agent did not discuss this redundancy.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `clusters.depths.npy` but never uses it. It also loads cluster metrics (`clusters.metrics.pqt`) for labels but never applies them for filtering. The `wheel_edges` and `whisker_edges` are returned from `process_session` but not used in the final data dictionary.

ii.
```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')  # never used
cluster_labels = metrics['label'].values  # loaded but not used for filtering
# In process_session return:
'wheel_edges': wheel_edges,    # not used in assemble
'whisker_edges': whisker_edges,  # not used in assemble
```

iii. The agent did not discuss discarding unused processing results.
