# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`bwm_release.csv`) from the reference code's data directory to get a list of all sessions (eids) with their lab, subject, date, and probe info. It then constructs file paths into the ONE cache directory structure (`lab/Subjects/subject/date/001`). For each session, it loads trials from parquet files (`_ibl_trials.table.pqt`), spike data from npy files in the `alf/probe/pykilosort/` directories, wheel data from `_ibl_wheel.position.npy`/`_ibl_wheel.timestamps.npy`, and whisker motion energy from `leftCamera.ROIMotionEnergy.npy`/`_ibl_leftCamera.times.npy`.

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

iii. The AI justified using `bwm_release.csv` because it's the same session list used by the reference code (`0_data_caching.py` line 37). Loading from the ONE cache filesystem directly avoids needing the ONE API.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` field in `bwm_release.csv`. As sessions are processed, unique subjects are accumulated into an ordered list (`subject_set`), and each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subject_set = []  # ordered unique subjects
...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
```

iii. The AI noted that subjects are identified per-session from the BWM release metadata, consistent with the reference approach.

## 1-c. How are the data split into sessions?

i. Sessions are identified by their unique `eid` (experiment ID) from `bwm_release.csv`. Multiple probes with the same `eid` are grouped into a single session. Each session is processed independently and stored as a separate entry in the output lists.

ii.
```python
for _, row in bwm_df.iterrows():
    eid = row['eid']
    ...
    if eid not in sessions:
        sessions[eid] = {
            'path': session_path, 'probes': [], 'subject': subject, ...
        }
    sessions[eid]['probes'].append(probe_name)
```

iii. The AI noted that the reference code also groups probes by eid and merges them within a session.

## 1-d. How are the data split into trials?

i. Trials are loaded from parquet files (`_ibl_trials.table.pqt`). For each valid trial, the time window is computed as `stimOn_times + (-0.5, 1.5)` to create a 2-second interval. Neural data is binned within this window. Each trial becomes a separate entry in the session's trial list.

ii.
```python
trials = pd.read_parquet(trials_files[0])
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    stim_on = trial[ALIGN_TIME]
    t_start = stim_on + TIME_WINDOW[0]
    t_end = stim_on + TIME_WINDOW[1]
```

iii. The AI justified using `stimOn_times` alignment and the (-0.5, 1.5) window based on the reference code params: `'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) any of the events `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` are NaN; (2) reaction time (firstMovement_times - stimOn_times) is outside [0.08, 2.0] seconds; (3) choice == 0 (no response). Additionally, trials are skipped if wheel speed or whisker ME interpolation fails or produces NaN values. The AI does NOT apply the `max_trial_len=10.0` filter (feedback_times - goCue_times > 10s) that the reference code uses.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType']
...
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()
if 'firstMovement_times' in trials.columns and 'stimOn_times' in trials.columns:
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)
```

iii. The AI justified these exclusions as matching `load_trials_and_mask()` with default parameters. However, the reference code also passes `max_trial_len=10.0` which the AI omitted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from the `alf/probe/pykilosort/` directories.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. The AI noted this matches the reference code which loads spike times and clusters via `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Spikes from multiple probes are merged (with cluster ID offsets), then sorted by time. For each trial, spikes in the time window [stimOn - 0.5s, stimOn + 1.5s] are binned into 20ms non-overlapping bins, producing a (n_clusters, 100) matrix of spike counts per trial. The AI uses `np.searchsorted` for efficiency and `np.add.at` for vectorized bin filling.

ii.
```python
def bin_spikes_trial(spike_times, spike_clusters, n_clusters, t_start, t_end, binsize, n_bins):
    i_start = np.searchsorted(spike_times, t_start, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    times_sel = spike_times[i_start:i_end]
    clusters_sel = spike_clusters[i_start:i_end]
    binned = np.zeros((n_clusters, n_bins), dtype=np.float32)
    bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    valid = clusters_sel < n_clusters
    np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
    return binned
```

iii. The AI justified this as equivalent to the reference's `bincount2D` approach but implemented without the iblutil dependency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to clusters/neurons. ALL clusters from Kilosort are included regardless of quality metrics. The AI loads `clusters.metrics.pqt` but does not use it for filtering.

ii.
```python
# In load_spikes:
metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
cluster_labels = None
if metrics_files:
    metrics = pd.read_parquet(metrics_files[0])
    cluster_labels = metrics['label'].values
# cluster_labels is never used for filtering
```

iii. The AI justified this by noting the reference code calls `load_spiking_data(one, pid)` without the `qc` parameter (defaults to None), meaning all clusters are loaded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to stimulus onset (`stimOn_times`). The spike binning window starts at `stimOn_times - 0.5s` and ends at `stimOn_times + 1.5s`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
```

iii. Matches the reference code params and decoder task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, producing 100 time bins per 2-second trial. No temporal rebinning is applied - this is the native resolution of the converted data.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. The AI justified 20ms based on the reference code's `'binsize': 0.02`. The methods paper mentions both 50ms (for choice/prior) and 20ms (for wheel/whisker ME), but the caching code uses 20ms uniformly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from raw data variables. It is computed as a deterministic time vector based on the alignment window parameters: `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)`, which gives bin center times relative to stimulus onset.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The AI noted this provides a continuous time signal from -0.48s to 1.5s in 100 steps, representing bin centers.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data is involved. The time vector is computed analytically using `np.linspace` from `-0.5 + 0.02 = -0.48` to `1.5` with 100 points. This is identical for every trial.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The bin centers match the reference code's interpolation grid: `np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time vector is constructed to match the neural data bin centers. Both use the same window (-0.5s to 1.5s) and bin size (20ms), so the 100 time values correspond 1:1 with the 100 neural time bins.

ii.
```python
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. Alignment is implicit through shared parameters.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where `probabilityLeft` changes value, and trials within each block are numbered sequentially starting from 0.

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

iii. The AI derived block structure from changes in `probabilityLeft`, which reflects the task's block structure (0.2/0.5/0.8 probability blocks).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected by finding positions where `probabilityLeft` changes value using `np.diff`. Within each block, trials are assigned sequential 0-indexed numbers. The computation is done only on masked (valid) trials, so excluded trials affect the numbering. The trial number is stored as a per-trial scalar broadcast to all time bins.

ii.
```python
trial_num = np.float32(trial_in_block[trial_idx])
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. The AI computed block boundaries from filtered trials, meaning trial-in-block numbering reflects position among valid trials only, not absolute position in the original trial sequence.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1  # IBL: 1=left, -1=right -> map: left=0, right=1
```

iii. The AI noted that IBL convention is 1=left, -1=right, and mapped to the decoder's convention of left=0, right=1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL choice values (1=left, -1=right) are remapped to decoder values (0=left, 1=right). No-choice trials (choice==0) have already been excluded. The choice value is per-trial and broadcast to all 100 time bins.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1
...
np.full(N_BINS, choice_val, dtype=np.int64),
```

iii. Matches decoder task specification: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column in the trials table.

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

iii. Matches the decoder task specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values are mapped to categorical integers: 0.2->0, 0.5->1, 0.8->2. Unknown values fall back to 1. The value is per-trial and broadcast to all time bins.

ii.
```python
np.full(N_BINS, prior_val, dtype=np.int64),
```

iii. Matches decoder task specification.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel_speed(session_path):
    wheel_pos = np.load(wheel_pos_files[0])
    wheel_ts = np.load(wheel_ts_files[0])
    dt = np.diff(wheel_ts)
    dp = np.diff(wheel_pos)
    vel = dp / dt
    speed = np.abs(vel)
    vel_ts = wheel_ts[:-1] + dt / 2
    return vel_ts, speed
```

iii. The AI noted this computes speed as absolute velocity, matching the reference's `np.abs(sess_loader.wheel['velocity'])`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel velocity is computed as a simple finite difference of position divided by time, then absolute value is taken for speed. Timestamps are set to midpoints between adjacent samples. This differs from the reference code which uses `SessionLoader.load_wheel()` that: (1) interpolates position to 1000Hz uniform sampling, (2) applies an 8th-order Butterworth low-pass filter with 20Hz corner frequency, then (3) takes absolute value of the filtered velocity.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

iii. The AI acknowledged this difference in CONVERSION_NOTES.md as a "known limitation," noting the implementation is "noisier but preserves the same information."

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using quantile-based thresholds computed PER-SESSION. The quantiles [0, 33.3, 66.7, 100] are used with edges at -inf and +inf for the outer bins. `np.digitize` maps continuous values to categories 0, 1, 2.

ii.
```python
def discretize_to_bins(values_list, n_bins, compute_edges=True, edges=None):
    all_vals = np.concatenate(all_vals)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_vals, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ...
    d = np.digitize(v, edges[1:-1])
```
```python
# Called per-session:
wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
```

iii. The AI noted using quantile-based discretization for approximately equal class frequencies. Per-session computation is noted as a design choice.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to match neural bin centers using linear interpolation with extrapolation. The bin centers are computed as `np.linspace(t_start + binsize, t_end, n_bins)`, matching the reference code's interpolation grid.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, t_start, t_end, binsize, n_bins):
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. The AI noted this matches the reference code's `interp1d(..., kind='linear', fill_value='extrapolate')`.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker ME is derived from `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy` (or right camera as fallback).

ii.
```python
def load_whisker_motion_energy(session_path):
    left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
    left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
    if left_me_files and left_times_files:
        me = np.load(left_me_files[0])
        times = np.load(left_times_files[0])
```

iii. The AI noted left camera is preferred (60Hz) with right camera fallback, matching the reference code's `whisker-motion-energy` loading behavior.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly without additional processing. They are then interpolated to neural bin centers (same as wheel speed). The reference code uses `SessionLoader.load_motion_energy()` which loads the `whiskerMotionEnergy` column from an ALF object - this may involve different preprocessing than loading `ROIMotionEnergy.npy` directly.

ii.
```python
me = np.load(left_me_files[0])
times = np.load(left_times_files[0])
if len(me) == len(times):
    return times, me
elif len(me) == len(times) - 1:
    return times[:-1], me
```

iii. The AI directly loads npy files rather than using the IBL data loading pipeline.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: discretized into 3 equal-count bins using per-session quantiles.

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. Same justification as wheel speed discretization.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated to neural bin centers using `interpolate_behavior_to_bins`.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```

iii. Same justification as wheel speed alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several fallback strategies: (1) If whisker ME length is one less than timestamps, the last timestamp is dropped. (2) If a session lacks wheel or whisker data, the session is skipped entirely. (3) If behavioral interpolation fails (insufficient data points or poor coverage), the trial is skipped. (4) If interpolated data contains NaN, the trial is skipped. (5) Sessions with fewer than 2 valid trials are skipped. (6) For probe paths, the code tries alternative session numbers if '001' doesn't exist. (7) For brain region mapping, unknown regions default to 'root' (ID 0).

ii.
```python
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
...
if len(neural_trials) < 2:
    print(f"    Skipping: only {len(neural_trials)} valid trials after behavioral filtering")
    return None
```

iii. The AI documented that 20 sessions were skipped due to missing behavioral data, leaving 439 of 459 sessions.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading and binning spike data for each trial, which requires searching through large spike time arrays for each of the ~100-400 trials per session. (2) Loading npy files from disk for each probe in each session. (3) The cluster ID remapping loop `[cluster_id_to_idx[c] for c in spike_clusters]` which iterates over every spike.

ii.
```python
# Slow Python loop for remapping clusters
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```

iii. No explicit justification provided by the AI for performance choices.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Key loops that could be vectorized: (1) The cluster remapping loop iterating over all spikes: `[cluster_id_to_idx[c] for c in spike_clusters]` - could use `np.searchsorted` or a lookup array instead. (2) The trial processing loop calling `bin_spikes_trial` per trial - could be parallelized or use the reference's approach of processing all intervals at once with `bincount2D`. (3) The brain region ID mapping loop iterating over clusters.

ii.
```python
# Could be vectorized with np.searchsorted or lookup array:
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])

# Could be vectorized:
for cid in cluster_ids:
    if cid < len(all_channels):
        ch = int(all_channels[cid])
        ...
```

iii. No justification provided.

## 12-c. What processing does the code repeat multiple times?

i. (1) The `interpolate_behavior_to_bins` function is called twice per trial (once for wheel, once for whisker) with very similar logic. (2) `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)` is computed identically in both `interpolate_behavior_to_bins` (as bin_centers) and in the main loop (as `time_since_stim`), when it could be computed once. (3) The `discretize_to_bins` function collects all values and computes quantiles separately for wheel and whisker.

ii.
```python
# Computed in interpolate_behavior_to_bins:
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
# Also computed in main loop:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. No justification provided.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code loads `clusters.metrics.pqt` and extracts `cluster_labels` but never uses them for any filtering or output. (2) The code computes and stores `wheel_edges` and `whisker_edges` in the session result dict but these are not included in the final output data dictionary. (3) The code loads `clusters.depths.npy` but it is never used. (4) All clusters are included in the binning even though many have zero spikes in a given trial, creating sparse matrices where neurons with no spikes contribute empty rows.

ii.
```python
# Loaded but unused:
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
cluster_labels = metrics['label'].values  # never used for filtering

# Stored but not in final output:
'wheel_edges': wheel_edges,
'whisker_edges': whisker_edges,
```

iii. The AI acknowledged loading all clusters (without filtering) matches the reference code approach.
