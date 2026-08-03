# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`bwm_release.csv`) from the reference code directory to get session metadata (eid, lab, subject, date, probe_name), then constructs file system paths to locate the data files directly on disk. It does NOT use the ONE API. Instead, it navigates the ONE cache directory structure (`lab/Subjects/subject/date/001/`) and loads `.npy` and `.pqt` files directly using `np.load` and `pd.read_parquet`.

ii.
```python
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
# ...
session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
# ...
trials = pd.read_parquet(trials_files[0])
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. The AI chose to read files directly from the ONE cache directory structure rather than using the ONE API, because the data was already cached locally. It used `bwm_release.csv` from the reference code to identify which sessions belong to the BWM release.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the `bwm_release.csv` file. Sessions are grouped by eid, and each session carries its subject name. At assembly, a unique ordered list of subjects is built incrementally as sessions are processed.

ii.
```python
subject = session_info['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
```

iii. The subject information is directly available from the BWM release CSV, so no parsing is required.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the `bwm_release.csv` defines a session. Multiple probes within the same session are grouped under the same eid, with probe names collected in a list.

ii.
```python
if eid not in sessions:
    sessions[eid] = {
        'path': session_path,
        'probes': [],
        'subject': subject,
        # ...
    }
sessions[eid]['probes'].append(probe_name)
```

iii. Sessions are naturally identified by their eid in the BWM release metadata.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file, which has one row per trial. Each row contains trial-level variables (stimOn_times, choice, probabilityLeft, etc.).

ii.
```python
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
trials = pd.read_parquet(trials_files[0])
```

iii. The trials table naturally provides one row per trial; no further splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters: (1) Exclude trials with NaN values in required event columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType). (2) Exclude trials with reaction time outside 0.08-2.0s. (3) Exclude no-choice trials (choice == 0). Additionally, trials where wheel speed or whisker motion energy cannot be interpolated (insufficient coverage or NaN values) are dropped later during per-trial processing.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)

# Later, during per-trial processing:
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. The AI stated this matches `load_trials_and_mask()` from the reference code. The NaN exclusion for `feedback_times` and `feedbackType` goes beyond the reference solution which only checks reaction time bounds, choice validity, and probabilityLeft validity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Also `clusters.channels.npy` for channel mapping, `channels.brainLocationIds_ccf_2017.npy` for brain regions, and `clusters.metrics.pqt` for quality labels.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
```

iii. These are the standard IBL spike sorting outputs needed to build neural activity matrices.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over a 2s window (-0.5 to 1.5s relative to stimulus onset), producing a (n_clusters, 100) count matrix per trial. The counts are stored as float32 but are NOT converted to firing rates (not divided by bin width). When a session has multiple probes, clusters are merged with an offset to create a continuous numbering.

ii.
```python
binned = np.zeros((n_clusters, n_bins), dtype=np.float32)
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
```

iii. The AI loads all clusters without quality filtering, stating this matches the reference caching code which calls `load_spiking_data` without a `qc` parameter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads quality labels from `clusters.metrics.pqt` but does NOT actually filter clusters based on quality. All clusters are included regardless of their quality label. The code loads `cluster_labels` but never uses it to filter spikes.

ii.
```python
metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
cluster_labels = None
if metrics_files:
    metrics = pd.read_parquet(metrics_files[0])
    cluster_labels = metrics['label'].values
# cluster_labels is passed around but never used for filtering
```

iii. The AI explicitly decided to load ALL clusters, stating: "Load ALL clusters (not filtered by quality) per the reference caching code" and "the reference code `prepare_data()` calls `load_spiking_data(one, pid)` without the `qc` parameter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset (`stimOn_times`). The bin window runs from `stim_on - 0.5` to `stim_on + 1.5` in absolute session time. Spikes within this window are selected via `searchsorted`, then binned relative to the window start.

ii.
```python
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
i_start = np.searchsorted(spike_times, t_start, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
```

iii. Alignment to stimulus onset is specified in both the instructions and the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, yielding 100 bins over a 2s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. This matches the reference code's `'binsize': 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time values are computed as bin centers relative to stimulus onset.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The instructions specify alignment to stimulus onset, and the time input represents the elapsed time from that event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes bin centers using `np.linspace(-0.48, 1.5, 100)`, which produces values from -0.48 to 1.50 with step 0.02. This differs from the reference which computes bin centers as `EDGES[:-1] + BIN/2`, yielding values from -0.49 to 1.49.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
# Produces: -0.48, -0.46, ..., 1.48, 1.50
```

Reference code produces:
```python
EDGES = T_START + BIN * np.arange(N_BINS + 1)  # [-0.50, -0.48, ..., 1.50]
TIME = EDGES[:-1] + BIN / 2                     # [-0.49, -0.47, ..., 1.49]
```

iii. The AI used `linspace` to generate bin centers but the formula produces right edges of bins rather than true centers, resulting in a 0.01s (half-bin) offset.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same number of bins (100) as the neural data, but the bin center values are offset by half a bin compared to the neural binning grid, because the neural data bins spikes starting from `t_start` while the time input starts from `t_start + BINSIZE`.

ii.
```python
# Neural binning uses:
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)  # bins from t_start

# Time input uses:
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
# First bin center at -0.48, but neural first bin covers [-0.50, -0.48)
```

iii. There is a misalignment: the neural bin centers should be at -0.49, -0.47, ..., but the time input provides -0.48, -0.46, ..., a half-bin shift.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
prob_left = trials_masked['probabilityLeft'].values
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
```

iii. The trials table has no explicit block identifier, so blocks must be inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected from changes in `probabilityLeft`, then trials are numbered within each block starting from 0. However, the AI computes trial_in_block AFTER applying the quality mask, meaning only surviving trials are counted. Dropped trials do not advance the counter, so the block position does not reflect the animal's real position in the block.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    trials_masked = trials_df[mask].copy()  # Only masked trials
    prob_left = trials_masked['probabilityLeft'].values
    block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
    trial_in_block = np.zeros(len(prob_left), dtype=int)
    for i in range(len(block_changes)):
        start = block_changes[i]
        end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
        trial_in_block[start:end] = np.arange(end - start)
    return trial_in_block
```

iii. The AI computes this after filtering, while the reference computes it before filtering so the count reflects the animal's actual position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1  # IBL: 1=left, -1=right -> map: left=0, right=1
```

iii. The IBL convention is +1 for left and -1 for right; the decoder task requires left=0, right=1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple recoding: +1 -> 0 (left), -1 -> 1 (right). No-response trials (choice==0) are already excluded by the trial mask.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. Straightforward mapping as specified in the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, or 0.8.

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

iii. Direct mapping as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Simple recoding of 0.2->0, 0.5->1, 0.8->2. There is a fallback to 1 for unexpected values, though such values should have been filtered by the trial mask.

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

iii. Matches the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, which provide the wheel position and corresponding timestamps.

ii.
```python
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. These are the raw wheel recording files from the IBL dataset.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes velocity using simple finite differences of the position signal, then takes the absolute value for speed. This does NOT use the IBL's `SessionLoader.load_wheel()` which applies interpolation to a 1000 Hz grid and a 20 Hz Butterworth low-pass filter before computing velocity.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

iii. The AI acknowledged this as a known limitation: "The reference code uses `SessionLoader.load_wheel()` which applies Gaussian smoothing. Our implementation uses simple finite differences, which is noisier but preserves the same information."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 equal-count bins using quantile-based thresholds computed per session. The quantile edges are computed from all valid trial values within the session, and `np.digitize` is used with the inner edges.

ii.
```python
def discretize_to_bins(values_list, n_bins, compute_edges=True, edges=None):
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_vals, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    d = np.digitize(v, edges[1:-1])  # Returns 0 to n_bins-1
```

iii. Per-session quantile-based discretization ensures approximately equal class frequencies.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to the same bin centers as the neural data using `scipy.interpolate.interp1d` with linear interpolation and extrapolation. However, the bin centers used are offset by half a bin from the reference (same issue as the time input).

ii.
```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
return f(bin_centers)
```

iii. The interpolation approach is consistent with the reference code, but the bin centers are shifted by 0.01s.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` with corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
# Falls back to right camera if left not available
```

iii. Left camera is preferred, matching the reference code's camera selection logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are used as-is (no filtering or normalization). They are interpolated to the neural bin centers using `interp1d` with linear interpolation, then discretized into 3 bins using per-session quantiles.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: discretized into 3 equal-count bins using per-session quantiles with `np.digitize`.

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to the same bin centers (though these are shifted by half a bin from the reference).

ii.
```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
return f(bin_centers)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms: (1) Missing probe data (no pykilosort directory or files) causes the probe to be skipped. (2) Missing wheel or whisker data causes the session to be skipped. (3) Trials where behavioral signals cannot be interpolated (insufficient coverage) or contain NaN are dropped. (4) Sessions with fewer than 2 valid trials are skipped. (5) ME array length mismatches are handled by trimming.

ii.
```python
if not probe_path.exists():
    return None, None, None, None, None
if wheel_speed is None:
    return None
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
if len(neural_trials) < 2:
    return None
# ME length adjustment:
if len(me) == len(times) - 1:
    return times[:-1], me
```

iii. Missing data is handled by dropping the affected trial or session.

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting data from disk (large `.npy` files with spike times and clusters), and the per-trial spike binning loop which iterates over all trials and uses `np.add.at` for each trial.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

Also the cluster remapping loop:
```python
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```

iii. File I/O dominates, plus the Python loop for cluster remapping over potentially millions of spikes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The cluster remapping loop iterates over every spike to look up a dictionary, which could be vectorized using numpy indexing. The per-trial spike binning loop could potentially be vectorized. The brain region lookup loop iterates per cluster.

ii.
```python
# This Python loop over all spikes is very slow:
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])

# Per-cluster brain region lookup:
for cid in cluster_ids:
    if cid < len(all_channels):
        ch = int(all_channels[cid])
        # ...
```

iii. The cluster remapping loop is particularly problematic as it runs a Python dictionary lookup for each of potentially millions of spikes.

## 10-c. What processing does the code repeat multiple times?

i. The `rglob` file searches are called multiple times for different file types within the same session path. The brain region mapping is done per cluster in a loop rather than vectorized.

ii.
```python
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
```

iii. Each `rglob` call traverses the directory tree independently.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `clusters.depths.npy` but never uses it. It loads `feedbackType` and `feedback_times` for NaN checking, but these are not used in the decoder. It loads cluster labels but never uses them for filtering.

ii.
```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')  # Never used
# NaN checking includes unused columns:
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
# Labels loaded but not used:
cluster_labels = metrics['label'].values  # Passed around but never filters anything
```

iii. Some data is loaded unnecessarily, adding I/O overhead without contributing to the output.
