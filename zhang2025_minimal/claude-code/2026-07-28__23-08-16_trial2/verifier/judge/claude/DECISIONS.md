# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API to connect to the IBL cache, but instead of using `one.search()` to discover sessions from the release index, it reads a pre-existing CSV file (`bwm_release.csv`) from the reference code repository. This CSV contains probe insertion records with eid, pid, subject, lab, and probe_name columns. The AI iterates over unique eids from this CSV and uses `SpikeSortingLoader` and `SessionLoader` to load spike sorting and trial/behavioral data for each session.

ii.
```python
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international', silent=True,
    cache_dir=args.cache_dir
)
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
# ...
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    # ...
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The AI used the BWM release CSV to identify sessions and their probe insertions, rather than querying the ONE API cache index directly. The CONVERSION_NOTES state the data source is "IBL Brain Wide Map (BWM) dataset" with "699 probe insertions across 459 sessions from 139 subjects."

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the `bwm_release.csv` DataFrame. A `subject_to_idx` dictionary maps subject names to integer indices as they are encountered. At the end, `subjects_list` contains all unique subjects in order of first appearance.

ii.
```python
subject = rows.iloc[0]['subject']
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. The subject name comes directly from the BWM release table. No parsing of paths or filenames is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the BWM release CSV corresponds to one session. The AI iterates over unique eids, so sessions are already the natural unit of iteration.

ii.
```python
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
```

iii. Sessions are identified by their eid in the BWM release table. No splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table loaded by `SessionLoader` has one row per trial. After filtering, each trial is processed individually to build per-trial neural, input, and output arrays.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
# ...
for trial in range(n_trials):
    session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. The trials table naturally provides one row per trial. No additional splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) Reaction time must be between 0.08s and 2.0s. (2) No-choice trials (choice == 0) are excluded. (3) Trials with NaN in key event columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType) are excluded. (4) Trials where wheel or whisker motion energy data doesn't cover the trial window are excluded via a behavior coverage check. Additionally, sessions with fewer than 10 valid trials are skipped.

ii.
```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= rt >= min_rt   # 0.08
mask &= rt <= max_rt   # 2.0
for event in nan_exclude:
    mask &= ~trials[event].isnull()
mask &= trials['choice'] != 0
# ...
# Behavior coverage check
combined_mask = wheel_mask & whisker_mask
if combined_mask.sum() < 10:
    # skip session
```

iii. The CONVERSION_NOTES state: "Exclude trials with RT < 0.08s or > 2.0s, NaN key events, no-choice." The AI also notes matching the reference code's `load_trials_and_mask` defaults.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader.load_spike_sorting()`. The cluster table is used for brain region mapping but not for quality filtering of clusters.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. The CONVERSION_NOTES state spike sorting data is loaded from each probe insertion using the ONE API's SpikeSortingLoader.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over the trial window (-0.5s to 1.5s around stimulus onset), giving 100 bins per trial. The raw spike counts are stored directly as float32 arrays. No division by the bin width is performed (i.e., the data is in spike counts, not firing rates in Hz). When a session has multiple probes, they are merged with cluster indices offset to avoid collisions.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
# ...
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
# ...
# Stored directly without dividing by BIN:
session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. The AI's code bins spikes into 20ms bins matching the reference parameters but does not convert spike counts to firing rates (Hz) by dividing by the bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. NO quality control filtering is applied to clusters. All spike-sorted clusters are included, regardless of their quality label. The AI explicitly chose to skip QC filtering, arguing that the reference code's `prepare_data` function uses `qc=None` by default.

ii.
```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Load spike sorting data for a probe insertion. Uses all clusters (no QC filter)
    matching the reference code's default behavior in prepare_data."""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

iii. The CONVERSION_NOTES state: "Use ALL spike-sorted clusters (no quality control filtering)... The reference code's `prepare_data` function calls `load_spiking_data` without the `qc` parameter, defaulting to `qc=None` which returns all clusters." The AI reports 599,865 total neurons vs the paper's 75,708 well-isolated neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset (`stimOn_times`). Spike times within the window [stimOn - 0.5s, stimOn + 1.5s] are selected and binned relative to the trial start.

ii.
```python
align_times = masked_trials['stimOn_times'].values
# ...
t_start = align_times[trial_idx] + window[0]   # stimOn - 0.5
t_end = align_times[trial_idx] + window[1]      # stimOn + 1.5
times_in_window = sorted_times[i_start:i_end]
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```

iii. The AI aligns to stimulus onset as specified in the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, producing 100 time bins per trial over the 2s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
```

iii. The AI matches the reference code's `binsize=0.02` and the methods paper's description of "20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time-since-stimulus-onset input is a synthetic variable derived from the window parameters. It is not directly from any raw data variable, but is defined by the window boundaries and bin size.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI constructs the time input as evenly spaced values from -0.48 to 1.5 using `np.linspace`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time input is computed as `np.linspace(-0.48, 1.5, 100)`, which gives 100 evenly spaced values from -0.48 to 1.5 inclusive. This differs from the bin-center approach (which would give values from -0.49 to 1.49).

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The CONVERSION_NOTES state: "Time since stimulus onset: Continuous time variable, same for all trials (linspace from -0.48 to 1.5, matching bin centers)." However, these are not true bin centers; they are shifted by +0.01 relative to the bin centers used in the reference code.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same number of bins (100) as the neural data, so they have matching dimensions. However, the time values are offset by 0.01s from the true bin centers used for neural data binning. Neural bins have centers at [-0.49, -0.47, ..., 1.49], while the time input values are [-0.48, -0.46, ..., 1.50].

ii.
```python
# Neural binning uses:
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
# Time input uses:
time_since_stim = np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS)
```

iii. The AI intended to match the reference code's bin centers but used `np.linspace(start + binsize, end, n_bins)` instead of `EDGES[:-1] + BIN/2`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column in the trials table. A block boundary is detected where `probabilityLeft` changes value between consecutive trials.

ii.
```python
pLeft = trials_df['probabilityLeft'].values
for i in range(len(pLeft)):
    if i == 0 or pLeft[i] != pLeft[i-1]:
        block_counter = 1
    trial_num[i] = block_counter
    block_counter += 1
```

iii. The AI identifies block boundaries from changes in `probabilityLeft`, matching the reference approach.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts the trial's position within each block starting from 1 (1-indexed). Block boundaries are detected when `probabilityLeft` changes. The count is computed on all trials before filtering, so filtered-out trials still advance the count. The result is then indexed by the trial mask.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    pLeft = trials_df['probabilityLeft'].values
    trial_num = np.zeros(len(pLeft), dtype=np.float32)
    block_counter = 1
    for i in range(len(pLeft)):
        if i == 0 or pLeft[i] != pLeft[i-1]:
            block_counter = 1
        trial_num[i] = block_counter
        block_counter += 1
    return trial_num[mask]
```

iii. The AI computes block membership before filtering, which preserves the real trial position. The count starts from 1, unlike the reference which uses 0-based counting via `cumcount()`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column in the trials table, which has values +1 (left), -1 (right), and 0 (no-go).

ii.
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. The CONVERSION_NOTES state: "In IBL data, choice=-1 is left, choice=1 is right." However, this is incorrect -- in IBL, choice=+1 is left and choice=-1 is right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == 1` to 1 and everything else (i.e., choice == -1) to 0. Given the actual IBL convention (choice=+1 is left, choice=-1 is right), this produces left=1, right=0, which is the OPPOSITE of the instructions (left=0, right=1).

ii.
```python
# Choice: left=-1 -> 0, right=1 -> 1 (as per decoder spec)
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. The AI's comment claims "left=-1 -> 0, right=1 -> 1" which misidentifies the IBL convention. The actual IBL convention is +1=left, -1=right, so the mapping is inverted relative to the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

iii. The AI correctly maps 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping of the three probability values to integer categories. No additional processing.

ii.
```python
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

iii. Matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from the wheel position and timestamps loaded by `SessionLoader.load_wheel()`. The speed is the absolute value of the velocity computed by the SessionLoader.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. The CONVERSION_NOTES state: "Use absolute velocity (speed = |velocity|) from SessionLoader.load_wheel()", matching the reference approach.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` internally interpolates the raw wheel position onto a 1000 Hz grid and applies a Butterworth low-pass filter to compute velocity. The speed (absolute velocity) is then interpolated onto the trial time grid using `interp1d` with linear interpolation and extrapolation.

ii.
```python
wheel_speed = np.abs(sl.wheel['velocity'].values)
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
# In interpolate_behavior_to_bins:
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The AI uses `scipy.interpolate.interp1d` with `fill_value='extrapolate'` instead of `np.interp`, and evaluates at `np.linspace(t_start + binsize, t_end, n_bins)` instead of bin centers.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The wheel speed is discretized into 3 bins using GLOBAL tercile thresholds computed across ALL sessions. The percentile edges at 33.3% and 66.7% are computed from the concatenated wheel speed values of all sessions, then applied uniformly.

ii.
```python
# Global thresholds:
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
# Applied per session:
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The CONVERSION_NOTES state: "Discretization: 3 bins using global tercile thresholds." This differs from the reference which uses per-session percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to the same number of time bins (100) as the neural data, evaluated at `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The behavioral data is interpolated onto a time grid with the same number of points as the neural bins, but the grid points are offset from the true bin centers by 0.01s.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from the whisker ROI motion energy from a side camera (`leftCamera.ROIMotionEnergy` or `rightCamera.ROIMotionEnergy`) and its frame timestamps. The left camera is preferred; right is used as fallback.

ii.
```python
try:
    sl.load_motion_energy(views=['left'])
    me_times = sl.motion_energy['leftCamera']['times'].values
    me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
except Exception:
    sl.load_motion_energy(views=['right'])
    me_times = sl.motion_energy['rightCamera']['times'].values
    me_values = sl.motion_energy['rightCamera']['whiskerMotionEnergy'].values
```

iii. The CONVERSION_NOTES state the AI tries left camera first, falls back to right, matching the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used without additional filtering or normalization. It is interpolated to the trial time grid using `interp1d` with linear interpolation.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. Same interpolation approach as wheel speed, using `interp1d` with `fill_value='extrapolate'`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 bins using GLOBAL tercile thresholds computed across all sessions.

ii.
```python
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
# ...
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. Global discretization thresholds rather than per-session percentiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same alignment approach as wheel speed: interpolated to 100 time points using `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
```

iii. Same interpolation grid as wheel speed, with the same 0.01s offset from true bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Failed probe loads are caught with try/except and skipped with a warning. (2) Sessions with no loaded probes are skipped. (3) Sessions with fewer than 10 valid trials are skipped (stricter than reference's threshold of 2). (4) Sessions without whisker motion energy are skipped. (5) Trials where behavioral data doesn't cover the window are excluded. (6) NaN values in behavioral data cause trials to be marked as invalid.

ii.
```python
# Failed probe load
except Exception as e:
    print(f'  Warning: Failed to load probe {pid}: {e}')
# No probes
if len(spikes_list) == 0:
    skipped_sessions += 1; continue
# Too few trials
if combined_mask.sum() < 10:
    skipped_sessions += 1; continue
# No whisker data
if whisker_binned is None:
    skipped_sessions += 1; continue
# Behavior coverage
if np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
```

iii. The AI handles missing data by dropping affected trials or sessions. The minimum trial threshold of 10 is stricter than the reference's threshold of 2.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk is the main bottleneck, as noted by the AI during development. Each session's spike data can be hundreds of megabytes. The AI noted processing took ~30s per session, mostly from data loading.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The trajectory shows the AI observing "the data loading (from disk via ONE) is the main bottleneck, not the binning."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning function `bin_spikes_in_window` loops over trials, and `interpolate_behavior_to_bins` also loops over trials. Both could potentially be vectorized by offsetting indices across trials. The cluster-to-index mapping also uses a Python loop.

ii.
```python
for trial_idx in range(n_trials):
    # spike binning per trial
    ...
for trial_idx in range(n_trials):
    # behavior interpolation per trial
    ...
for idx, cid in enumerate(cluster_ids):
    cluster_to_idx[cid] = idx
```

iii. The AI attempted to optimize the spike binning with vectorized numpy operations within each trial but kept the per-trial loop.

## 10-c. What processing does the code repeat multiple times?

i. The code creates a new `interp1d` function object for every trial in `interpolate_behavior_to_bins`, even though the behavioral time series is continuous across all trials in a session. A single interpolation function could be created once per session.

ii.
```python
for trial_idx in range(n_trials):
    # ...
    f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
    binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. Each trial creates its own interpolation function from a subset of the behavioral data, repeating the function construction.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads all spike attributes via `load_spike_sorting()` without restricting to only the needed arrays ('clusters' and 'times'). The reference code overrides `bio.SPIKES_ATTRIBUTES = ['clusters', 'times']` to avoid loading unnecessary spike data (like 'amps', 'depths'). Additionally, `SpikeSortingLoader.merge_clusters()` computes and attaches metrics and histology data to every cluster, most of which is unused beyond the brain region acronym.

ii.
```python
# AI loads all default attributes:
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()

# Reference restricts to needed attributes:
import brainbox.io.one as bio
bio.SPIKES_ATTRIBUTES = ['clusters', 'times']
```

iii. Loading unnecessary spike attributes increases I/O time and memory usage.
