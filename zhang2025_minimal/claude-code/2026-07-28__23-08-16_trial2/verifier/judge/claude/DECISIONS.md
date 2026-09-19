# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded through the ONE API (`from one.api import ONE`) with `SpikeSortingLoader` and `SessionLoader` from `brainbox.io.one`. The session list is driven by the BWM release CSV file at `/app/code/code_zhang2025/data/bwm_release.csv`, which lists probe insertions (pids) grouped by session (eid). The ONE client points to the local cache at `/app/data/one_cache`. For each session eid, trials are loaded via `SessionLoader`, spikes via `SpikeSortingLoader` (one per probe), wheel via `SessionLoader.load_wheel()`, and whisker motion energy via `SessionLoader.load_motion_energy()`.

ii.
```python
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international', silent=True,
    cache_dir=args.cache_dir
)
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
eids = bwm_df['eid'].unique()
```

iii. The agent chose to use `bwm_release.csv` from the reference code directory to enumerate sessions, rather than using `one.search()` with the Brainwidemap release tag. The agent noted this matches the reference code's approach in `0_data_caching.py`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in `bwm_release.csv`. A dictionary maps subject names to indices, and each session records its subject index.

ii.
```python
rows = bwm_df[bwm_df['eid'] == eid]
subject = rows.iloc[0]['subject']
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. Subject names come directly from the BWM release metadata CSV. The ordering is based on the order sessions are encountered, not sorted alphabetically.

## 1-c. How are the data split into sessions?

i. Sessions correspond to unique `eid` values in `bwm_release.csv`. Each eid is processed independently and contributes one entry to the output lists.

ii.
```python
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    ...
```

iii. The BWM release CSV already lists sessions by eid, so no splitting is needed.

## 1-d. How are the data split into trials?

i. Trials come from the IBL trials table loaded by `SessionLoader.load_trials()`. Each row is one trial. After filtering, each trial produces one entry in the per-session lists.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied. (1) Reaction time (firstMovement_times - stimOn_times) must be in [0.08, 2.0] seconds. (2) Six columns must not be NaN: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType. (3) No-choice trials (choice == 0) are excluded. (4) After behavioral interpolation, trials where wheel or whisker data has insufficient temporal coverage are dropped via a combined mask. Sessions with fewer than 10 valid trials are skipped entirely.

ii.
```python
nan_exclude = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]
mask = pd.Series(True, index=trials.index)
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= rt >= min_rt
mask &= rt <= max_rt
for event in nan_exclude:
    mask &= ~trials[event].isnull()
mask &= trials['choice'] != 0
```

```python
combined_mask = wheel_mask & whisker_mask
if combined_mask.sum() < 10:
    ...skip...
```

iii. The agent stated these filters match the reference code's defaults: `min_rt=0.08, max_rt=2.0`, `nan_exclude='default'`, and `exclude_nochoice=True`. The minimum 10-trial threshold was chosen by the agent to ensure sufficient data per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times` and `spikes.clusters` loaded by `SpikeSortingLoader`. The cluster table (merged with channel information) provides brain region labels.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. Standard IBL spike sorting data, loaded via the ibllib API.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over a [-0.5, 1.5] s window around stimulus onset, giving spike counts per cluster per bin. Counts are NOT converted to firing rates (not divided by bin width). When a session has multiple probes, clusters are merged by offsetting cluster IDs and sorting by spike time.

ii.
```python
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. The agent followed the reference code's binning approach but did not divide by bin width to convert counts to rates. The reference code divides by BIN (0.02) to produce Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. NO quality control filtering is applied to neurons. All spike-sorted clusters are kept, regardless of their quality label. The agent explicitly decided not to filter on `label >= 1`.

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

iii. The agent justified this by reading the reference code's `prepare_data` function, which calls `load_spiking_data` without a `qc` argument, defaulting to `qc=None` which returns all clusters. The agent noted that the BWM paper's filtering happens at the analysis level, not during data loading. However, the agent did not apply the `void` region filter either.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are selected in the window [stimOn - 0.5s, stimOn + 1.5s] using `searchsorted`, then binned relative to the window start.

ii.
```python
t_start = align_times[trial_idx] + window[0]
t_end = align_times[trial_idx] + window[1]
i_start = np.searchsorted(sorted_times, t_start, side='left')
i_end = np.searchsorted(sorted_times, t_end, side='left')
times_in_window = sorted_times[i_start:i_end]
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```

iii. The agent aligned to stimulus onset as specified in the instructions and matching the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins for the 2-second window. No rebinning or smoothing is applied.

ii.
```python
WINDOW = (-0.5, 1.5)
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
```

iii. The agent set the bin size to 20 ms matching the reference code's `'binsize': 0.02` and the methods paper description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a fixed time vector computed from the window parameters, not derived from any raw data variable per trial. The alignment event is `stimOn_times`.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The agent used the reference code's `get_behavior_per_interval` formula for bin centers.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time vector is computed as `np.linspace(-0.48, 1.5, 100)`, placing 100 evenly-spaced points from -0.48 to 1.50 seconds. This differs from the reference's `EDGES[:-1] + BIN/2` which produces centers at [-0.49, -0.47, ..., 1.49].

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
# = np.linspace(-0.48, 1.5, 100)
# First bin center: -0.48, Last bin center: 1.50
```

iii. The agent cited the reference code's `get_behavior_per_interval` as the source of this formula. However, it produces slightly different values from the neural bin centers (which are at edges + bin/2 = [-0.49, ..., 1.49]).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses `np.linspace(-0.48, 1.5, 100)` while the neural bins have centers at `T_START + BIN/2 + BIN*i` = [-0.49, ..., 1.49]. These are offset by 0.01 seconds, so the time input is not perfectly aligned with the neural bin centers.

ii. Neural bin assignment:
```python
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```
Time input:
```python
time_since_stim = np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS)
```

iii. The agent did not explicitly discuss this misalignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block boundary is detected when `probabilityLeft` changes from one trial to the next.

ii.
```python
pLeft = trials_df['probabilityLeft'].values
for i in range(len(pLeft)):
    if i == 0 or pLeft[i] != pLeft[i-1]:
        block_counter = 1
    trial_num[i] = block_counter
    block_counter += 1
```

iii. The agent identified blocks from changes in probabilityLeft, same approach as the reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number within each block is 1-indexed (starts at 1), computed over the full unmasked trial sequence before filtering, then indexed into masked trials. The reference uses 0-indexed counting (cumcount starts at 0).

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

iii. The agent computed the block number over the full trial sequence before filtering, so dropped trials still advance the counter. The 1-indexing vs 0-indexing is a minor difference from the reference.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which uses IBL convention: +1 = left, -1 = right, 0 = no response.

ii.
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. The agent stated that IBL choice=1 is right and choice=-1 is left, but the IBL convention is actually the opposite: choice=1 is LEFT and choice=-1 is RIGHT. This led to an inverted mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps IBL choice=1 to decoder value 1 and IBL choice=-1 to decoder value 0, using `np.where(choice_vals == 1, 1, 0)`. Because IBL choice=1 is actually LEFT (not right as the agent assumed), this produces left=1, right=0, which is the opposite of the instructions (left=0, right=1).

ii.
```python
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

The reference maps correctly:
```python
CHOICE = {1.0: 0, -1.0: 1}  # +1 is a leftward choice, -1 rightward
```

iii. The agent misidentified the IBL convention for choice direction, resulting in an inverted choice mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

iii. Mapping matches the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct recoding of the three probability values to integers 0, 1, 2. No further processing.

ii. Same as 6-a.

iii. Straightforward mapping as specified.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `SessionLoader.load_wheel()`, which provides wheel position and timestamps. The velocity is computed internally by SessionLoader (interpolation to 1000 Hz, Butterworth low-pass filter, differentiation). Speed is taken as the absolute value of velocity.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. Matches the reference approach of using absolute velocity from SessionLoader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The absolute velocity is interpolated to bin centers using `scipy.interpolate.interp1d` with linear interpolation. Coverage checks flag trials with insufficient data. The continuous values are then discretized into 3 bins using GLOBAL tercile thresholds (percentiles computed across ALL sessions combined).

ii.
```python
# Interpolation
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

```python
# Global discretization
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The agent chose global discretization to enable consistent bin edges across sessions. The reference uses per-session percentiles instead.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tercile thresholds are computed from all wheel speed values across all sessions. Values are digitized using `np.digitize` with edges at the 33.3rd and 66.7th percentiles, producing categories 0, 1, 2.

ii.
```python
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The agent explicitly designed a two-pass architecture to support global discretization.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to bin centers at `np.linspace(t_start + binsize, t_end, n_bins)` = `np.linspace(-0.48, 1.5, 100)`. This is the same grid used for the time input, but differs from the neural bin centers by 0.01 seconds.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The agent used the same behavioral interpolation grid for all behavioral variables.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` (or `rightCamera.ROIMotionEnergy` as fallback) loaded via `SessionLoader.load_motion_energy()`. The `whiskerMotionEnergy` column is used.

ii.
```python
sl.load_motion_energy(views=['left'])
me_times = sl.motion_energy['leftCamera']['times'].values
me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
```

iii. Left camera is preferred, with right camera as fallback, matching the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are interpolated to the same bin centers as wheel speed using `interp1d`. Then discretized into 3 bins using GLOBAL tercile thresholds (across all sessions).

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
# Global discretization
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. Same interpolation and global discretization approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: global tercile thresholds at 33.3rd and 66.7th percentiles across all sessions, producing categories 0, 1, 2 via `np.digitize`.

ii.
```python
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. Global discretization chosen for consistency.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Interpolated to the same bin centers as wheel speed: `np.linspace(-0.48, 1.5, 100)`, which is offset from the neural bin centers by 0.01 seconds.

ii. Same `interpolate_behavior_to_bins` function as wheel speed.

iii. The agent used a consistent behavioral interpolation grid but did not match it exactly to the neural bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Failed probe loads are warned but remaining probes are used. (2) Sessions with no loaded probes are skipped. (3) Sessions without whisker motion energy are skipped. (4) Trials with NaN in key fields are masked out. (5) Trials with insufficient behavioral data coverage are masked out via the combined wheel/whisker mask. (6) Sessions with fewer than 10 valid trials after all filtering are skipped. (7) Any session-level exception is caught, logged, and the session is skipped.

ii.
```python
except Exception as e:
    print(f'  Warning: Failed to load probe {pid}: {e}')
...
if combined_mask.sum() < 10:
    print(f'  Skipping: only {combined_mask.sum()} trials with valid behavior')
    skipped_sessions += 1
    continue
```

iii. The agent handled missing data by dropping at various levels (trial, probe, session).

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk via `SpikeSortingLoader.load_spike_sorting()` is the most expensive step, as the spike arrays can be hundreds of megabytes per probe. The spike binning loop was also initially very slow until the agent rewrote it with vectorized numpy operations.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The agent noted the data loading from disk (parquet/numpy files) was the dominant bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `bin_spikes_in_window` iterates over trials sequentially. This could potentially be vectorized by computing all trial windows at once. The per-trial loop in `interpolate_behavior_to_bins` could also be vectorized.

ii.
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
    binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The agent initially had a much slower spike-by-spike loop and rewrote it to be vectorized within each trial. Full cross-trial vectorization was not attempted.

## 10-c. What processing does the code repeat multiple times?

i. The code loads trials via `SessionLoader` twice for the same session: once in `load_trials_and_mask` and once implicitly via the SessionLoader in the main loop (the same SessionLoader object is reused for wheel and motion energy, but trials are loaded separately). The code also constructs `interp1d` objects per trial for both wheel and whisker, which is redundant work.

ii.
```python
trials, mask, sl = load_trials_and_mask(one, eid)  # loads trials
sl.load_wheel()  # reuses same SessionLoader
```

iii. The double-pass architecture (first pass collects raw data, second pass discretizes) requires storing all behavioral data in memory across sessions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `feedback_times` and `feedbackType` and checks them for NaN as part of trial filtering, but these variables are not used in the decoder task. The code also stores extra metadata (lab names, full session info) that isn't required by the target format.

ii.
```python
nan_exclude = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]
```

iii. The agent included these extra NaN checks to match the reference code's default NaN exclusion list.
