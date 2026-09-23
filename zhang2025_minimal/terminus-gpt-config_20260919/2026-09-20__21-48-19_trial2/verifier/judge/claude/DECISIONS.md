# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV release manifest (`bwm_release.csv`) from the reference code repository to enumerate all sessions and their probe insertions (PIDs). When a `DATALIMIT_SUBSET.csv` file exists, sessions are filtered to that subset. A ONE client is created with `mode='local'` pointing at the local cache. Spike sorting is loaded via `SpikeSortingLoader`, trials via a custom local loader that reads parquet files directly, and wheel/whisker via `SessionLoader` and manual file globbing. The reference code's `merge_probes` and `bin_spiking_data` utilities are imported and used directly.

ii.
```python
bwm = pd.read_csv(RELEASE, index_col=0)
subset_file = ROOT/'data/DATALIMIT_SUBSET.csv'
if subset_file.exists():
    subset = pd.read_csv(subset_file)
    values = set(subset.astype(str).to_numpy().ravel())
    bwm = bwm[bwm.eid.astype(str).isin(values) | bwm.pid.astype(str).isin(values)]
...
one = ONE(cache_dir=CACHE, mode='local', silent=True)
```

iii. The agent initially tried to use ONE with remote Alyx authentication but network access was blocked. The agent then discovered data was pre-staged locally and switched to local mode. The CSV-based session selection was chosen because the release table lists all sessions with their probe information, allowing straightforward enumeration.

## 1-b. How are the data split into subjects?

i. The subject name is read from the `bwm_release.csv` table, which lists subject alongside each session's eid. Sessions are grouped by eid, and the subject from the first row of each group is used.

ii.
```python
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid),
                 list(group.pid.astype(str)), list(group.probe_name.astype(str))))
```

iii. The release CSV already contains subject information for each session, so no parsing of paths is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV represents one session. Sessions are processed individually.

ii.
```python
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid), ...))
```

iii. The release CSV is organized by eid (session identifier), so the split is given by the data.

## 1-d. How are the data split into trials?

i. The trials table (loaded from parquet) has one row per trial. The reference utility `bin_spiking_data` is used to bin spikes per trial, aligning to `stimOn_times` with a window of (-0.5, 1.5) seconds. Each trial becomes a separate array.

ii.
```python
trials = load_local_trials(one, eid)
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. The trials table naturally provides one row per trial. `bin_spiking_data` handles the per-trial binning.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a `paper_trial_mask` that requires: (1) non-NaN values for stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, and feedbackType; (2) reaction time (firstMovement_times - stimOn_times) between 0.08 and 2.0 seconds; (3) trial duration (feedback_times - goCue_times) <= 10 seconds; (4) choice != 0 (no-response excluded). Additionally, trials must have valid wheel and whisker coverage within the trial window.

ii.
```python
def paper_trial_mask(trials):
    required = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                'firstMovement_times', 'feedbackType']
    valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
    rt = trials.firstMovement_times.to_numpy() - trials.stimOn_times.to_numpy()
    duration = trials.feedback_times.to_numpy() - trials.goCue_times.to_numpy()
    valid &= (rt >= 0.08) & (rt <= 2.0)
    valid &= duration <= 10.0
    valid &= trials.choice.to_numpy() != 0
    return valid
```

iii. The agent reproduced the paper's trial QC criteria as found in the reference code. The duration filter (feedback - goCue <= 10s) and broader NaN checks are additional compared to the human reference, which only checks reaction time, choice membership, probabilityLeft membership, and behavioral coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe insertion, loaded via `SpikeSortingLoader`. Cluster metadata comes from `SpikeSortingLoader.merge_clusters`.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
spikes['clusters'] = np.array(spikes['clusters'], dtype=np.int32, copy=True)
clusters = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
```

iii. The spike sorting loader provides the raw spike times and cluster assignments needed for binning.

## 2-b. How is the `neural` data processed?

i. For sessions with multiple probes, the reference code's `merge_probes` utility is used to merge spikes/clusters across probes. Then `bin_spiking_data` bins spikes into 20 ms non-overlapping bins from -0.5 to 1.5 s around stimulus onset, producing 100 time bins per trial. The output is stored as spike counts (int16), NOT firing rates.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
```

iii. The agent used the reference code's own utility functions for probe merging and spike binning to match the reference processing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ALL clusters are included without quality filtering. The AI passes `cluster_ids = np.arange(len(clusters))` to `bin_spiking_data`, which includes every Kilosort 2.5 cluster regardless of quality label. Brain regions are mapped to Beryl but no `void` or other filtering is applied to exclude clusters.

ii.
```python
cluster_ids = np.arange(len(clusters), dtype=np.int64)
neural_df = {'spike_times': spikes['times'], 'spike_clusters': spikes['clusters']}
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. The agent noted that the reference code uses `qc=None` (all clusters) in `load_spiking_data`, and explicitly stated the metadata as "all clusters; probes in the same session merged".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to stimulus onset (`stimOn_times`). The `bin_spiking_data` utility takes `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)` to extract per-trial spike data aligned to stimulus onset.

ii.
```python
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. The reference code's `bin_spiking_data` handles the alignment internally using the specified `align_time` parameter.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial spanning the 2-second window. No rebinning or smoothing is applied; spikes are directly counted into these bins.

ii.
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
PARAMS = dict(interval_len=2, binsize=BIN, ...)
```

iii. The 20 ms bin size matches the reference code's `binsize: 0.02` parameter.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a synthetic time array derived from the bin parameters, not from any raw data variable. It represents the right edges of the 100 time bins.

ii.
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
```

iii. The time array is defined by the binning parameters and represents when each bin ends relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as bin right edges: `np.arange(1, 101) * 0.02 + (-0.5)`, giving values from -0.48 to 1.50 in steps of 0.02. This differs from the reference, which uses bin centers (from -0.49 to 1.49).

ii.
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
# Result: [-0.48, -0.46, ..., 1.48, 1.50]
```

iii. The agent chose bin right edges rather than bin centers. The reference uses `EDGES[:-1] + BIN/2` for bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time array has the same 100 elements as the neural bins, so each element corresponds one-to-one with a neural time bin. However, the time values represent right edges while the neural bins span from left to right edge, so the alignment point within each bin differs from the reference (right edge vs center).

ii.
```python
T = s['neural'][j].shape[1]
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. Same number of bins ensures 1:1 correspondence between time input and neural data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A block boundary is detected when `probabilityLeft` changes value.

ii.
```python
def trial_number_in_block(prob):
    out = np.empty(len(prob), dtype=np.float32)
    k = 0
    previous = None
    for i, value in enumerate(prob):
        if i == 0 or value != previous:
            k = 1
        else:
            k += 1
        out[i] = k
        previous = value
    return out
```

iii. The trials table has no explicit block identifier, so blocks are recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The counter starts at 1 for the first trial in each block and increments by 1 for each subsequent trial with the same `probabilityLeft`. This differs from the reference which uses `cumcount()` starting from 0. The count is computed over ALL trials before filtering, so dropped trials still advance the counter.

ii.
```python
def trial_number_in_block(prob):
    ...
    if i == 0 or value != previous:
        k = 1
    else:
        k += 1
    ...
block_trial = trial_number_in_block(trials.probabilityLeft.to_numpy())[idx]
```

iii. The agent chose to start counting from 1 rather than 0.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which uses +1 and -1 (and 0 for no response).

ii.
```python
choice_raw = trials.choice.to_numpy()[idx]
```

iii. The choice column directly encodes the animal's decision.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice using `(choice_raw == 1).astype(np.int8)`, which gives: choice=+1 maps to 1 and choice=-1 maps to 0. In IBL convention, +1 is a leftward choice and -1 is rightward. This means the AI produces left=1, right=0. However, the instructions specify "left = 0, right = 1", and the `output_values` list is `['left', 'right']` (implying index 0='left', index 1='right'). The AI's mapping is **inverted** relative to both the instructions and the reference solution.

ii.
```python
choice = (choice_raw == 1).astype(np.int8)
# +1 (left in IBL) -> 1, -1 (right in IBL) -> 0
# But instructions say: left = 0, right = 1
```

iii. The agent stated "left (-1) = 0, right (+1) = 1", suggesting it believed -1 is left and +1 is right in the IBL convention. The reference code states "+1 is a leftward choice, -1 rightward" and maps `{1.0: 0, -1.0: 1}`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pleft_raw = trials.probabilityLeft.to_numpy()[idx]
pmap = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([pmap[round(float(x), 1)] for x in pleft_raw], np.int8)
```

iii. The three values correspond to the block prior probabilities.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Values are rounded to one decimal place before lookup to handle floating-point imprecision.

ii.
```python
pmap = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([pmap[round(float(x), 1)] for x in pleft_raw], np.int8)
```

iii. The mapping matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel position and timestamps loaded via `SessionLoader.load_wheel()`. The velocity is computed internally by SessionLoader (interpolation to 1000 Hz, Butterworth low-pass filter, differentiation). The speed is the absolute value of velocity.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
wheel_t = sl.wheel.times.to_numpy()
wheel_v = np.abs(sl.wheel.velocity.to_numpy())
```

iii. Same approach as the reference - using SessionLoader's processed velocity and taking absolute value.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. SessionLoader internally interpolates the wheel position onto a 1000 Hz grid and applies a Butterworth low-pass filter to compute velocity. The absolute value is taken for speed. The speed is then linearly interpolated onto bin right edges for each trial. Finally, it is discretized into 3 categories using **global pooled tertiles** (quantiles at 1/3 and 2/3 computed across ALL sessions), not per-session percentiles.

ii.
```python
wheel_v = np.abs(sl.wheel.velocity.to_numpy())
# In interpolate():
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
# Global edges:
wheel_edges = category_edges(sessions, 'wheel')
wb = np.digitize(s['wheel'][j], wheel_edges).astype(np.int8)
```

iii. The agent chose global pooled tertiles for discretization rather than per-session percentiles (as the reference uses).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global pooled tertiles: all wheel speed values from all sessions are concatenated, quantiles at 1/3 and 2/3 are computed, and `np.digitize` assigns each value to one of three bins (0=low, 1=medium, 2=high).

ii.
```python
def category_edges(sessions, key):
    values = np.concatenate([np.concatenate(s[key]) for s in sessions])
    edges = np.quantile(values, [1/3, 2/3])
    ...
    return edges.astype(float)

wb = np.digitize(s['wheel'][j], wheel_edges).astype(np.int8)
```

iii. The agent chose global tertiles to ensure consistent bin boundaries across sessions. The reference uses per-session percentiles at 33.33 and 66.67.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is linearly interpolated at bin right edges (same grid as the time input), which are the 100 time points matching the neural data bins.

ii.
```python
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. The interpolation grid has 100 points matching the 100 neural bins, ensuring 1:1 temporal correspondence.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The ROI motion energy from the side camera (`leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy`) with matching camera timestamps. Left camera is preferred, with right camera as fallback.

ii.
```python
for view in ('left', 'right'):
    tf = sorted(alf.glob(f'#*#/_ibl_{view}Camera.times.npy'))
    ...
    mf = sorted(alf.glob(f'#*#/{view}Camera.ROIMotionEnergy.npy'))
    ...
```

iii. The agent follows the same left-preferred, right-fallback logic as the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is with no additional filtering. It is linearly interpolated onto bin right edges for each trial, then discretized into 3 categories using **global pooled tertiles** (same approach as wheel speed).

ii.
```python
# In interpolate():
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
# Global edges:
whisk_edges = category_edges(sessions, 'whisk')
mb = np.digitize(s['whisk'][j], whisk_edges).astype(np.int8)
```

iii. Same discretization approach as wheel speed - global tertiles rather than per-session.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical to wheel speed: global pooled tertiles across all sessions. All whisker motion energy values are pooled, quantiles at 1/3 and 2/3 computed, then `np.digitize` assigns categories.

ii.
```python
whisk_edges = category_edges(sessions, 'whisk')
mb = np.digitize(s['whisk'][j], whisk_edges).astype(np.int8)
```

iii. The agent consistently uses global tertiles for all continuous-to-categorical conversions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: linearly interpolated at bin right edges, giving 100 values matching the neural time bins.

ii.
```python
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials with NaN in required fields are excluded by the trial mask. (2) Trials without valid wheel or whisker coverage are excluded. (3) Non-finite behavior values cause trial exclusion. (4) Sessions with fewer than 2 valid trials are skipped. (5) Sessions where whisker motion energy files are missing or mismatched are skipped entirely. (6) Failed sessions are logged and recorded in metadata. (7) Read-only memory-mapped arrays are copied before in-place modification.

ii.
```python
valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
...
if len(vx) == 0 or not np.all(np.isfinite(vx)):
    out.append(None); continue
...
if len(idx) < 2:
    raise RuntimeError('fewer than two valid trials')
...
spikes['clusters'] = np.array(spikes['clusters'], dtype=np.int32, copy=True)
```

iii. The agent encountered multiple data issues during processing and added handlers progressively. Failed sessions are cached to avoid re-processing.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and binning spike sorting data is the most expensive step, as each probe's spike arrays are hundreds of megabytes. The agent processed 459 sessions using a parallel sharding system (up to 16 shards) with per-session intermediate caching to manage the workload.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. The cost is dominated by file I/O for spike data and the binning computation.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_number_in_block` function iterates through every trial with a Python loop, which could be vectorized using pandas `groupby().cumcount()` (as the reference does). The behavior interpolation loop iterates per-trial, which could potentially be vectorized.

ii.
```python
def trial_number_in_block(prob):
    out = np.empty(len(prob), dtype=np.float32)
    k = 0
    previous = None
    for i, value in enumerate(prob):
        ...
```

```python
for i, onset in enumerate(trials.stimOn_times.to_numpy()):
    ...
    out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. Both loops operate over trials which is typically a few hundred iterations, so the performance impact is minor.

## 10-c. What processing does the code repeat multiple times?

i. The code loads and processes each session twice when intermediate caching is not available - once for the initial processing pass and again when assembling the final output (loading from intermediate pickle). Additionally, `BrainRegions()` is instantiated once per session in `process_session` (line 184) rather than reusing a shared instance.

ii.
```python
beryl = BrainRegions().acronym2acronym(clusters.acronym.to_numpy(), mapping='Beryl')
```

iii. The intermediate caching design means each session is processed once and stored, then loaded again for assembly - but this is intentional for parallelization.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores raw spike counts as int16 rather than converting to firing rates. The downstream decoder may expect firing rates (as the reference provides). Additionally, the code computes and stores `source_trial_idx` in metadata, which is not required by the target format. The `compute_metrics=False` is passed but `merge_clusters` still performs some unnecessary metadata computation.

ii.
```python
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
# Reference: rate = spike_counts(...) / BIN  # converts to Hz
```

iii. Storing counts vs rates is a difference in representation; the decoder could handle either, but it doesn't match the reference format.
