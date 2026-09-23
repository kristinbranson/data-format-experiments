# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API in local mode (`ONE(mode='local', cache_dir=CACHE)`) for session discovery and spike sorting, but reads the release CSV (`bwm_release.csv`) as the authoritative list of sessions and probe insertions. Trials are loaded by directly reading parquet files via a custom `newest()` function that handles revision directories, rather than using `SessionLoader`. Wheel data is loaded via `one.load_object(eid, 'wheel', collection='alf')`. Whisker motion energy is loaded by directly reading numpy files from disk. SpikeSortingLoader is used for spike data.

ii.
```python
one = ONE(mode='local', cache_dir=CACHE)
one.load_cache(CACHE / 'Brainwidemap')
release = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
...
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    spath = one.eid2path(eid)
    tr = load_trials(spath)
    ...
    ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
```

iii. The agent initially tried using ONE's `load_object` for trials but found it returned only legacy unrevised files. It switched to direct parquet reading with a `newest()` function that selects the most recent revision. The release CSV was used because it is the authoritative session list from the Zhang et al. caching script.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the `subject` column of `bwm_release.csv`. They are deduplicated in encounter order (order of appearance in the release CSV). The subject index maps each session to its position in this list.

ii.
```python
subjects = list(dict.fromkeys(s['subject'] for s in sessions))
subj_map = {x:i for i,x in enumerate(subjects)}
...
subject_idx=np.array([subj_map[s['subject']] for s in kept], np.int32)
```

iii. The subject identity comes directly from the release CSV's subject column, so no parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV defines one session. Multiple probe insertions for the same `eid` are grouped together via `release.groupby('eid', sort=False)`.

ii.
```python
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
```

iii. The release CSV already lists sessions by `eid`, so the groupby naturally separates sessions.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. Valid trial indices are collected in `ti = np.flatnonzero(valid)`.

ii.
```python
tr = load_trials(spath)
...
valid = np.ones(len(tr), bool)
...
ti = np.flatnonzero(valid)
```

iii. The trials table is already one row per trial; no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on: (1) all six required trial event fields (`choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`) must be finite; (2) reaction time between 0.08 and 2.0 seconds; (3) valid binary choice (`choice in [-1, 1]`); (4) valid prior probability (`probabilityLeft in [0.2, 0.5, 0.8]`); (5) complete behavioral bins (both wheel and whisker traces must have finite values in all 100 time bins after gap-filling). Sessions with fewer than 2 valid trials are skipped.

ii.
```python
valid = np.ones(len(tr), bool)
for c in needed:
    valid &= np.isfinite(tr[c].to_numpy(float))
rt = tr.firstMovement_times.to_numpy(float) - tr.stimOn_times.to_numpy(float)
valid &= (rt >= .08) & (rt <= 2.00)
valid &= np.isin(tr.choice.to_numpy(float), [-1, 1])
valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
...
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
```

iii. The agent followed the data paper's inclusion criteria: reaction time bounds of 0.08-2.0 s, exclusion of no-go trials, and requirement for finite event times. The additional requirement that `feedbackType` and `feedback_times` be finite goes beyond the reference solution but is a reasonable quality control.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster assignments). The cluster table provides quality labels (`label`) and atlas IDs (`atlas_id`) for filtering.

ii.
```python
ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
```

iii. Standard IBL spike sorting data loaded via SpikeSortingLoader.

## 2-b. How is the `neural` data processed?

i. Spikes from good clusters are binned into 20 ms bins over the [-0.5, 1.5) s trial window (100 bins), stored as spike counts (uint16). For each trial, spike times are sliced via `searchsorted`, mapped to neuron indices via a cluster-to-row mapping, and accumulated using `np.add.at`. Multiple probes in a session are concatenated (neurons pooled). The neural data is stored as raw spike counts, NOT converted to firing rates.

ii.
```python
neural = np.zeros((len(ti), n_neurons, len(CENTERS)), dtype=np.uint16)
...
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
    cc = cmap[sc[lo:hi]]
    ok = cc >= 0
    if np.any(ok):
        bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
        inside = (bb >= 0) & (bb < len(CENTERS))
        np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. The agent implemented vectorized trial-wise binning for efficiency, avoiding per-neuron scanning of the full spike vector.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters are applied: (1) Only clusters with `label == 1` (RIGOR well-isolated units) are kept. (2) Clusters with non-finite or non-positive `atlas_id` are excluded. (3) Beryl-mapped regions `root`, `void`, and `fiber tracts` are excluded as non-grey-matter. Additionally, a per-session region threshold of >=5 good neurons is applied, and a cross-session prevalence filter requires each region to appear in >=2 sessions.

ii.
```python
good = (labels == 1) & np.isfinite(atlas_ids) & (atlas_ids > 0)
...
mapped = atlas.remap(atlas_ids[good].astype(int), source_map='Allen', target_map='Beryl')
acr = np.asarray(atlas.get(mapped)['acronym']).astype(str)
keep = ~np.isin(acr, ['root','void','fiber tracts'])
...
# Per-session: >=5 neurons per region
vals, cnt = np.unique(all_regions, return_counts=True)
allowed = set(vals[cnt >= 5])
...
# Cross-session: >=2 sessions per region
allowed_global = {r for r,n in prevalence.items() if n >= 2}
```

iii. The agent followed the data paper's quality criteria: well-isolated units (label==1), grey-matter regions only, >=5 neurons per region per session, and >=2 sessions per region globally.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to stimulus onset by subtracting each trial's `stimOn_times` from the spike timestamps. The window is [-0.5, 1.5) s relative to stimulus onset.

ii.
```python
onsets = tr.stimOn_times.to_numpy(float)[ti]
...
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
```

iii. All data streams are on the same session clock (IBL synchronization), so alignment is a simple subtraction of the onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, producing 100 time bins over the 2 s window. No rebinning or interpolation is applied to neural data.

ii.
```python
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The 20 ms bin size follows the reference code's `binsize: 0.02` and the method paper's description of "T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table. The input is the bin center times of the 100 bins spanning [-0.5, 1.5) s around stimulus onset.

ii.
```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
input_out.append([np.vstack((CENTERS.astype(np.float32), ...)) for j in ...])
```

iii. The bin centers define the time axis, identical for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing -- the variable is the bin center times, defined by the binning grid. The same CENTERS array is used for every trial.

ii.
```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. N/A

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centers are the midpoints of the same time bins used for neural data, so they are aligned by construction.

ii.
```python
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` between consecutive trials marks a new block boundary.

ii.
```python
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
```

iii. The trials table has no explicit block identifier, so blocks must be inferred from changes in the prior probability.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter that resets when `probabilityLeft` changes between consecutive trials (or when the previous value is non-finite). The count is computed over all trials before filtering, then indexed by valid trial indices.

ii.
```python
block_trial = np.zeros(len(tr), np.float32); k = 0
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
```

iii. The count is taken before trial filtering so the trial number reflects the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table. IBL convention: +1 = left, -1 = right. Recoded to 0 = left, 1 = right.

ii.
```python
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)
```

iii. The agent initially encoded choice incorrectly (choice==+1 as 1), then corrected it after noticing the IBL convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A simple boolean cast: `choice == -1` maps to 1 (right), everything else to 0 (left). No-response trials (choice==0) are already filtered out.

ii.
```python
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)
```

iii. N/A

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, 0.8, mapped to 0, 1, 2.

ii.
```python
pleft = np.round(tr.probabilityLeft.to_numpy(float)[ti], 1)
prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
```

iii. The three-value mapping follows the instructions directly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A direct dictionary lookup mapping 0.2->0, 0.5->1, 0.8->2 after rounding to 1 decimal place.

ii.
```python
prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
```

iii. N/A

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `wheel.timestamps` and `wheel.position` loaded via `one.load_object(eid, 'wheel', collection='alf')`.

ii.
```python
wheel = one.load_object(eid, 'wheel', collection='alf')
wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
ws, _ = velocity_filtered(wpos, 1000)
```

iii. Standard IBL wheel data.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) Position is interpolated onto a 1000 Hz grid. (2) Velocity is computed with a 20 Hz Butterworth low-pass filter via `velocity_filtered`. (3) Absolute value gives speed. The speed is then averaged into 20 ms bins per trial using `binned_mean()`, with `fill_short_gaps()` to interpolate occasional empty bins.

ii.
```python
wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
ws, _ = velocity_filtered(wpos, 1000)
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
```

iii. The agent stated this is the "standard IBL wheel processing" pipeline.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-wise tertiles: the 1/3 and 2/3 quantiles of all finite wheel speed values across all trials in the session define the bin boundaries. `np.digitize` maps values to bins 0, 1, 2.

ii.
```python
def tertiles(x):
    finite = x[np.isfinite(x)]
    q = np.quantile(finite, [1/3, 2/3])
    ...
    return np.digitize(x, q, right=False).astype(np.int8), q.tolist()
...
wheel_c, wheel_q = tertiles(wheel_b)
```

iii. Session-wise tertiles give approximately equal-frequency bins within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is averaged into the same 20 ms bins as neural data, aligned to the same stimulus onset, so they share the same time axis.

ii.
```python
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
```

iii. The wheel timestamps are on the same session clock as the neural data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy`, loaded directly from disk via numpy.

ii.
```python
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
tf = newest(alf, '_ibl_leftCamera.times.npy')
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
```

iii. The agent noted the left camera is "the high-frame-rate side view used for whisker-pad motion."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is with no filtering or normalization. It is averaged into 20 ms bins per trial using `binned_mean()`, with `fill_short_gaps()` for interpolation of empty bins. Then discretized into 3 session-wise tertile bins.

ii.
```python
n = min(len(me), len(mt))
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
...
whisk_c, whisk_q = tertiles(whisk_b)
```

iii. No additional processing beyond binning and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel: session-wise tertiles using 1/3 and 2/3 quantiles. `np.digitize` maps to bins 0, 1, 2.

ii.
```python
whisk_c, whisk_q = tertiles(whisk_b)
```

iii. Same approach as wheel speed for consistency.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker ME trace is averaged into the same 20 ms bins as neural data, aligned to the same stimulus onset.

ii.
```python
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
```

iii. Camera frame times are on the same session clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Trials missing any required event field (NaN) are excluded. (2) Sessions missing left camera motion energy are skipped entirely (22 sessions). (3) Sessions with no well-isolated grey-matter neurons are skipped. (4) Empty temporal bins in behavioral traces are interpolated via `fill_short_gaps()`. (5) Trials still containing NaN after gap-filling are excluded. (6) Sessions with <2 valid trials are skipped. (7) The right camera is NOT used as fallback (unlike the reference).

ii.
```python
def fill_short_gaps(x):
    x = np.asarray(x, np.float32)
    q = np.arange(x.shape[1])
    for row in x:
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            row[~ok] = np.interp(q[~ok], q[ok], row[ok])
    return x
...
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
```

iii. The agent took a conservative approach, dropping sessions/trials with insufficient data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (hundreds of megabytes per probe) and binning spikes into trials. The full conversion took approximately 60 minutes for 459 sessions.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The cost is mainly file I/O for the large spike arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop iterates over each trial's onset to slice and bin spikes. This could potentially be vectorized using offset indices across all trials. The `fill_short_gaps` function also loops per trial row.

ii.
```python
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
    ...
    np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. The agent did optimize from per-neuron scanning to per-trial slicing, but per-trial loops remain.

## 10-c. What processing does the code repeat multiple times?

i. The `newest()` function performs `rglob` searches for the same file patterns (trials table, camera files) for each session. The `np.round` of `probabilityLeft` is done twice (once for filtering, once for block trial computation).

ii.
```python
valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
...
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
```

iii. Minor redundancy, not significant for performance.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `wheel_q` and `whisk_q` (tertile thresholds) in session_info metadata, which are not used by the decoder. The code also requires `feedbackType` and `feedback_times` to be finite, which is stricter filtering than needed (these fields are not used in the output). The region prevalence filtering (>=5 neurons/session, >=2 sessions/region) removes neurons that the reference keeps, potentially discarding useful neural data.

ii.
```python
needed = ['choice','probabilityLeft','feedbackType','feedback_times',
          'stimOn_times','firstMovement_times']
...
session_info=[dict(eid=s['eid'],source_trial_indices=s['trials'].tolist(),
              wheel_tertiles=s['wheel_q'],whisker_tertiles=s['whisk_q']) for s in kept]
```

iii. The extra filtering criteria come from the data paper's analysis pipeline, which is more restrictive than what the decoder task requires.
