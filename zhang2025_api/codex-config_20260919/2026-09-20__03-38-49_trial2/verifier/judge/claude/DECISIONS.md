# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API in `local` mode to interface with the IBL ONE cache. It enumerates sessions from the `bwm_release.csv` file shipped with the reference code (Zhang et al.), optionally restricting to a `DATALIMIT_SUBSET.csv` file if present. For each session, it uses `SessionLoader` to load trials, wheel, and motion energy data, and `SpikeSortingLoader` to load spike sorting data. The AI processes sessions sequentially in a loop.

ii.
```python
def get_one():
    return ONE(cache_dir=CACHE, tables_dir=CACHE / 'Brainwidemap', mode='local')

def release_rows():
    rows = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
    if SUBSET.exists():
        subset = pd.read_csv(SUBSET)
        key = next((x for x in ('eid', 'session', 'session_id') if x in subset), None)
        if key:
            rows = rows[rows.eid.astype(str).isin(subset[key].astype(str))]
    return rows
```

iii. The AI chose to use `bwm_release.csv` from the reference code to enumerate sessions, and `mode='local'` for the ONE client to avoid network access. However, this approach failed entirely: all 459 sessions were skipped because `SessionLoader` could not find the required trials columns (`choice`, `probabilityLeft`, etc.) when using `local` mode, as `local` mode cannot resolve revisioned datasets. The AI documented this as a "blocking source-cache defect" in CONVERSION_NOTES.md.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject name from the release CSV rows (`rows.subject`) for each session. Subject identifiers are collected into a sorted unique list and each session is assigned an index into that list.

ii.
```python
subjects = sorted({x['subject'] for x in infos})
subject_lookup = {s:i for i,s in enumerate(subjects)}
```

iii. The subject information comes from the `bwm_release.csv` which contains subject names per session. This is a valid approach as the CSV is authoritative for the release.

## 1-c. How are the data split into sessions?

i. Sessions are the unit of the release CSV, with one or more rows per `eid` (one row per probe insertion). The AI groups the CSV by `eid` and processes each group as a single session.

ii.
```python
grouped = list(rows.groupby('eid', sort=False))
for k, (eid, erows) in enumerate(grouped, 1):
    vals = process_session(one, str(eid), erows, ...)
```

iii. The grouping by `eid` is correct since each `eid` represents one session, and multiple rows per `eid` represent multiple probes within that session.

## 1-d. How are the data split into trials?

i. The trials table from `SessionLoader.load_trials()` has one row per trial, so the split is inherent in the data. Each valid trial becomes one entry in the neural, input, and output lists.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
tr = sl.trials.copy()
```

iii. Standard approach using the IBL trials table structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple filters: (1) finiteness check on `stimOn_times`, `choice`, `probabilityLeft`, `firstMovement_times`, `feedback_times`, `feedbackType`; (2) reaction time between 0.08 and 2.0 seconds; (3) trial duration (`feedback_times - goCue_times`) <= 10 seconds; (4) choice != 0 (excluding no-go trials); (5) wheel and whisker motion energy coverage across the trial window.

ii.
```python
finite = np.ones(len(tr), dtype=bool)
for col in ['stimOn_times', 'choice', 'probabilityLeft', 'firstMovement_times',
            'feedback_times', 'feedbackType']:
    finite &= np.isfinite(tr[col].to_numpy(dtype=float))
rt = tr.firstMovement_times.to_numpy() - tr.stimOn_times.to_numpy()
duration = tr.feedback_times.to_numpy() - tr.goCue_times.to_numpy()
mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
```

iii. The AI found the duration <= 10s filter in the reference code's `prepare_data` function and included it. The finiteness checks go beyond the reference solution, which relies on NaN-comparison behavior and specific value checks. The AI does not explicitly filter on valid `probabilityLeft` values ({0.2, 0.5, 0.8}) but relies on the later mapping to fail for unexpected values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader`, plus `clusters.acronym` for brain region labels and `clusters.channels` for merging metadata.

ii.
```python
ssl = SpikeSortingLoader(pid=str(row.pid), eid=str(row.eid), pname=row.probe_name, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
all_times.append(np.asarray(spikes['times'], dtype=float))
all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
```

iii. Standard approach using IBL spike sorting loaders, same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the 2-second trial window (-0.5 to 1.5 s around stimulus onset). Multiple probes are merged by offsetting cluster indices. The counts are stored as float32 but are NOT converted to firing rates (not divided by the bin width).

ii.
```python
def bin_spikes(times, clusters, onsets, n_neurons):
    result = []
    for onset in onsets:
        beg, end = onset + OFF_START, onset + OFF_END
        lo, hi = np.searchsorted(times, [beg, end])
        relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
        ok = (relbin >= 0) & (relbin < N_TIME)
        flat = clusters[lo:hi][ok] * N_TIME + relbin[ok]
        counts = np.bincount(flat, minlength=n_neurons * N_TIME).reshape(n_neurons, N_TIME)
        result.append(counts.astype(np.float32))
    return result
```

iii. The AI's metadata explicitly states `'neural_measure': 'spike counts per 20-ms bin'`. The reference solution divides by BIN to produce firing rates in Hz. This is a meaningful difference in scale, though it should not affect decoder performance since the decoder can learn arbitrary linear scaling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT apply any cluster quality filtering. All spike-sorted clusters are retained, regardless of their quality label. The AI explicitly decided this based on the observation that the reference Zhang et al. caching code calls `load_spiking_data(qc=None)`, which does not filter by quality.

ii.
```python
n = len(clusters['channels'])
all_times.append(np.asarray(spikes['times'], dtype=float))
all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
```

No QC filtering is applied. The AI also does not filter out `void` brain regions during spike loading. Beryl mapping is applied later during assembly.

iii. The AI justifies this in CONVERSION_NOTES.md: "the published caching call does not filter to good clusters (qc=None). Therefore all loaded sorted clusters are the reference choice." The AI's metadata confirms: `'cluster_filter': 'all spike-sorted clusters, matching Zhang et al. caching code'`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike times are aligned to stimulus onset (`stimOn_times`) by defining the window as `[onset + OFF_START, onset + OFF_END]` and computing bin indices relative to the window start.

ii.
```python
beg, end = onset + OFF_START, onset + OFF_END
lo, hi = np.searchsorted(times, [beg, end])
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. Standard temporal alignment to stimulus onset, consistent with the reference approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, producing 100 time bins per trial over the 2-second window. No rebinning is applied.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_TIME = 100
```

iii. Matches the reference code configuration exactly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From the bin time grid itself, not directly from any raw data variable. The time values are the bin RIGHT EDGES relative to stimulus onset.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```

This produces values: [-0.48, -0.46, ..., 1.48, 1.50].

iii. The AI notes in its metadata: `'bin_coordinate': 'right edge'`. It justified this as matching the reference code's behavior interpolation coordinates.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as right edges of the 100 bins spanning -0.5 to 1.5 seconds. No further processing.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```

iii. The AI chose right edges to match the reference code's behavior interpolation points.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The TIME array is used both as the time input and as the interpolation coordinates for behavioral signals (wheel, whisker). Neural spike counts are binned into the same 100 time bins. The alignment is by construction.

ii.
```python
inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```

iii. Alignment is ensured by using the same time grid throughout.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From the `probabilityLeft` column of the trials table, which is constant within a block.

ii.
```python
def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out
```

iii. Same source variable as the reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter that increments for each consecutive trial with the same `probabilityLeft` value and resets to 0 when the value changes. Computed before trial filtering so excluded trials still advance the counter.

ii.
```python
def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out

# Called before filtering:
block_no = trial_number_in_block(tr.probabilityLeft.to_numpy())
```

iii. Equivalent to the reference's `cumcount()` approach. The AI correctly computes this before filtering, matching the reference behavior.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values -1, 0, or 1.

ii.
```python
choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice=-1 to 0 and choice=+1 to 1 (choice=0 is excluded by filtering). The AI interprets the IBL convention as -1=left, +1=right, so this produces left=0, right=1.

ii.
```python
choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
```

iii. The reference solution uses the opposite mapping: `{1.0: 0, -1.0: 1}`, interpreting +1=left, -1=right. The IBL standard documentation defines choice as -1=left, 0=no-go, 1=right, which would make the AI's interpretation correct. However, the IBL community has used conflicting conventions, and the reference code's comment explicitly states "+1 is a leftward choice, -1 rightward."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

iii. Same source variable and mapping as the reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The AI adds `round(float(...), 1)` for numerical robustness.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

iii. Matches the reference exactly. The rounding is a defensive measure against floating point precision issues.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `SessionLoader.wheel.velocity` (which internally derives from `_ibl_wheel.position` and `_ibl_wheel.timestamps`). Speed is the absolute value of velocity.

ii.
```python
sl.load_wheel()
wt = sl.wheel.times.to_numpy(dtype=float)
ws = np.abs(sl.wheel.velocity.to_numpy(dtype=float))
```

iii. Same source and processing as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The absolute wheel velocity is linearly interpolated onto the TIME grid (bin right edges) for each trial. SessionLoader internally applies interpolation and a Butterworth low-pass filter to the raw wheel position before computing velocity.

ii.
```python
def interpolate_trials(times, values, onsets):
    query = onsets[:, None] + TIME[None, :]
    good = ((query[:, 0] >= times[0]) & (query[:, -1] <= times[-1]))
    out = np.full(query.shape, np.nan, dtype=np.float32)
    if np.any(good):
        out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
    return out, good & np.isfinite(out).all(axis=1)
```

iii. The AI uses a vectorized interpolation approach across all trials at once, which is more efficient than the reference's per-trial loop. The interpolation target is bin right edges rather than the reference's bin centres.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 categories using within-session tertiles (1/3 and 2/3 quantiles). A rank-based fallback is used when quantile edges are degenerate (equal).

ii.
```python
def tertiles(x):
    edges = np.quantile(x, [1 / 3, 2 / 3])
    if edges[0] < edges[1]:
        return np.digitize(x, edges, right=False).astype(np.int64), edges
    order = np.argsort(x.ravel(), kind='stable'); labels = np.empty(order.size, dtype=np.int64)
    labels[order] = np.minimum(2, np.arange(order.size) * 3 // order.size)
    return labels.reshape(x.shape), edges
```

iii. The tertile split is equivalent to the reference's percentile approach (`np.percentile(trace, [100/3, 200/3])` is the same as `np.quantile(x, [1/3, 2/3])`). The degenerate-case fallback is extra robustness the reference does not have.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated at the same TIME grid (bin right edges) as the neural bins, aligned to stimulus onset.

ii.
```python
query = onsets[:, None] + TIME[None, :]
out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
```

iii. Alignment by construction using the shared time grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` (preferred) or `rightCamera.ROIMotionEnergy` (fallback), with corresponding camera timestamps.

ii.
```python
for view in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[f'{view}Camera']
        return wt, ws, df.times.to_numpy(dtype=float), df.whiskerMotionEnergy.to_numpy(dtype=float), view
    except Exception as exc:
        last_error = exc
```

iii. Same approach as reference: prefer left camera, fall back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is linearly interpolated onto the TIME grid (bin right edges) for each trial. No additional filtering or normalization.

ii.
```python
whisk, whisk_good = interpolate_trials(mt, me, onsets)
```

iii. Same `interpolate_trials` function used for wheel speed, applied to the motion energy trace.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same tertile discretization as wheel speed: within-session 1/3 and 2/3 quantiles, with rank-based fallback for degenerate cases.

ii.
```python
whisk_cat, whisk_edges = tertiles(whisk[idx])
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same alignment approach as wheel speed: interpolated at the shared TIME grid relative to stimulus onset.

ii.
```python
whisk, whisk_good = interpolate_trials(mt, me, onsets)
```

iii. Alignment by construction using the shared time grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) finiteness checks on required trial columns; (2) `interpolate_trials` returns a `good` mask marking trials where the behavioral trace spans the full window and produces finite values; (3) sessions with fewer than 2 jointly valid trials raise an exception and are skipped; (4) probes that fail to load raise an exception handled by the session-level try/except.

ii.
```python
good = ((query[:, 0] >= times[0]) & (query[:, -1] <= times[-1]))
...
return out, good & np.isfinite(out).all(axis=1)
```

```python
valid = mask & wheel_good & whisk_good
idx = np.flatnonzero(valid)
if idx.size < 2:
    raise RuntimeError(f'only {idx.size} jointly valid trials')
```

iii. The AI's approach is robust but potentially more aggressive in dropping sessions (any exception skips the entire session).

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data via `SpikeSortingLoader` is the most expensive step, involving reading large spike time and cluster arrays from disk.

ii.
```python
ssl = SpikeSortingLoader(pid=str(row.pid), eid=str(row.eid), pname=row.probe_name, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The AI identified this correctly in CONVERSION_NOTES.md and noted that vectorized processing was added to minimize compute time per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop (one iteration per trial onset) could be vectorized by offsetting all spike indices at once. The `trial_number_in_block` function uses a Python loop that could be replaced with a vectorized cumulative approach.

ii.
```python
# Per-trial spike binning loop
for onset in onsets:
    beg, end = onset + OFF_START, onset + OFF_END
    lo, hi = np.searchsorted(times, [beg, end])
    ...
```

```python
# Python loop for trial number
def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out
```

iii. The AI vectorized the behavioral interpolation but left the spike binning and trial numbering as loops.

## 10-c. What processing does the code repeat multiple times?

i. The AI calls `np.searchsorted` for trials both in coverage checking (via `interpolate_trials`) and in spike binning (via `bin_spikes`). Behavioral data interpolation is vectorized, so it does not repeat per trial.

ii.
```python
# In interpolate_trials:
query = onsets[:, None] + TIME[None, :]
# In bin_spikes:
lo, hi = np.searchsorted(times, [beg, end])
```

iii. The AI noted in CONVERSION_NOTES that it avoids redundant behavior loading by using a single SessionLoader.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and checks `feedbackType` and `goCue_times` for the trial duration filter (`feedback_times - goCue_times <= 10`), which the reference solution does not use. The AI also computes `wheel_edges` and `whisk_edges` that are stored in session info but not used by the decoder.

ii.
```python
duration = tr.feedback_times.to_numpy() - tr.goCue_times.to_numpy()
mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
```

```python
info = dict(..., wheel_edges=wheel_edges.tolist(), whisker_edges=whisk_edges.tolist(), ...)
```

iii. The duration filter comes from the reference Zhang et al. code's `prepare_data` function but is not present in the human reference solution.
