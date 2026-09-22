# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the paper release CSV to define candidate probe/session rows, optionally restricts it with `DATALIMIT_SUBSET.csv`, constructs a local ONE client, groups rows by session EID, and uses `SessionLoader` and `SpikeSortingLoader` for the actual neuroscience arrays. It attempts every grouped release EID serially and skips sessions raising exceptions.

ii.
```python
rows = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
return ONE(cache_dir=CACHE, tables_dir=CACHE / 'Brainwidemap', mode='local')
grouped = list(rows.groupby('eid', sort=False))
```

iii. The notes say the exact 459-session/699-probe release CSV matches the paper, while ONE/brainbox should perform all actual loading. The AI expected incomplete sessions to reduce this toward the 433 sessions in the methods paper.

## 1-b. How are the data split into subjects?

i. Subject labels come from the release CSV. After successful sessions are collected, unique subject strings are sorted and each session receives an index into that vocabulary.

ii.
```python
info = dict(eid=str(eid), subject=str(rows.subject.iloc[0]), ...)
subjects = sorted({x['subject'] for x in infos})
subject_idx = np.asarray([subject_lookup[x['subject']] for x in infos], dtype=np.int64)
```

iii. The notes treat the release CSV as authoritative because it reproduces the paper's 139 mice and probe membership.

## 1-c. How are the data split into sessions?

i. Rows are grouped by EID; each EID is one session and all probe rows in the group are merged into that session.

ii.
```python
grouped = list(rows.groupby('eid', sort=False))
for k, (eid, erows) in enumerate(grouped, 1):
    vals = process_session(one, str(eid), erows, ...)
```

iii. The AI identified EID as the session identifier and simultaneous probes as one behavioral recording, following the supplied caching pipeline.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` returns one row per trial. Retained row indices select stimulus onsets, and each onset defines one neural/input/output trial window.

ii.
```python
sl.load_trials()
idx = np.flatnonzero(valid)
neural = bin_spikes(st, sc, onsets[idx], len(regions))
for j, raw_i in enumerate(idx):
```

iii. The AI regarded the trials table as the natural trial partition and used onset-indexed windows to keep streams synchronized.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have finite stimulus, choice, prior, first movement, feedback time/type; reaction time must be 0.08–2 s; choice must be nonzero; go-cue-to-feedback duration must be at most 10 s; and wheel and whisker interpolation must be finite over the full window. Sessions with fewer than two jointly valid trials are skipped.

ii.
```python
mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
valid = mask & wheel_good & whisk_good
if idx.size < 2:
    raise RuntimeError(f'only {idx.size} jointly valid trials')
```

iii. The notes attribute missing-event, no-choice, RT, and 10-s duration cuts to the supplied methods code, and add joint behavioral coverage because both dynamic decoder outputs are required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from every release probe's `spikes.times` and `spikes.clusters`; cluster/channel tables determine the number and region label of units.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
all_times.append(np.asarray(spikes['times'], dtype=float))
all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
```

iii. The notes identify these as the reference electrophysiology variables and explicitly state that calcium dF/F is inapplicable.

## 2-b. How is the `neural` data processed?

i. Probe spike trains are concatenated, cluster IDs offset, sorted by time, and counted per cluster in 100 half-open 20-ms bins for each retained trial. Counts are stored as float32; they are not divided by bin width or smoothed.

ii.
```python
flat = clusters[lo:hi][ok] * N_TIME + relbin[ok]
counts = np.bincount(flat, minlength=n_neurons * N_TIME).reshape(n_neurons, N_TIME)
result.append(counts.astype(np.float32))
```

iii. The AI says this matches the methods-paper cache, which bins spikes without smoothing, and that float32 controls memory. It explicitly labels the result “spike counts per 20-ms bin.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not quality-filtered: every loaded spike-sorted cluster is retained, including `root`/`void`; regions are only mapped to Beryl afterward.

ii.
```python
clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
n = len(clusters['channels'])
all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
```

iii. The notes reason that the methods-paper caching call used `qc=None` and “all neurons,” despite the data paper's 75,708 stringent-QC neurons. They therefore chose the 621,733-scale all-cluster population.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial spans stimulus onset minus 0.5 s through onset plus 1.5 s. Absolute spike times are sliced by those bounds and assigned bins relative to the window start.

ii.
```python
beg, end = onset + OFF_START, onset + OFF_END
lo, hi = np.searchsorted(times, [beg, end])
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. The AI chose the common stimulus-onset window because the downstream task mandates stimulus alignment and the supplied cache uses the same two-second window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 100 bins over two seconds. Raw spikes are binned once; no further temporal rebinning or smoothing is applied.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_TIME = 100
```

iii. The notes cite the method paper and supplied caching implementation for 20-ms bins and 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a constructed coordinate based on the requested offsets and bin size, used relative to each trial's raw `stimOn_times`; it is not read as a separate raw variable.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
onsets = tr.stimOn_times.to_numpy(dtype=float)
```

iii. The AI says the reference behavior code samples at bin right edges, so it selected `[-0.48, ..., 1.50]` seconds.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It generates 100 right-edge offsets at 20-ms increments and copies the same vector into every trial input.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```

iii. The notes justify right edges as matching the supplied behavior interpolation coordinates.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each value is the right edge of the corresponding preceding neural count bin; behavior is queried at those same right-edge times.

ii.
```python
query = onsets[:, None] + TIME[None, :]
relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. The AI explicitly says neural counts cover preceding half-open bins while the coordinate and behavioral values use their right edges.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive values of `trials.probabilityLeft`.

ii.
```python
block_number = trial_number_in_block(tr.probabilityLeft.to_numpy())
```

iii. The notes explain that probability-left is constant within blocks, so a change identifies a new block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter increments while consecutive prior values match and resets to zero when they differ. It is computed before filtering, then repeated across all 100 time points.

ii.
```python
for i in range(1, len(prob)):
    out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
np.full(N_TIME, block_no[raw_i], dtype=np.float32)
```

iii. Computing before filtering preserves the animal's actual within-block position rather than collapsing excluded trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`.

ii.
```python
choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
```

iii. The notes assert that IBL −1 means left and +1 means right, and no-choice zero trials are excluded.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps −1 to category 0 (“left”) and every retained alternative (+1) to category 1 (“right”), then repeats it over time.

ii.
```python
choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
np.full(N_TIME, choice)
```

iii. It says this recoding implements the requested left=0/right=1 categorical output.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

iii. The AI identifies this raw field as the experimental block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2 and repeated over time.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
np.full(N_TIME, prior)
```

iii. This is the exact mapping required by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses the absolute value of `SessionLoader`'s wheel velocity, itself derived from wheel timestamps and position.

ii.
```python
sl.load_wheel()
wt = sl.wheel.times.to_numpy(dtype=float)
ws = np.abs(sl.wheel.velocity.to_numpy(dtype=float))
```

iii. The notes say the reference behavior loader defines wheel speed as absolute interpolated wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` supplies velocity; its absolute value is linearly interpolated at every trial's 100 right-edge coordinates, invalid full-window trials are removed, and retained values are discretized by session-wide tertiles.

ii.
```python
query = onsets[:, None] + TIME[None, :]
out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
wheel_cat, wheel_edges = tertiles(wheel[idx])
```

iii. The AI cites the reference interpolation and argues session tertiles accommodate rig scaling while producing three learnable categories.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles over all retained trial-time samples in that session. `digitize` yields low/medium/high; tied thresholds trigger a deterministic rank-based fallback.

ii.
```python
edges = np.quantile(x, [1 / 3, 2 / 3])
return np.digitize(x, edges, right=False).astype(np.int64), edges
```

iii. The notes say within-session tertiles are robust to session/camera scale differences and the fallback ensures non-degenerate class representation.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel samples are interpolated at stimulus onset plus `TIME`, i.e. the right edges of the corresponding neural bins.

ii.
```python
query = onsets[:, None] + TIME[None, :]
out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
```

iii. The AI chose right-edge sampling to follow the supplied behavioral cache.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and camera frame times from the left camera when loadable, otherwise the right camera.

ii.
```python
for view in ('left', 'right'):
    sl.load_motion_energy(views=[view])
    df = sl.motion_energy[f'{view}Camera']
    return ..., df.times.to_numpy(...), df.whiskerMotionEnergy.to_numpy(...), view
```

iii. The notes state this left-preferred fallback matches the supplied reference logic and that views should not be averaged.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is linearly interpolated at trial right-edge coordinates, trials without complete finite coverage are removed, and retained values are discretized into session-wide tertiles. No extra filtering or normalization is applied.

ii.
```python
whisk, whisk_good = interpolate_trials(mt, me, onsets)
valid = mask & wheel_good & whisk_good
whisk_cat, whisk_edges = tertiles(whisk[idx])
```

iii. The AI says the released trace should be used directly and processed like wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the 1/3 and 2/3 session quantiles over retained trial-time samples, with the same rank fallback for equal edges.

ii.
```python
edges = np.quantile(x, [1 / 3, 2 / 3])
labels[order] = np.minimum(2, np.arange(order.size) * 3 // order.size)
```

iii. The notes use the same scale-robust, balanced-class rationale as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at stimulus onset plus each right-edge time coordinate.

ii.
```python
query = onsets[:, None] + TIME[None, :]
out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
```

iii. The AI says these are the reference behavior coordinates corresponding to neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trial columns raise an error for that session; nonfinite required events fail the trial mask; incomplete behavior windows are excluded; either camera is tried; sessions with fewer than two trials or any processing exception are logged and skipped. A tied-tertile fallback handles degenerate continuous signals.

ii.
```python
if missing:
    raise RuntimeError(f'trials object missing columns {sorted(missing)}')
except Exception as exc:
    failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
```

iii. The notes characterize joint stream coverage as necessary for valid outputs. In practice, the staged cache mismatch caused all 459 sessions to be skipped, and the AI documented that Steps 7–13 could not be completed.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant operations are loading/merging spike sorting for every probe and then per-trial spike binning; the code records total elapsed time per session but does not separately profile stages.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
neural = bin_spikes(st, sc, onsets[idx], len(regions))
```

iii. The notes focus their speedups on avoiding redundant loads and using sorted spikes/flattened bincount, implying I/O and spike processing dominate.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over trials in `bin_spikes` could be vectorized by assigning spikes to trial/bin indices in bulk. The trial assembly loop could also be replaced by batched array construction. The AI did vectorize behavior interpolation across trials, but session and probe loading remain necessarily iterative.

ii.
```python
for onset in onsets:
    ...
    counts = np.bincount(...)
for j, raw_i in enumerate(idx):
    inputs.append(inp); outputs.append(out)
```

iii. The notes claim “vectorized interpolation,” `searchsorted` spike windows, and flattened `bincount` as speedups, without claiming the remaining per-trial spike loop is fully vectorized.

## 10-c. What processing does the code repeat multiple times?

i. It repeats spike slicing/binning and input/output object construction for every trial, and creates/loads a spike loader for each probe. Behavioral loading is consolidated into one `SessionLoader` per session rather than repeated per target.

ii.
```python
for row in probe_rows.itertuples():
    ssl = SpikeSortingLoader(...)
for onset in onsets:
    lo, hi = np.searchsorted(times, [beg, end])
```

iii. The notes explicitly say the reference performed redundant behavior loads and the AI avoided them with one session loader.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and retains diagnostic quantile edges, camera view, raw-trial counts, elapsed seconds, failure details, and optional plots; these are not decoder features. `merge_clusters` also loads cluster metadata primarily to obtain count/regions, while no QC metrics are used.

ii.
```python
info = dict(..., wheel_edges=wheel_edges.tolist(), whisker_edges=whisk_edges.tolist(), seconds=elapsed)
if show:
    fig, ax = plt.subplots(...)
```

iii. These are diagnostics and provenance. The notes planned plots and statistics as sanity checks, so the extra work was intentional rather than part of downstream decoding.
