# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent initializes the staged Brainwidemap cache with `ONE`, repairs the trials-table default-revision flag in memory, and uses `SessionLoader` and `SpikeSortingLoader` for scientific arrays. It chooses sessions and probes from the methods-paper freeze CSV, optionally limiting sample mode to its first two EIDs. Full mode processes every EID in that freeze, in freeze order, with session-level parallel workers.

ii.
```python
one = ONE(silent=True)
one.load_cache(tag='Brainwidemap', clobber=True)
ds = one._cache['datasets']
mask = ds.rel_path.str.endswith('_ibl_trials.table.pqt')
ds.loc[mask, 'default_revision'] = True
```
```python
f = pd.read_csv(FREEZE).drop(columns=['Unnamed: 0'], errors='ignore')
eids = list(dict.fromkeys(f.eid.astype(str)))
```

iii. The notes say this keeps scientific loading exclusively within ONE/brainbox while using the freeze only for curated EID/PID/probe identifiers. The metadata repair addresses a staged-cache revision anomaly without directly reading scientific data files.

## 1-b. How are the data split into subjects?

i. Subject identity is read from ONE session metadata, with the freeze metadata used only as a fallback. At assembly, unique subject names are sorted and each accepted session receives an integer index into that list.

ii.
```python
details = one._cache['sessions'].loc[uuid.UUID(str(eid))]
subject = str(details.subject)
...
subjects = sorted({x['subject'] for x in infos})
subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int32)
```

iii. The agent states that ONE session metadata is the authoritative source and the freeze fields are identifiers/provenance only.

## 1-c. How are the data split into sessions?

i. Each unique EID is one session. Probe rows sharing an EID are grouped and processed together; successfully converted EIDs become entries in the session-level lists.

ii.
```python
eids = list(dict.fromkeys(f.eid.astype(str)))
...
process_session(one, freeze[freeze.eid.astype(str) == eid], eid)
```

iii. The freeze order is preserved because the agent describes it as the paper/reference order, and probes are merged within rather than treated as independent sessions.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies a table with one row per native trial. Retained raw row indices are used to build one neural, input, and output array per trial.

ii.
```python
sl.load_trials()
tr = sl.trials
...
idx = np.flatnonzero(valid)
for j, raw_i in enumerate(idx):
    inputs.append(inp); outputs.append(out); neural.append(neural_all[j].copy())
```

iii. The notes treat the trials table as the native trial partition and retain raw indices in metadata for provenance.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have finite choice, prior, feedback type/time, stimulus time, first-movement time, and interval bounds; binary choice and a supported prior; duration at most 10 s; first movement 0.08–2.00 s after stimulus; and fully finite interpolated wheel and whisker windows. Sessions with fewer than two retained trials are excluded.

ii.
```python
valid = np.all(np.isfinite(vals), axis=1)
valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
valid &= (tr.intervals_1.to_numpy() - tr.intervals_0.to_numpy() <= 10.0)
latency = tr.firstMovement_times.to_numpy() - stim
valid &= (latency >= 0.08) & (latency <= 2.0)
valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
```

iii. The agent says it combines the paper's movement/no-go criteria with the reference loader's finite-event and 10 s duration checks, and requires complete behavior because every requested output must be defined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices come from `spikes['times']` and `spikes['clusters']`. Merged cluster `cluster_id`, `label`, and `acronym` fields determine which units are retained and their regions.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
ids, labels, acr = cluster_fields(clusters)
pm = bin_probe(spikes['times'], spikes['clusters'], good_ids, stim)
```

iii. The notes identify spike times/assignments as the activity source and merged cluster metadata as the operational QC and anatomy source.

## 2-b. How is the `neural` data processed?

i. For each retained trial and qualified unit, spikes are histogrammed into 100 half-open 20 ms bins from -0.5 to +1.5 s around stimulus onset. Probe matrices are concatenated on the unit axis. Values are stored as `uint16` spike counts, with no smoothing, z-scoring, or conversion to firing rate.

ii.
```python
bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
flat = ui[keep][valid].astype(np.int64) * 100 + bi[valid]
cnt = np.bincount(flat, minlength=len(good_ids) * 100).reshape(len(good_ids), 100)
...
return np.concatenate(mats, axis=1), regions, probe_info, failures
```

iii. The agent justifies counts as matching the methods repository's cached representation and preserving count information; compact integer storage also reduces memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters must have merged `label >= 1` and a nonempty acronym other than `void`, `root`, `nan`, or `none`. A failed probe is logged and skipped; a session is retained if another probe yields qualified units, but excluded if none does.

ii.
```python
INVALID_REGIONS = {'', 'void', 'root', 'nan', 'none'}
region_ok = np.array([x.strip().lower() not in INVALID_REGIONS for x in acr])
keep = (labels >= 1) & region_ok
```

iii. The notes connect `label >= 1` to the paper's well-isolated-unit criterion. They call non-root anatomy a valid assignment and deliberately omit region-level minimum-neuron/cross-session filters because this output is session-level.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is aligned to `stimOn_times`; spike bins span `[stimulus onset - 0.5 s, stimulus onset + 1.5 s)`.

ii.
```python
lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
```

iii. The agent says stimulus alignment is explicitly required and matches the methods-code cache, even though another paper analysis used movement alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 20 ms, yielding 100 bins in a 2 s window. Raw spikes are binned once at that resolution; there is no later temporal rebinning.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
```

iii. The notes cite the supplied caching configuration and paper's 20 ms neural/behavior bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the fixed relative bin grid around each trial's `stimOn_times`, specifically the centers of the same bins used for neural activity.

ii.
```python
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. The agent describes it as the decoder-task/reference cache grid, common to all trials.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Adjacent -0.5-to-1.5 s edges are averaged to form centers `-0.49, -0.47, ..., 1.49` seconds; the vector is copied into every trial input.

ii.
```python
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
```

iii. No data-dependent transformation is used; the notes explicitly define the fixed centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Its entries are the centers of the exact neural bins, so column `t` in the input and neural matrices refers to the same 20 ms interval.

ii.
```python
bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
inp = np.vstack((CENTERS_REL, ...))
```

iii. The agent validates that neural, input, and output time dimensions are all 100 and uses the common grid for all streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from contiguous runs of `trials.probabilityLeft` in the unfiltered native trial sequence.

ii.
```python
prior = tr.probabilityLeft.to_numpy(float)
block_num = trial_number_in_block(prior)
```

iii. The notes say the trials table has no explicit block counter, while constant prior runs encode blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter increments when the current finite prior equals the immediately preceding prior and resets to zero otherwise. It is computed before filtering, then repeated across all 100 bins of each retained trial.

ii.
```python
for i in range(1, len(prior)):
    out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
```

iii. The agent explains that filtered trials should still advance the animal's real position in the experimental block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`; zero and non-finite choices are excluded. The agent interprets `-1` as left and maps it to 0, and interprets `+1` as right and maps it to 1.

ii.
```python
choice = tr.choice.to_numpy(float)
valid &= np.isin(choice, [-1, 1])
...
out[0] = 0 if choice[raw_i] == -1 else 1
```

iii. The notes assert that this is the required left=0/right=1 convention. That assertion rests on a reversed interpretation of IBL's native sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The retained native sign is recoded to 0/1 and repeated over the 100 time bins so all outputs share a `(4, 100)` array.

ii.
```python
out = np.empty((4, 100), dtype=np.uint8)
out[0] = 0 if choice[raw_i] == -1 else 1
```

iii. The agent says repeating per-trial values avoids ragged/mixed-dimensional output arrays.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from `trials.probabilityLeft`.

ii.
```python
prior = tr.probabilityLeft.to_numpy(float)
valid &= np.isin(prior, [0.2, 0.5, 0.8])
```

iii. These are the three task block priors specified by the instructions and source data.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are mapped `0.2 → 0`, `0.5 → 1`, `0.8 → 2` and repeated across 100 bins.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
out[1] = prior_map[float(prior[raw_i])]
```

iii. The mapping exactly follows the task; unsupported or missing values are rejected during QC.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from wheel timestamps and position loaded by `SessionLoader.load_wheel()`, using the loader's velocity and then taking its absolute value.

ii.
```python
sl.load_wheel()
w = sl.wheel
speed = interpolate_trials(w['times'].to_numpy(),
                           np.abs(w['velocity'].to_numpy()), stim)
```

iii. The notes say this follows the reference/brainbox wheel-velocity derivation and the requested quantity is speed rather than signed velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Brainbox derives filtered velocity from position; the absolute velocity is linearly interpolated at the 100 neural-bin centers. Non-finite/out-of-support trial windows are rejected, and retained samples are discretized by per-session tertiles.

ii.
```python
flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
...
speed_cls, speed_q = quantile_classes(speed, valid)
```

iii. The agent cites scale variation across sessions and balanced classes as reasons for session-level tertiles, and prohibits extrapolation outside stream support.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles of all finite wheel samples from retained trials in the session are thresholds for low/medium/high classes 0/1/2. A duplicated upper edge is nudged upward.

ii.
```python
q = np.quantile(vals, [1/3, 2/3]).astype(float)
if q[1] <= q[0]:
    q[1] = np.nextafter(q[0], np.inf)
cls = np.digitize(x, q, right=False).astype(np.uint8)
```

iii. The instructions specify three classes but not physical thresholds; the agent argues tertiles are reproducible and approximately balanced.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is evaluated at `stimOn_times + CENTERS_REL`, the centers of the neural bins.

ii.
```python
q = np.asarray(stim)[:, None] + centers_rel[None, :]
flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The common session clock and common bin centers provide bin-for-bin alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It comes from a side camera's `whiskerMotionEnergy` ROI trace and camera timestamps loaded through `SessionLoader.load_motion_energy`. Left is tried first, then right.

ii.
```python
for side in ('left', 'right'):
    sl.load_motion_energy(views=[side])
    me = sl.motion_energy[f'{side}Camera']
    vals = me['whiskerMotionEnergy'].to_numpy()
```

iii. The notes say the left-then-right fallback matches the reference helper and avoids mixing camera scales, rates, or ROI geometries.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Finite samples are sorted, duplicate timestamps are removed, and the released trace is linearly interpolated at neural-bin centers. Complete finite retained windows are then discretized using per-session tertiles; no filtering or normalization is added.

ii.
```python
good = np.isfinite(times) & np.isfinite(values)
order = np.argsort(times)
keep = np.r_[True, np.diff(times) > 0]
...
whisk_cls, whisk_q = quantile_classes(whisk, valid)
```

iii. The agent states that the released motion energy is already the relevant processed ROI quantity and requires only temporal resampling and task-required categorization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same retained-session 1/3 and 2/3 quantile rule as wheel speed, producing classes 0, 1, and 2.

ii.
```python
whisk_cls, whisk_q = quantile_classes(whisk, valid)
```

iii. The agent uses the same balanced-class rationale and records the thresholds in session metadata.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at every `stimOn_times + CENTERS_REL` query, exactly the grid used for wheel and neural bins.

ii.
```python
q = np.asarray(stim)[:, None] + centers_rel[None, :]
flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The notes rely on synchronized session clocks and reject windows outside camera support instead of extrapolating.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite required trial fields and incomplete behavior windows are removed. Camera loading falls back from left to right. Duplicate/non-finite continuous samples are cleaned. Probe exceptions are logged and skipped; sessions are excluded if no probe yields units, no camera works, or fewer than two trials survive. Exclusion/failure details are stored in metadata.

ii.
```python
except Exception as ex:
    failures.append({'pid': str(row.pid), 'probe': row.probe_name,
                     'error': f'{type(ex).__name__}: {ex}'})
...
if not mats:
    raise RuntimeError('no probe yielded qualified units')
```

iii. The agent emphasizes explicit logging rather than silent substitution and retaining a session when another valid probe or camera stream remains usable.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and processing tens of millions of spikes per probe dominates. The notes report roughly 4.5–7.4 seconds of scientific processing per sample session and introduce session-level multiprocessing for the full conversion.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
```

iii. The agent identifies raw spike volume and the number of native trials as the main cost, not final assembly or pickle writing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning still loops over retained trials and probe loading loops over probes; the trial loop could in principle be vectorized further. Behavior interpolation is already vectorized over all trial/bin queries, while output-list construction remains a cheap per-trial loop.

ii.
```python
for ti, st in enumerate(stim):
    lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
...
q = np.asarray(stim)[:, None] + centers_rel[None, :]
```

iii. The agent says `searchsorted` plus `bincount` avoids generic interval-by-unit loops and that behavior interpolation was vectorized where straightforward.

## 10-c. What processing does the code repeat multiple times?

i. ONE is initialized once per worker, and each worker repeatedly loads one session's trials, wheel, camera, and probe sorting. Within a session, the same interpolation helper and quantile helper are applied separately to wheel and whisker traces; output arrays are constructed per retained trial.

ii.
```python
speed_cls, speed_q = quantile_classes(speed, valid)
whisk_cls, whisk_q = quantile_classes(whisk, valid)
```

iii. The notes specifically avoid repeated pickle writes and concatenate probes only once. They do not identify a substantial avoidable repeated scientific transformation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion, little is deliberately discarded beyond intermediates needed for QC and categorization. Continuous speed/whisker arrays are returned from `process_session` even though only their classes enter the pickle; they are used only by the optional diagnostic plotting path. Detailed trial fields such as feedback and intervals are loaded solely for QC.

ii.
```python
return neural, inputs, outputs, regions, info, speed[idx], whisk[idx]
...
if args.show_processing and len(infos) <= 2:
    processing_plot(eid, n, y, speed, whisk)
```

iii. The agent justifies optional plots as alignment/processing sanity checks. Its notes otherwise claim compact dtypes, QC-before-neural-binning, and no repeated writes minimize discarded work.
