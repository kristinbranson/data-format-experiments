# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI creates a local AllenSDK cache, enumerates locally present NWB experiment IDs, intersects those IDs with the SDK experiment table, then restricts the data to active familiar-image sessions. Each selected experiment (imaging plane) is loaded with `get_behavior_ophys_experiment`; eight worker processes convert sessions in parallel.

ii.
```python
return bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
ids = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
             for f in glob.glob(os.path.join(NWB_DIR, '*.nwb')))
et = get_cache().get_ophys_experiment_table()
et = et.loc[et.index.intersection(ids)]
et = et[~et.passive]
et = et[et.session_type.isin(FAMILIAR_SESSION_TYPES)]
...
ds = bc.get_behavior_ophys_experiment(int(eid))
```

iii. The AI says local-ID intersection prevents fetching absent release files, active sessions are needed for task outcomes, and familiar image-set A matches the paper and gives the same eight images in every session. It loads each NWB once and parallelizes session conversion for speed.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among successfully converted sessions. They are sorted globally, and each session receives an index into that list.

ii.
```python
mouse_id=str(meta['mouse_id'])
...
subjects = sorted({r['mouse_id'] for r in good})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
subject_idx=np.array([subj_to_idx[r['mouse_id']] for r in good], dtype=np.int64)
```

iii. The AI treats the SDK's `mouse_id` as the animal identifier and verifies the resulting mouse count against its independently scanned local subset.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. All experiment IDs/imaging planes sharing that ID are grouped and their neurons concatenated; behavior is read only from the first plane because it is shared across simultaneous planes. Sessions are sorted numerically by session ID.

ii.
```python
for sid, grp in et.groupby('ophys_session_id'):
    sessions.append((int(sid), list(grp.index.values), meta))
sessions.sort(key=lambda s: s[0])
...
for eid in exp_ids:
    ds = bc.get_behavior_ophys_experiment(int(eid))
    planes.append(...)
    if beh is None:
        beh = dict(trials=ds.trials.copy(), ...)
...
neural = np.concatenate(neural_blocks, axis=0)
```

iii. The AI states that planes with the same `ophys_session_id` were recorded simultaneously, have identical behavior streams, and should form one decoder session while retaining per-neuron region labels.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. Go and catch, non-auto-rewarded rows are selected. Every retained trial is represented by a fixed window from 2.25 s before through 3.75 s after `change_time`, divided into eight 750 ms bins.

ii.
```python
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
tr = tr[keep]
change_times = tr['change_time'].values.astype(np.float64)
offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
edges = change_times[:, None] + offs[None, :]
```

iii. The AI chose `change_time` because it is a real change for go trials and sham change for catch trials and always coincides with a flash onset. It chose a fixed window wholly contained in every selected trial and aligned bins to full 750 ms presentation intervals.

## 1-e. How are trials filtered based on quality controls?

i. The AI retains `(go | catch) & ~auto_rewarded`, asserts finite `change_time`, valid outcome partitioning, and that the fixed window fits within trial bounds. It then drops trials with any nonfinite running or pupil bin. A session is skipped if it has no eye tracking or fewer than two usable trials.

ii.
```python
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
assert tr['change_time'].notna().all()
assert np.all(outcome_mat.sum(axis=1) == 1)
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
if good.sum() < 2:
    return dict(session_id=session_id, skip=f'<2 usable trials ({int(good.sum())})', ...)
```

iii. It says go/catch selection inherently excludes aborted trials, free rewards are not task trials, required outputs must be finite, and decoder evaluation requires at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived by default from `BehaviorOphysExperiment.events['events']` and `ophys_timestamps`; `filtered_events` is an optional command-line alternative.

ii.
```python
ev = np.vstack(ds.events[signal].values).astype(np.float64)
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
...
ap.add_argument('--neural-signal', default='events',
                choices=['events', 'filtered_events'])
```

iii. The AI says the paper used detected calcium events and the SDK describes filtered events as visualization smoothing, so unfiltered detected events are the default.

## 2-b. How is the `neural` data processed?

i. Event magnitudes are averaged over imaging frames in each trial/bin, planes are concatenated along the neuron dimension, and the resulting binned trace is z-scored per neuron across all selected trials and bins by default.

ii.
```python
sums, counts = bin_sum_count(p['events'], p['ts'], edges)
neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
neural = np.concatenate(neural_blocks, axis=0)
flat = neural.reshape(neural.shape[0], -1)
mu = flat.mean(axis=1, keepdims=True)
sd = flat.std(axis=1, keepdims=True)
flat = (flat - mu) / sd
```

iii. Mean rather than sum is intended to be frame-rate independent. Z-scoring was added because cell event magnitudes span orders of magnitude and the supplied decoder's raw SVD would otherwise be dominated by large-amplitude cells; decoder experiments favored normalization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no extra per-cell filter. The code uses all cells exposed by the SDK, asserts event/timestamp length agreement, nonempty bins, and finite final neural arrays.

ii.
```python
assert ev.shape[1] == ts.size, 'events / timestamps length mismatch'
assert counts.min() > 0, 'empty time bin (no imaging frame): bin size too small'
out['neural_finite'] = bool(np.isfinite(neural).all())
```

iii. The AI relies on AllenSDK/release ROI and plane QC, reporting that returned cells are valid ROIs with no NaNs or all-zero full traces.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins are aligned to each trial's `change_time`, with a fixed `[-2.25, +3.75)` s window. The same absolute bin-edge matrix is used for all streams.

ii.
```python
edges = change_times[:, None] + offs[None, :]
sums, counts = bin_sum_count(p['events'], p['ts'], edges)
```

iii. The AI cites reference analysis code aligning response tables to `change_time`, notes that go changes and catch sham changes are flash onsets, and uses common timestamped edges to guarantee stream alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Final resolution is 750 ms, eight bins per six-second trial. Native event samples are explicitly rebinned by mean aggregation.

ii.
```python
BIN_SIZE = 0.750
OFF_START = -2.25
OFF_END = 3.75
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The AI originally tested 250 ms but changed to 750 ms because the paper defines a 750 ms image-presentation interval, bin edges then match flash onsets, sparse events gain signal-to-noise, and decoder accuracy improved.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the change-detection rows of `stimulus_presentations`, primarily `start_time`, `image_name`, and `omitted`.

ii.
```python
stim = beh['stim']
stim = stim[stim.stimulus_block_name.astype(str).str.contains('change_detection')]
fstart = stim['start_time'].values.astype(np.float64)
omitted = stim['omitted'].fillna(False).astype(bool).values
names = stim['image_name'].astype(str).values
```

iii. The AI uses the presentation table because image identity is a time-varying stimulus stream, including repeats and omissions, rather than only a two-value trial summary.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted rows are replaced by the previous non-omitted image (with backward fill at the leading edge). Sorted session image names are mapped to integer classes. Each bin takes the identity of the 750 ms presentation interval containing its center.

ii.
```python
names_ff = pd.Series(np.where(omitted, None, names)).ffill().bfill().values
image_names = sorted(set(names_ff[~pd.isna(names_ff)]) - {'omitted'})
name_to_idx = {n: i for i, n in enumerate(image_names)}
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]
```

iii. The AI says an omission replaces a repeat, so ongoing identity remains the prior image. Holding identity over the entire presentation interval follows the paper's convention and accommodates delayed calcium responses.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Bin centers from the same change-aligned edges used for neural aggregation are looked up in sorted flash start times; the selected interval label becomes the output for that bin.

ii.
```python
centers = edges[:, :-1] + BIN_SIZE / 2.0
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]
```

iii. The AI says all streams share the SDK sync clock, and common edges/centers provide exact bin-level alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `stimulus_presentations.is_change`, using `start_time` to locate the presentation interval for each bin center. Missing `is_change` values are treated as false.

ii.
```python
is_change = stim['is_change'].fillna(False).astype(bool).values
j = np.searchsorted(fstart, centers, side='right') - 1
image_change = is_change[j].astype(np.int64)
```

iii. The AI says this directly marks real image changes and naturally leaves sham-change catch trials at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The Boolean `is_change` flag is sampled at each bin center and cast to integer. With 750 ms bins it marks the single presentation interval beginning at a real change.

ii.
```python
image_change = is_change[j].astype(np.int64)
```

iii. It interprets “right after a change” as the complete 750 ms image-presentation interval beginning with the changed flash.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numerical threshold is estimated: the raw Boolean is mapped directly to category 0 (`no_change`) or 1 (`change`).

ii.
```python
image_change = is_change[j].astype(np.int64)
...
['no_change', 'change']
```

iii. The source already supplies a categorical change flag, so further thresholding is unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It uses the same bin centers and presentation lookup as image identity and thus the same change-aligned trial bins as neural data.

ii.
```python
j = np.searchsorted(fstart, centers, side='right') - 1
image_change = is_change[j].astype(np.int64)
```

iii. The AI validates that go trials have a change only in the change interval and catch trials have none.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `running_speed['speed']` and its `timestamps` from the first plane's behavior stream.

ii.
```python
run=ds.running_speed.copy()
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. The AI identifies this as the SDK's filtered running-wheel speed in cm/s and notes simultaneous planes share it.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. A NaN-aware mean is computed from all running samples falling in each 750 ms edge interval. After all sessions are converted, these binned values are digitized using dataset-wide quintile cut points.

ii.
```python
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)
run_all = np.concatenate([r['run_binned'].ravel() for r in good])
_, run_edges_g = quantile_bin(run_all)
run_cls = np.digitize(r['run_binned'], run_edges_g)
```

iii. Mean aggregation matches the common binning scheme. Global percentiles are the literal interpretation of equal percentile bins and keep physical class meanings consistent for the shared decoder readout.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global thresholds at the 20th, 40th, 60th, and 80th percentiles produce five integer classes using `np.digitize`.

ii.
```python
qs = np.linspace(0, 100, nq + 1)[1:-1]
edges = np.percentile(finite, qs)
labels = np.digitize(v, edges, right=False)
```

iii. The AI chose global quintiles for approximately balanced categories and consistent labels across sessions; a session-scoped option remains available but is not the default/full-run choice.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are aggregated directly within the exact absolute timestamp edges used for neural event aggregation.

ii.
```python
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)
sums, counts = bin_sum_count(p['events'], p['ts'], edges)
```

iii. The AI says SDK streams are hardware-synchronized, so using identical time boundaries guarantees bin alignment without first resampling to ophys frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from `eye_tracking['pupil_area']` and `eye_tracking['timestamps']`.

ii.
```python
eye=ds.eye_tracking.copy()
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
```

iii. The AI uses the SDK circular-area convention, interpreting diameter as `2*sqrt(area/pi)` in pixels.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to circular-equivalent diameter. NaN runs with two-sided support and duration at most one second are linearly interpolated; longer/edge gaps remain missing. A NaN-aware mean is taken per 750 ms bin, then values are globally quintile-binned.

ii.
```python
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_INTERP_MAX_GAP)
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
...
pupil_cls = np.digitize(r['pupil_binned'], pupil_edges_g)
```

iii. The AI aims to repair short blink gaps without fabricating long tracking failures, while global quintiles give consistent class meanings across sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four dataset-wide percentile thresholds (20/40/60/80) define five categories via `np.digitize`.

ii.
```python
_, pupil_edges_g = quantile_bin(pupil_all)
pupil_cls = np.digitize(r['pupil_binned'], pupil_edges_g)
```

iii. The same global equal-percentile rationale used for running speed is applied to pupil diameter.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye samples are averaged within the same timestamp edge matrix used to aggregate neural events.

ii.
```python
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
```

iii. The AI relies on synchronized SDK timestamps and common bins for alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table Boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
```

iii. The AI describes these as the natural four mutually exclusive task outcomes and asserts that they partition all retained trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` maps the one-hot outcome flags to codes 0–3, and the static trial code is broadcast over all eight time bins.

ii.
```python
assert np.all(outcome_mat.sum(axis=1) == 1)
outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
...
np.full(NBINS, outcome[i], dtype=np.int64)
```

iii. Broadcasting satisfies the common `(n_output, time)` output layout while preserving a static per-trial target.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye tracking drops the session. Short pupil NaN gaps are interpolated, but trials with any remaining nonfinite running or pupil bin are dropped; sessions with fewer than two survivors are skipped. Nullable stimulus Booleans are filled false, omitted image identity is forward-filled, zero-variance neural SD is set to one, and conversion exceptions skip the session while retaining a traceback.

ii.
```python
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', ...)
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
sd[sd == 0] = 1.0
is_change = stim['is_change'].fillna(False).astype(bool).values
except Exception as ex:
    return dict(..., skip='ERROR: ' + repr(ex), traceback=traceback.format_exc())
```

iii. The AI distinguishes brief blinks from long tracking failures, avoids imputing required outputs when unsupported, and uses assertions/sanity checks to fail loudly rather than silently corrupt data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading large NWB experiments through AllenSDK is the dominant cost; neural binning and optional plotting are additional costs. The code records separate load, neural, stimulus, behavior, and total timings and parallelizes sessions.

ii.
```python
t0 = time.time()
ds = bc.get_behavior_ophys_experiment(int(eid))
...
timings['load'] = time.time() - t0
...
with Pool(min(args.workers, len(jobs))) as pool:
    for i, r in enumerate(pool.imap_unordered(convert_session, jobs)):
```

iii. The AI measured about three seconds per plane and estimated that one-open-per-plane plus eight workers reduced the full conversion to roughly one to three minutes (observed about 69 seconds).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes the expensive neuron-by-trial-by-bin aggregation using `searchsorted` and cumulative sums. Remaining small loops include loading planes, assembling per-trial arrays, assigning discretized outputs, constructing region indices, gap-run interpolation, and sanity-check list comprehensions; trial assembly/output assignment could be reduced further but is unlikely to dominate I/O.

ii.
```python
idx = np.searchsorted(timestamps, edges_flat)
cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
sums = cs[:, idx[:, 1:]] - cs[:, idx[:, :-1]]
...
for i in range(n_good):
    output_trials.append(np.stack([...]))
```

iii. Its notes explicitly claim a 10–50x gain from replacing per-trial/per-bin masks with cumulative-sum binning and regard file I/O as the larger bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Each plane is opened once, but every worker independently constructs an AllenSDK cache object. Common binning machinery is called repeatedly for each plane and for running/pupil (twice each in NaN-aware mode). Several full-data passes are made for global quantiles, output insertion, class fractions, assembly, and summaries.

ii.
```python
bc = get_cache()
...
for p in planes:
    sums, counts = bin_sum_count(...)
...
sums, _ = bin_sum_count(np.where(valid, values, 0.0), timestamps, edges)
cnts, _ = bin_sum_count(valid.astype(np.float64), timestamps, edges)
```

iii. The AI emphasizes that it avoids the most serious repetition: NWBs are opened once per plane and behavior streams are copied only from the first plane. It accepts repeated lightweight array passes to support global discretization, checks, and metadata.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `depths` and `cell_ids` are collected into each session result but never included in the final dataset. Raw plotting payloads and six-panel plots are produced only under `--show-processing` and then deleted from results. Extensive sanity-check statistics and session metadata remain in the pickle but are ignored by decoder training. The code also creates zero placeholders for running/pupil before overwriting them after global threshold computation.

ii.
```python
depth_list += [p['depth']] * p['events'].shape[0]
cellid_list += list(p['cell_ids'])
...
np.zeros(NBINS, dtype=np.int64)  # placeholder: running quintile
...
if '_plotargs' in r:
    plot_processing(r, run_edges, pupil_edges)
    del r['_plotargs']
```

iii. The AI justifies plots and checks as validation aids and placeholders as a convenient two-pass assembly design; it does not explicitly justify retaining unused depth/cell-ID arrays during conversion.
