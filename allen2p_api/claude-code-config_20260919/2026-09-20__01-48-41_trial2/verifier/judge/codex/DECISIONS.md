# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent opens the local AllenSDK cache, intersects the SDK experiment table with locally present NWB experiment IDs, selects active `VisualBehavior` experiments, and loads every selected experiment through `get_behavior_ophys_experiment`, in a multiprocessing pool. It does not open NWB contents directly.

ii.
```python
return VisualBehaviorOphysProjectCache.from_local_cache(
    cache_dir=CACHE_DIR, use_static_cache=False)
sub = et.loc[locally_available_experiment_ids()]
sel = sub[(~sub.passive) & (sub.project_code == 'VisualBehavior')]
ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. The notes justify local-ID restriction because only 284 files are available, `VisualBehavior` because it has a common 30.95 Hz single-plane grid, and active-only selection because passive recordings do not contain the requested go/catch behavioral structure. Parallel loading addresses the dominant I/O cost.

## 1-b. How are the data split into subjects?

i. Subject identity comes from each experiment's SDK metadata `mouse_id`; the final unique subject list is built in result order and each session gets an index into it.

ii.
```python
'mouse_id': str(md['mouse_id'])
if r['mouse_id'] not in subjects:
    subjects.append(r['mouse_id'])
'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok])
```

iii. The agent treats the SDK mouse identifier as the canonical animal identity.

## 1-c. How are the data split into sessions?

i. Each selected single-plane ophys experiment is one output session. Experiments are sorted by experiment ID; unlike a multiscope solution, planes are not merged because multiscope experiments were excluded.

ii.
```python
return sel.sort_index()
def convert_experiment(ophys_experiment_id, collect_debug=False):
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. The notes state that the selected `VisualBehavior` variant has one experiment/plane per session and a uniform frame interval; this motivated excluding the locally available multiscope subset.

## 1-d. How are the data split into trials?

i. SDK trial rows classified as go or catch are sorted by start time. Each converted trial contains all ophys frames in the half-open interval `[start_time, stop_time)`; boundaries are found with vectorized `searchsorted`.

ii.
```python
sel = trials[trials['go'].to_numpy().astype(bool)
             | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
```

iii. The notes cite SDK trial definitions and tutorials, and choose the experiment-defined full trial because the outputs are time-varying and include pre-change context and the response period.

## 1-e. How are trials filtered based on quality controls?

i. Only `go | catch` trials are retained, thereby excluding aborted and auto-rewarded trials. Trials with fewer than two ophys frames are dropped; sessions that cannot be converted (notably missing/insufficient eye data) are skipped.

ii.
```python
sel = trials[trials['go'].to_numpy().astype(bool)
             | trials['catch'].to_numpy().astype(bool)]
keep = (b_idx - a_idx) >= 2
a_idx, b_idx = a_idx[keep], b_idx[keep]
```

iii. The agent verified that go, catch, aborted, and auto-rewarded are mutually exclusive and that retained trials map to exactly one outcome. The two-frame minimum protects downstream arrays.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from each experiment's AllenSDK `dff_traces['dff']` and aligned using `ophys_timestamps`.

ii.
```python
ts = np.asarray(ds.ophys_timestamps, dtype=float)
dff = ds.dff_traces
traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)
```

iii. Although the paper used detected calcium events, the agent chose dF/F because events were nonzero in only 0.23% of cell-frame entries and performed poorly for native-frame decoding; it documents this as a deliberate deviation.

## 2-b. How is the `neural` data processed?

i. Precomputed dF/F rows are stacked, converted to `float32`, checked against the timestamp count, and sliced into contiguous neuron-by-time trial matrices. No normalization, smoothing, or rebinning is added.

ii.
```python
traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)
if traces.shape[1] != n_frames:
    raise ValueError(...)
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes rely on the Allen pipeline's demixing, neuropil subtraction, baseline normalization, and detrending, avoiding duplicate processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The SDK's valid ROIs are used. A defensive `valid_roi` filter is applied only if the cell specimen table contains invalid rows; no additional cell threshold is imposed.

ii.
```python
cst = ds.cell_specimen_table
if not bool(cst['valid_roi'].all()):
    keep = cst['valid_roi'].to_numpy().astype(bool)
    traces = traces[keep]
```

iii. The agent reports that all 29,097 exposed cells were valid and that release-level segmentation and session QC had already removed unsuitable ROIs/sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural frames are indexed on the native ophys clock and each trial is aligned to its SDK `start_time`, ending just before `stop_time`; there is no fixed change-centered window.

ii.
```python
a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. All timestamps share the synchronized session clock, and full trial alignment preserves both pre- and post-change labels.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain on the native single-plane ophys grid, about 32.319 ms (30.95 Hz). The reported bin size is the mean of per-session median frame intervals; no neural rebinning is performed.

ii.
```python
'ophys_frame_interval_s': float(np.median(np.diff(ts)))
'time_bin_size': float(np.mean(dts) * 1000.0)
```

iii. The agent found only 0.06% frame-interval variation and judged native ophys timestamps to satisfy the common-bin requirement.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `stimulus_presentations.image_name`, `start_time`, and `stimulus_block_name`, not the trial table's initial/change image columns.

ii.
```python
sp = sp[sp['stimulus_block_name'] == STIM_BLOCK].sort_values('start_time')
start = sp['start_time'].to_numpy(dtype=float)
names = sp['image_name'].to_numpy()
```

iii. The notes say presentation records provide the actual flash-by-flash stimulus, including omissions, and match the paper's 750 ms presentation-interval analysis unit.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The 16 fixed global image names map to integer categories. Omitted/invalid presentation names receive `-1` temporarily and are forward-filled from the latest real image; the category is held through the gray interval.

ii.
```python
raw = np.array([IMAGE_TO_IDX.get(n, -1) if isinstance(n, str) else -1
                for n in names], dtype=np.int64)
valid_pos = np.where(raw >= 0)[0]
fill_src = valid_pos[np.searchsorted(valid_pos, np.arange(len(raw)), side='right') - 1]
filled = raw[fill_src]
```

iii. A global vocabulary prevents the same code meaning different images across image sets. Holding the last image avoids a dominant gray category and keeps the requested variable restricted to real image identities.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Every ophys timestamp is assigned to the most recent stimulus-presentation start, and the resulting full-session series is sliced with exactly the same trial indices as neural activity.

ii.
```python
k = np.searchsorted(start, ophys_timestamps, side='right') - 1
k = np.clip(k, 0, len(start) - 1)
return filled[k], change[k].astype(np.int64), k
out[0] = image_idx[a:b]
```

iii. This makes output and neural samples share the ophys frame clock and gives each flash its full presentation interval.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change derives from `stimulus_presentations.is_change` and presentation start times within the change-detection block.

ii.
```python
change = sp['is_change'].to_numpy().astype(bool)
return filled[k], change[k].astype(np.int64), k
```

iii. The presentation table directly identifies real image changes and distinguishes them from catch/sham events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The boolean presentation flag is converted to integer and held for the complete interval beginning with that presentation (normally 750 ms).

ii.
```python
k = np.searchsorted(start, ophys_timestamps, side='right') - 1
is_change = change[k].astype(np.int64)
```

iii. The notes interpret “right after a change” using the paper's 750 ms image-presentation interval, covering the calcium response rather than a single frame.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: `False` becomes 0 (`no_change`) and `True` becomes 1 (`change`); no numerical threshold is estimated.

ii.
```python
CHANGE_NAMES = ['no_change', 'change']
change = sp['is_change'].to_numpy().astype(bool)
```

iii. The raw SDK flag supplies the requested binary categorization.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change flag is expanded to ophys frames through the same presentation index and sliced by the same `[a:b]` trial bounds as neural data.

ii.
```python
image_idx, is_change, interval = build_stimulus_series(ds.stimulus_presentations, ts)
out[1] = is_change[a:b]
```

iii. Thus it is exactly sample-aligned to the neural matrix.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `dataset.running_speed['timestamps']` and `['speed']`.

ii.
```python
rs = ds.running_speed
run_t = rs['timestamps'].to_numpy(dtype=float)
run_v = rs['speed'].to_numpy(dtype=float)
```

iii. The notes identify this as the SDK's already unwrapped, transient-corrected, 10 Hz low-pass-filtered encoder speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite speed samples are linearly interpolated to all ophys timestamps, then converted into per-session quintile labels calculated only from frames belonging to retained trials.

ii.
```python
run_ok = np.isfinite(run_v)
run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
```

iii. Interpolation synchronizes streams. Per-session bins guarantee approximately balanced labels and mirror the pupil treatment.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four per-session thresholds at the 20th, 40th, 60th, and 80th percentiles create five integer categories, with threshold-equal values assigned to the higher bin.

ii.
```python
QUANTILE_PCTS = [20., 40., 60., 80.]
run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
```

iii. This implements the requested five equal-percentile bins at session level.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly at `ophys_timestamps`, categorized, then sliced by the same trial indices.

ii.
```python
run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
out[2] = run_bin[a:b]
```

iii. The common synchronized clock makes every running label correspond to one neural frame.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter derives from eye-tracking timestamps plus both `pupil_width` and `pupil_height`; finite values implicitly exclude SDK blink/outlier NaNs.

ii.
```python
diam = 2.0 * np.maximum(eye_tracking['pupil_width'].to_numpy(dtype=float),
                        eye_tracking['pupil_height'].to_numpy(dtype=float))
t = eye_tracking['timestamps'].to_numpy(dtype=float)
good = np.isfinite(diam)
```

iii. The agent traces the SDK circular-area definition and interprets `max(width,height)` as radius, hence diameter `2*max(width,height)`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink/outlier NaNs are removed, diameter is linearly interpolated onto ophys frames, and retained-trial values are discretized using per-session quintiles.

ii.
```python
eye_t, eye_d, eye_ok = pupil_diameter_series(eye)
pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
```

iii. Interpolation fills brief blink gaps; per-session percentiles avoid camera-pixel scale differences becoming session identity.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four session-specific 20/40/60/80 percentile edges make five bins via right-sided `searchsorted`.

ii.
```python
pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
```

iii. The notes argue that global pixel thresholds would mostly identify the camera/session rather than within-session pupil state.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Clean eye measurements are interpolated at native ophys timestamps; binned values are sliced with the same trial boundaries.

ii.
```python
pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
out[3] = pupil_bin[a:b]
```

iii. This places eye and neural samples on the same synchronized frame grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the retained trial table's boolean `hit`, `miss`, `false_alarm`, and `correct_reject` columns.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
for j, name in enumerate(OUTCOME_NAMES):
    outcome[sel[name].to_numpy().astype(bool)] = j
```

iii. The agent verified that each retained go/catch trial has exactly one of the four canonical SDK outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Each boolean outcome is mapped to its fixed 0–3 index; any unclassified trial raises an error. The scalar category is broadcast across every frame of that trial.

ii.
```python
if np.any(outcome < 0):
    raise ValueError(...)
out[4] = outcome[i]
```

iii. Broadcasting supplies a time-shaped categorical output while retaining a static per-trial meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Shape inconsistencies and unclassified trials raise errors; empty or severely insufficient eye data cause the whole session to be skipped and recorded; nonfinite running samples and pupil blink/outlier samples are omitted before interpolation; too-short trials are removed; pre-first-presentation frame indices are clipped; omitted images are forward-filled. Worker exceptions do not abort other sessions.

ii.
```python
if eye is None or len(eye) == 0: raise ValueError(...)
if eye_ok.sum() < 100: raise ValueError(...)
k = np.clip(k, 0, len(start) - 1)
keep = (b_idx - a_idx) >= 2
except Exception as exc:
    return {'error': f'{type(exc).__name__}: {exc}', 'eid': int(eid)}
```

iii. The notes document three skipped no-eye sessions and a corrected pre-first-flash `-1` image-label bug. The policy favors complete aligned outputs over inventing a pupil label.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and parsing large experiments through AllenSDK dominates; pickle serialization and optional diagnostic plotting are secondary. The code uses 24 worker processes by default.

ii.
```python
with Pool(min(args.workers, max(1, len(jobs)))) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
        ...
pickle.dump(data, fh, protocol=4)
```

iii. Notes report 168 sessions converted in 4.8 minutes with 24 workers and identify SDK loading as the main cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most heavy mappings are already vectorized. Remaining loops include marking each retained trial in `frame_mask`, slicing/assembling variable-length trial arrays, assigning four outcome columns, collecting unique metadata, and optional plotting. The mask loop could be vectorized with difference-array/cumulative-sum logic; variable-length slicing is naturally loop-based.

ii.
```python
for a, b in zip(a_idx, b_idx):
    frame_mask[a:b] = True
for i, (a, b) in enumerate(zip(a_idx, b_idx)):
    neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes emphasize that presentation mapping and resampling were vectorized with `searchsorted`/`interp`, reducing mapping from about 1.5 s to 0.05 s per session.

## 9-c. What processing does the code repeat multiple times?

i. Every worker independently opens the cache and loads one experiment; trial boundaries are traversed once to build the frame mask and again to assemble trials. Optional debug mode retains/revisits the same streams for plots, and final summary concatenates outputs again.

ii.
```python
cache = get_cache()
for a, b in zip(a_idx, b_idx): frame_mask[a:b] = True
for i, (a, b) in enumerate(zip(a_idx, b_idx)): ...
allout = np.concatenate([o for r in ok for o in r['output']], axis=1)
```

iii. Repetition is limited and intentional: per-process cache objects avoid unsafe sharing, while separate mask and assembly passes keep the code simple.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `build_stimulus_series` computes and returns `interval`, but conversion never uses it. `cell_specimen_ids` are carried in intermediate results but omitted from the final dataset. `sel_kept` is only needed for optional debug data, and extensive `info` statistics are metadata rather than decoder inputs. With `--show-processing`, large debug arrays and plots are diagnostic only.

ii.
```python
image_idx, is_change, interval = build_stimulus_series(...)
'cell_specimen_ids': cell_ids,
sel_kept = sel[keep]
```

iii. The diagnostics support validation and provenance; the unused `interval` and discarded cell IDs are small avoidable overheads relative to experiment loading and neural arrays.
