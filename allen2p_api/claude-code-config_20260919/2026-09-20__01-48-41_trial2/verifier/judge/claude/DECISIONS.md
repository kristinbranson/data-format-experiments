# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the `VisualBehaviorOphysProjectCache` from AllenSDK, opened via `from_local_cache()`. It calls `get_ophys_experiment_table()` to get the metadata, filters to locally available NWB files and `project_code == 'VisualBehavior'`, and also filters out passive sessions (`~sub.passive`). Each experiment is loaded via `cache.get_behavior_ophys_experiment(ophys_experiment_id)`.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)

def select_experiments(cache):
    et = cache.get_ophys_experiment_table()
    sub = et.loc[locally_available_experiment_ids()]
    sel = sub[(~sub.passive) & (sub.project_code == 'VisualBehavior')]
    return sel.sort_index()
```

iii. The AI documented (CONVERSION_NOTES D1, D2) that single-plane VisualBehavior experiments all run at 30.95 Hz, and passive sessions lack go/catch trial structure. It checks which NWB files are locally available before filtering.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `metadata['mouse_id']` from each successfully converted experiment. A unique ordered list of mouse IDs is built during assembly.

ii.
```python
subjects, brain_regions = [], []
for r in ok:
    if r['mouse_id'] not in subjects:
        subjects.append(r['mouse_id'])
```

iii. Each experiment carries mouse metadata from the SDK. The AI collects unique mouse IDs as results come in from the parallel conversion.

## 1-c. How are the data split into sessions?

i. Each ophys experiment is treated as one session. Since VisualBehavior uses single-plane imaging, there is exactly one experiment per session. The AI does not group experiments by `ophys_session_id`.

ii.
```python
eids = list(meta.index)  # each experiment ID is a session
# ...
def convert_experiment(ophys_experiment_id, collect_debug=False):
    """Convert one ophys experiment (= one session) into the target structure."""
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. The AI documented (CONVERSION_NOTES D1) that for the VisualBehavior project, each session has exactly one imaging plane (one experiment), so treating each experiment as a session is equivalent to grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `ds.trials` table. The AI selects rows where `go == True` or `catch == True`, which excludes aborted and auto-rewarded trials (these four categories are mutually exclusive in the SDK). Trial windows span `[start_time, stop_time)`, with frames found via `np.searchsorted`. Trials with fewer than 2 ophys frames are dropped.

ii.
```python
trials = ds.trials
sel = trials[trials['go'].to_numpy().astype(bool)
             | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
# ...
a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
keep = (b_idx - a_idx) >= 2
```

iii. The AI verified (CONVERSION_NOTES Step 1) that `go`, `catch`, `aborted`, and `auto_rewarded` are mutually exclusive in the SDK code (`trial.py`), so `go | catch` is equivalent to excluding aborted and auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) selecting only `go | catch` (excluding aborted and auto-rewarded), (2) requiring at least 2 ophys frames per trial, (3) sessions with no eye-tracking data are dropped entirely. Sessions are not further filtered by minimum trial count (but the format requires >= 2).

ii.
```python
sel = trials[trials['go'].to_numpy().astype(bool)
             | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
# ...
keep = (b_idx - a_idx) >= 2
# ...
if eye is None or len(eye) == 0:
    raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
```

iii. The AI documented (CONVERSION_NOTES D10) that 3 sessions lacked eye-tracking data and were excluded since pupil diameter is a required output. The minimum 2-frame trial threshold prevents degenerate trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces), accessed via `ds.dff_traces['dff']`.

ii.
```python
dff = ds.dff_traces
traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)   # (N, F)
```

iii. The AI documented (CONVERSION_NOTES D3) that while the reference paper uses detected calcium events, dF/F was chosen because events are non-zero at only 0.23% of (cell, frame) pairs, making per-timepoint decoding impossible. A pilot comparison showed dF/F gives much better decoder accuracy.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are stacked into a (N_neurons, F) matrix. Each experiment has a single imaging plane, so no cross-plane merging is needed (unlike the reference which groups by session). Data is cast to float32.

ii.
```python
dff = ds.dff_traces
cell_ids = dff.index.to_numpy()
traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)   # (N, F)
```

iii. The Allen pipeline already applies motion correction, neuropil subtraction, dF/F normalization, and detrending. No additional processing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI defensively re-applies the `valid_roi` filter from `cell_specimen_table`, although all cells in the release already have `valid_roi == True`.

ii.
```python
cst = ds.cell_specimen_table
if not bool(cst['valid_roi'].all()):
    keep = cst['valid_roi'].to_numpy().astype(bool)
    traces = traces[keep]
    cell_ids = cell_ids[keep]
    n_neurons = traces.shape[0]
```

iii. The AI verified (CONVERSION_NOTES D11) that `valid_roi` is True for all 29,097 cells. The defensive check adds no filtering in practice but guards against unexpected edge cases.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. Per-trial data spans `[start_time, stop_time)` using `np.searchsorted` to find frame indices. Variable-length trial windows are used.

ii.
```python
a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
# ...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The instructions say "Temporally align based on ophys timestamp", and the AI uses the native ophys frame grid without resampling. All data streams are placed on the same ophys timestamp grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native ophys frame rate (~30.95 Hz, ~32.3 ms) is used. The time bin size is computed as the mean of median inter-frame intervals across sessions.

ii.
```python
dts = np.array([r['info']['ophys_frame_interval_s'] for r in ok])
# ...
'time_bin_size': float(np.mean(dts) * 1000.0),
```

iii. All single-plane VisualBehavior sessions share the same frame rate (sd 0.0055 ms across sessions), so no rebinning is needed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, specifically the `image_name`, `start_time`, and `is_change` columns, filtered to the `change_detection_behavior` stimulus block.

ii.
```python
def build_stimulus_series(stimulus_presentations, ophys_timestamps):
    sp = stimulus_presentations
    sp = sp[sp['stimulus_block_name'] == STIM_BLOCK].sort_values('start_time')
    start = sp['start_time'].to_numpy(dtype=float)
    names = sp['image_name'].to_numpy()
    # ...
    raw = np.array([IMAGE_TO_IDX.get(n, -1) if isinstance(n, str) else -1
                    for n in names], dtype=np.int64)
```

iii. The AI uses the stimulus presentations table to assign each ophys frame to the image-presentation interval that contains it. Omitted flashes are forward-filled with the previous image identity. This approach maps each frame to the stimulus interval `[start_time_i, start_time_{i+1})`, consistent with the reference paper's 750ms "image-presentation interval".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a fixed global mapping of 16 images (both image sets A and B). Omitted flashes (`image_name == 'omitted'`) are forward-filled with the previous real image. Each ophys frame is assigned to a presentation interval using vectorized `np.searchsorted`.

ii.
```python
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063',
               'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_NAMES)}
# ...
# forward-fill the omitted (-1) entries with the previous image identity
valid_pos = np.where(raw >= 0)[0]
fill_src = valid_pos[np.searchsorted(valid_pos, np.arange(len(raw)), side='right') - 1]
filled = raw[fill_src]

k = np.searchsorted(start, ophys_timestamps, side='right') - 1
k = np.clip(k, 0, len(start) - 1)
return filled[k], change[k].astype(np.int64), k
```

iii. The AI documented (CONVERSION_NOTES D5, D6) that holding the image identity through gray screens and omissions is consistent with the reference paper's definition, and that using 16 global image names (rather than 8 per-session) ensures consistent labels across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at every ophys frame timestamp by mapping each frame to its containing stimulus presentation interval using `np.searchsorted`. Since the ophys timestamps are the time base for all data, alignment is automatic.

ii.
```python
k = np.searchsorted(start, ophys_timestamps, side='right') - 1
k = np.clip(k, 0, len(start) - 1)
return filled[k], change[k].astype(np.int64), k
# ...
out[0] = image_idx[a:b]
```

iii. The same ophys timestamp indices are used for neural data and all outputs.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `stimulus_presentations`, filtered to the `change_detection_behavior` block.

ii.
```python
change = sp['is_change'].to_numpy().astype(bool)
# ...
return filled[k], change[k].astype(np.int64), k
```

iii. The `is_change` flag from `stimulus_presentations` marks flashes where the image identity changed from the previous flash. This naturally excludes catch trials (sham changes).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is converted to int (0/1). Each ophys frame inherits the `is_change` value of its containing presentation interval. This means all frames within a change-flash interval (typically ~750ms) are marked as 1.

ii.
```python
change = sp['is_change'].to_numpy().astype(bool)
# In build_stimulus_series:
return filled[k], change[k].astype(np.int64), k
```

iii. No additional processing beyond the boolean-to-int conversion.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii. N/A — it is directly derived as a binary variable.

iii. The categories are `['no_change', 'change']`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — each ophys frame is mapped to a stimulus presentation interval, and inherits the `is_change` flag of that interval.

ii.
```python
out[1] = is_change[a:b]
```

iii. The same frame-to-interval mapping ensures alignment with neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, which provides the 60 Hz running wheel speed trace (already low-pass filtered at 10 Hz by the Allen pipeline).

ii.
```python
rs = ds.running_speed
run_t = rs['timestamps'].to_numpy(dtype=float)
run_v = rs['speed'].to_numpy(dtype=float)
```

iii. The SDK's `running_speed` is used as-is.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the ophys timestamps using `np.interp` (dropping non-finite values first), then discretized into 5 quintile bins computed per session.

ii.
```python
run_ok = np.isfinite(run_v)
run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
# ...
frame_mask = np.zeros(n_frames, dtype=bool)
for a, b in zip(a_idx, b_idx):
    frame_mask[a:b] = True
run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
```

iii. The AI used `np.interp` for resampling (handles edge values by clamping rather than producing NaN). Discretization uses per-session quintile edges computed from only the frames that end up in trials.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile (quintile) bins. Bin edges are the 20th, 40th, 60th, and 80th percentiles, computed per session over trial-included frames only.

ii.
```python
QUANTILE_PCTS = [20., 40., 60., 80.]
run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
```

iii. The AI documented (CONVERSION_NOTES D8) that per-session quintiles are used because pupil diameter varies in camera pixels across sessions/rigs, and the same convention is used for running speed for consistency.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the ophys timestamps before trial segmentation, so it shares the same frame indices as neural data.

ii.
```python
run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
# ...
out[2] = run_bin[a:b]
```

iii. Same ophys-frame-based alignment as all other streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, using both `pupil_width` and `pupil_height` columns. Diameter is computed as `2 * max(pupil_width, pupil_height)`.

ii.
```python
def pupil_diameter_series(eye_tracking):
    diam = 2.0 * np.maximum(eye_tracking['pupil_width'].to_numpy(dtype=float),
                            eye_tracking['pupil_height'].to_numpy(dtype=float))
    t = eye_tracking['timestamps'].to_numpy(dtype=float)
    good = np.isfinite(diam)
    return t, diam, good
```

iii. The AI documented that the AllenSDK computes `pupil_area = pi * max(pupil_width, pupil_height)^2`, so `max(width, height)` is the radius and `2 * max(width, height)` is the diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from width/height, blink/outlier frames (where values are NaN) are dropped, the remaining values are linearly interpolated onto ophys timestamps using `np.interp`, then discretized into 5 per-session quintile bins.

ii.
```python
eye_t, eye_d, eye_ok = pupil_diameter_series(eye)
if eye_ok.sum() < 100:
    raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')
pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
# ...
pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
```

iii. Blink frames are dropped by checking `np.isfinite(diam)` (the SDK sets values to NaN on `likely_blink` frames). `np.interp` linearly interpolates across the gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — 5 equal percentile (quintile) bins computed per session.

ii.
```python
pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
```

iii. Per-session quintiles ensure each session has balanced bin distributions, removing cross-session confounds from camera-pixel-based measurements.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto ophys timestamps before trial segmentation, sharing the same frame indices as neural data.

ii.
```python
pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
# ...
out[3] = pupil_bin[a:b]
```

iii. Same alignment approach as running speed and all other outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
# ...
outcome = np.full(len(sel), -1, dtype=np.int64)
for j, name in enumerate(OUTCOME_NAMES):
    outcome[sel[name].to_numpy().astype(bool)] = j
if np.any(outcome < 0):
    raise ValueError(f'{ophys_experiment_id}: unclassified go/catch trial')
```

iii. The AI verifies that every go/catch trial has exactly one of the four outcomes, raising an error if any trial is unclassified.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The outcome is a per-trial static value, broadcast as a constant across all frames of the trial.

ii.
```python
out[4] = outcome[i]  # broadcast scalar across all T frames
```

iii. The outcome is stored as row 4 of the (5, T) output array with the same value at every time step within a trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If conversion throws an exception, the experiment is skipped with an error message.
- **Missing eye tracking**: Sessions with no or too-few eye-tracking frames are raised as errors and skipped (3 sessions).
- **Blink frames**: Pupil values are NaN on blink frames; these are dropped and interpolated through.
- **Non-finite running speed**: Non-finite values are dropped before interpolation.
- **Short trials**: Trials with < 2 ophys frames are dropped.
- **Omitted stimulus flashes**: Image identity is forward-filled from the previous flash.
- **Trial starting before first flash**: The presentation index is clamped to `[0, n_flashes-1]`.

ii.
```python
# Missing eye tracking
if eye is None or len(eye) == 0:
    raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
if eye_ok.sum() < 100:
    raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')

# Short trials
keep = (b_idx - a_idx) >= 2

# Omissions forward-fill
valid_pos = np.where(raw >= 0)[0]
fill_src = valid_pos[np.searchsorted(valid_pos, np.arange(len(raw)), side='right') - 1]
filled = raw[fill_src]

# Clamping for edge case
k = np.clip(k, 0, len(start) - 1)
```

iii. The AI documented edge cases extensively in CONVERSION_NOTES Step 10 Check 5.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads large NWB files. The AI measured ~3.5-5.0 seconds per experiment. Multiprocessing with up to 32 workers reduces wall-clock time to ~34 seconds for 168 experiments.

ii.
```python
with Pool(min(args.workers, max(1, len(jobs)))) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
```

iii. The AI documented timing in CONVERSION_NOTES Step 7.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial assembly loop iterates over each trial to slice arrays, but this is already efficient since it uses numpy slicing. The `build_stimulus_series` function is fully vectorized using `np.searchsorted`. The per-frame image mapping avoids any Python loops over frames.

ii.
```python
# Vectorized stimulus mapping
k = np.searchsorted(start, ophys_timestamps, side='right') - 1
k = np.clip(k, 0, len(start) - 1)
return filled[k], change[k].astype(np.int64), k
```

iii. The AI noted that NWB loading dominates runtime, making further vectorization of the trial loop negligible.

## 9-c. What processing does the code repeat multiple times?

i. Each experiment is loaded independently in its own worker process, which means the cache object is re-created in each worker. However, this is by design for parallelism. No data processing is repeated unnecessarily within a single experiment's conversion.

ii.
```python
def convert_experiment(ophys_experiment_id, collect_debug=False):
    cache = get_cache()  # re-created per worker call
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. The per-worker cache creation is a tradeoff for parallel execution.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects extensive debug information (`collect_debug=True`) for up to 2 sessions when `--show-processing` is used, including raw traces for up to 30 neurons, full stimulus presentations, lick times, reward times, etc. This is discarded after plotting and not included in the output pickle. Per-session metadata (`info` dict) is stored in `metadata['session_info']` which may be more than needed for downstream analysis but is useful for documentation.

ii.
```python
if collect_debug:
    result['debug'] = {
        'ts': ts, 'traces_mean': traces.mean(axis=0),
        'traces': traces[:min(30, n_neurons)],
        # ... extensive debug data
    }
```

iii. The debug data serves the `--show-processing` diagnostic plots and is not included in the final output.
