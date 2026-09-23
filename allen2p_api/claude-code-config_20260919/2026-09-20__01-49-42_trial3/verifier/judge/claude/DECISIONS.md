# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses AllenSDK's `VisualBehaviorOphysProjectCache.from_local_cache()` to open the on-disk cache. It calls `get_ophys_experiment_table()` to get the master experiment table, filters to `project_code == 'VisualBehavior'` and `passive == False`, then intersects with locally present NWB files. For each selected experiment, it calls `get_behavior_ophys_experiment(oeid)` to load the full data.

ii.
```python
def get_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = bpc.VisualBehaviorOphysProjectCache.from_local_cache(
            cache_dir=CACHE_DIR)
    return _CACHE

def select_experiments():
    et = get_cache().get_ophys_experiment_table()
    ids = local_experiment_ids()
    sel = et[(et.index.isin(ids)) &
             (et.project_code == PROJECT_CODE) &
             (~et.passive)].copy()
    return sel.sort_index()
```

iii. The AI chose `from_local_cache` because there is no network access available. Filtering by `project_code == 'VisualBehavior'` selects the named dataset variant. Filtering out passive sessions was justified because passive sessions have a retracted lick spout and hence no go/catch trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values across all selected experiments. They are collected as a sorted set of unique mouse IDs from the kept sessions.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_idx = np.array([subjects.index(s['mouse_id']) for s in sessions], dtype=np.int64)
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a single `ophys_experiment_id` (one imaging plane / one NWB file). The AI treats each experiment as a separate session. This is valid because the VisualBehavior project uses single-plane imaging, so each session has exactly one experiment.

ii.
```python
def extract_session(oeid):
    ds = get_cache().get_behavior_ophys_experiment(oeid)
    ophys_ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    # ... processes one experiment as one session
```

iii. The AI justified this by noting that in the VisualBehavior project, there is exactly one imaging plane per ophys session, so session == experiment == NWB file. This differs from the reference which groups experiments by `ophys_session_id`, but the result is equivalent for this dataset.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `ds.trials` table. The AI selects trials where `go == True` or `catch == True`, spanning `[start_time, stop_time)`. This gives variable-length trials.

ii.
```python
trials = ds.trials
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
# ...
for k, (_, tr) in enumerate(sel.iterrows()):
    i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
    if i1 - i0 < 2:
        continue
```

iii. The AI verified that `go` and `catch` are mutually exclusive with `aborted` and `auto_rewarded` in the SDK's `Trial._get_trial_data()` logic, so filtering by `go | catch` is equivalent to filtering out aborted and auto-rewarded trials. Assertions confirm no selected trial is aborted or auto-rewarded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) `go | catch` (excludes aborted and auto-rewarded), (2) trials with fewer than 2 ophys frames are dropped, (3) sessions without eye tracking are dropped entirely (3 sessions), (4) sessions with fewer than 2 usable trials are dropped.

ii.
```python
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
assert not sel.aborted.any()
assert not sel.auto_rewarded.any()
# ...
if i1 - i0 < 2:
    continue
# ...
sessions = [r for r in results if not r['skip'] and r['n_trials_kept'] >= 2]
```

iii. Filtering out passive sessions and sessions without eye tracking were additional curation steps beyond the basic trial filtering. The AI documented that 3 sessions were dropped due to empty eye-tracking tables (pupil diameter is a required output).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `ds.dff_traces` (detrended dF/F calcium fluorescence traces), accessed via `ds.dff_traces['dff']`.

ii.
```python
ev = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
```

iii. The AI chose dF/F over detected calcium events after an empirical comparison. The whitepaper's processing chain ends at detrended dF/F, and at the 32 ms bin, event traces are 99.77% zeros. dF/F achieved higher decoder accuracy on every output variable.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are stacked into a (n_neurons, T) matrix using `np.vstack`. No additional processing (normalization, z-scoring, filtering) is applied. The data is cast to float32.

ii.
```python
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
assert activity.shape[1] == len(ophys_ts)
```

iii. The Allen SDK pipeline already applies motion correction, segmentation, ROI filtering, demixing, neuropil subtraction, and dF/F computation. Per-neuron z-scoring was tested and rejected (worse on 4 of 5 outputs).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All neurons present in the SDK's `dff_traces` are included.

ii. N/A (no filtering code)

iii. The AI verified that `valid_roi` is all True in the released data, meaning the Allen pipeline's ROI filtering has already been applied upstream. Neither reference paper describes further cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start (`start_time`). The ophys frames from `start_time` to `stop_time` are extracted using `np.searchsorted` with `side='left'`, giving a half-open interval `[start_time, stop_time)`.

ii.
```python
def frame_slice(ophys_ts, t_start, t_stop):
    i0 = int(np.searchsorted(ophys_ts, t_start, side='left'))
    i1 = int(np.searchsorted(ophys_ts, t_stop, side='left'))
    return i0, i1
# ...
neural.append(activity[:, i0:i1].copy())
```

iii. The AI uses `searchsorted(..., 'left')` for both boundaries, giving frames with `start_time <= t < stop_time`. This is the same alignment used by the reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native ophys frame rate (~31 Hz, ~32.3 ms per frame). No temporal rebinning is applied. The time bin size is computed as the mean of median inter-frame intervals across all sessions.

ii.
```python
dt_all = np.array([s['dt'] for s in sessions])
'time_bin_size': float(np.mean(dt_all) * 1000.0),   # ms
```

iii. The ophys timestamps are at a consistent frame rate determined by the microscope scanning. No resampling is needed since all data streams are aligned to the same ophys timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically the `image_name`, `start_time`, `end_time`, `omitted`, and `is_change` columns from the change-detection block. Grey screen periods (inter-stimulus intervals and omitted flashes) are coded as a separate `grey` category.

ii.
```python
def build_stimulus_timeseries(stim, ophys_ts):
    block = stim[stim.stimulus_block_name.str.contains('change_detection', na=False)]
    shown = block[~block.omitted.astype(bool)]
    image_names = sorted(set(shown.image_name.unique()))
    code = {name: i + 1 for i, name in enumerate(image_names)}
    # ...
    for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
        img_local[a:b] = c
```

iii. The AI used the flash-level stimulus_presentations table rather than the trials table to get precise per-frame image identity. This captures the actual visual stimulus at each timepoint, including grey screen periods between flashes and during omitted presentations.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A whole-session per-frame timeseries is built: default is 0 (grey), and during each non-omitted flash the image code is set. A global vocabulary of `grey` + all unique image names is built across all sessions. Session-local codes are mapped to global codes via a lookup table.

ii.
```python
image_names_global = sorted({n for s in sessions for n in s['image_names']})
image_values = [GREY_LABEL] + image_names_global
global_code = {n: i + 1 for i, n in enumerate(image_names_global)}
# ...
lut = np.zeros(len(s['image_names']) + 1, dtype=np.int64)
for i, name in enumerate(s['image_names']):
    lut[i + 1] = global_code[name]
out[0] = lut[s['image_local'][k]]
```

iii. The global mapping ensures consistent integer codes across sessions. Grey = 0, images = 1..N. This differs from the reference which uses only `initial_image_name`/`change_image_name` from the trials table and has no grey category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame within the trial window. The whole-session stimulus timeseries is built using `np.searchsorted` on flash start/end times against `ophys_ts`, then sliced per trial using the same frame indices as the neural data.

ii.
```python
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
# per trial:
img_tr.append(img_local[i0:i1].copy())
```

iii. Since the stimulus timeseries and neural data both use `ophys_ts` as the time base, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `ds.stimulus_presentations`. The change label is set to 1 during the 250 ms presentation of a changed image flash.

ii.
```python
changes = shown.is_change.values.astype(bool)
# ...
if ch:
    is_change[a:bc] = 1
```

iii. The AI uses the stimulus_presentations table's `is_change` flag rather than the trials table's `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A whole-session binary timeseries is built: 0 by default, 1 during the ophys frames of a flash where `is_change == True`. The window defaults to the flash duration (250 ms, from `start_time` to `end_time`).

ii.
```python
is_change = np.zeros(n, dtype=np.uint8)
# With CHANGE_WINDOW == 'flash':
chg_ends = ends
i1c = np.searchsorted(ophys_ts, chg_ends, side='left')
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    if ch:
        is_change[a:bc] = 1
```

iii. The AI tested both a 250 ms flash window and a 750 ms presentation-interval window, finding the 250 ms window produced better decoder accuracy and was more consistent with the image identity coding (which also uses flash boundaries).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary: 0 = no change, 1 = change. No thresholding is needed.

ii. The output is directly `uint8` values of 0 or 1.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - built per ophys frame using `searchsorted` on the flash table, then sliced with the same trial frame indices.

ii.
```python
chg_tr.append(is_change_ts[i0:i1].copy())
```

iii. Both use the same ophys frame index array, guaranteeing alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, which provides timestamps and speed values from the running wheel encoder (60 Hz, already 10 Hz low-pass filtered by the SDK).

ii.
```python
def resample_running(ds, ophys_ts):
    rs = ds.running_speed
    t = rs.timestamps.values.astype(np.float64)
    v = rs.speed.values.astype(np.float64)
    good = np.isfinite(v)
    if not good.all():
        t, v = t[good], v[good]
    return np.interp(ophys_ts, t, v).astype(np.float32)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase using `np.interp`, then discretized into 5 percentile-based bins. Bin edges are computed globally across all sessions using `np.quantile`. Non-finite values in the source data are excluded before interpolation.

ii.
```python
return np.interp(ophys_ts, t, v).astype(np.float32)
# ...
def quantile_edges(values, nbins=N_QUANTILE_BINS):
    qs = np.arange(1, nbins) / nbins
    return np.quantile(values, qs)
def digitize(x, edges):
    return np.digitize(x, edges, right=False).astype(np.int64)
```

iii. `np.interp` is used instead of `scipy.interp1d`. This extrapolates with edge values rather than NaN, which means no NaN handling is needed after interpolation. Percentile-based binning ensures roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five percentile bins are computed globally. The interior edges (20th, 40th, 60th, 80th percentiles) are computed using `np.quantile`. Values are then digitized using `np.digitize` with `right=False`, yielding bins 0-4.

ii.
```python
def quantile_edges(values, nbins=N_QUANTILE_BINS):
    qs = np.arange(1, nbins) / nbins
    return np.quantile(values, qs)
def digitize(x, edges):
    return np.digitize(x, edges, right=False).astype(np.int64)
```

iii. Global edges ensure consistent categories across sessions. The distribution is exactly 20% per bin dataset-wide.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase before trial segmentation, so it shares the same time indices as the neural data. The same `i0:i1` slice is used for both.

ii.
```python
running = resample_running(ds, ophys_ts)
# ...
run_tr.append(running[i0:i1].copy())
```

iii. By interpolating onto `ophys_ts` upfront, alignment with neural data is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, specifically `pupil_area`. The area is converted to diameter via `2 * sqrt(area / pi)`. Blink frames (where `pupil_area` is NaN, corresponding to `likely_blink`) are excluded before interpolation.

ii.
```python
def resample_pupil(ds, ophys_ts):
    eye = ds.eye_tracking
    t = eye.timestamps.values.astype(np.float64)
    area = eye.pupil_area.values.astype(np.float64)
    good = np.isfinite(area) & (area > 0)
    if good.sum() < 2:
        return None
    diam = 2.0 * np.sqrt(area[good] / np.pi)
    return np.interp(ophys_ts, t[good], diam).astype(np.float32)
```

iii. The AI used `pupil_area` and converted to diameter, while the reference uses `pupil_width` directly. Both should be closely related measures. Blink frames are NaN in the SDK and are excluded before interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter (`2*sqrt(area/pi)`), blink frames (NaN/non-positive area) are removed, then linearly interpolated onto the ophys timebase using `np.interp`. Then discretized into 5 percentile bins globally.

ii.
```python
good = np.isfinite(area) & (area > 0)
diam = 2.0 * np.sqrt(area[good] / np.pi)
return np.interp(ophys_ts, t[good], diam).astype(np.float32)
```

iii. `np.interp` extrapolates with edge values rather than producing NaN. The area-to-diameter transform is monotone, so percentile bins would be identical whether computed on area or diameter.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile bins computed globally using `np.quantile` with interior edges.

ii.
```python
pup_edges = quantile_edges(all_pup)
out[3] = digitize(s['pupil'][k], pup_edges)
```

iii. Global edges ensure consistent categories across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed - pupil diameter is interpolated to the ophys timebase before trial segmentation, sharing the same time indices as neural data.

ii.
```python
pupil = resample_pupil(ds, ophys_ts)
# ...
pup_tr.append(pupil[i0:i1].copy())
```

iii. Same alignment mechanism as neural and running data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table. The AI asserts that exactly one of these is True for every selected trial.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_flags = sel[OUTCOME_NAMES].values.astype(bool)
assert (outcome_flags.sum(axis=1) == 1).all()
outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)
```

iii. These four columns are the SDK's canonical trial outcome labels. The assertion verifies mutual exclusivity.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via `np.argmax` over the boolean outcome columns. The integer code is broadcast as a constant across all time bins within a trial.

ii.
```python
outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)
# per trial:
out[4] = s['outcome'][k]  # static, broadcast over T
```

iii. The mapping order follows `OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']`, matching the reference.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled: (1) Sessions that fail to load are caught by `extract_session_safe` and reported as errors, (2) Sessions without eye tracking return `None` for pupil and are skipped, (3) Trials with fewer than 2 ophys frames are dropped, (4) Sessions with fewer than 2 kept trials are dropped, (5) `np.interp` handles edge extrapolation (no NaN produced), (6) Non-finite running speed values are excluded before interpolation.

ii.
```python
def extract_session_safe(oeid):
    try:
        return extract_session(oeid)
    except Exception as exc:
        return {'oeid': oeid, 'skip': f'ERROR {exc!r}', ...}
# ...
if pupil is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
# ...
if i1 - i0 < 2:
    continue
```

iii. The try/except ensures a single bad session doesn't crash the pipeline. Using `np.interp` instead of `scipy.interp1d` avoids NaN propagation from extrapolation.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `get_behavior_ophys_experiment()`, which reads large NWB files from disk. This is I/O bound and takes about 2.2 s per session serially.

ii. N/A

iii. The AI measured and documented timing: Phase 1 (read+extract) takes 58.7 s with 16 workers for 168 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the flash-to-frame mapping using a single `np.searchsorted` over all flashes per session, rather than iterating per-trial. The per-trial loop for slicing data is simple array slicing that is not a bottleneck. A small loop `for a, b, bc, c, ch in zip(...)` iterates over ~4800 flashes per session.

ii.
```python
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
```

iii. The per-flash loop could theoretically be vectorized further, but it operates on ~4800 items per session and is not a bottleneck compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is loaded once, and the extracted trial data is reused for global binning and final assembly without reloading. However, the `show_processing` mode re-opens the NWB file for verification plots (this is intentional for independent checking, not repeated in normal mode).

ii. N/A

iii. The `show_processing` re-load is by design (independent verification), not accidental repetition.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects extensive per-session metadata (`session_info` list with cell_specimen_ids, equipment_name, cre_line, etc.) that is stored in `metadata['session_info']` but not used by the decoder. The `show_processing` mode re-loads data for independent verification plots. The code also computes and prints detailed summary statistics.

ii.
```python
session_info.append({
    'ophys_experiment_id': int(s['oeid']),
    'ophys_session_id': s['ophys_session_id'],
    'behavior_session_id': s['behavior_session_id'],
    'mouse_id': s['mouse_id'],
    'cre_line': s['cre_line'],
    # ... many more fields
    'cell_specimen_ids': s['cell_specimen_ids'],
})
```

iii. The extra metadata is useful for documentation and debugging but adds to file size. The `cell_specimen_ids` arrays in particular add memory overhead per session.
