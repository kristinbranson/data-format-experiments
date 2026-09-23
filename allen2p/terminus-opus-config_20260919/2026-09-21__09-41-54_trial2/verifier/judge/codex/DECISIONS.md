# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads AllenSDK metadata from a local cache at `/app/data`, scans the local NWB directory to find which experiments are physically present, intersects those experiment IDs with the experiment table, removes passive experiments up front, groups rows into sessions, and then loads each experiment in a session with `get_behavior_ophys_experiment()`.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import \
        VisualBehaviorOphysProjectCache
    return VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
...
cache = get_cache()
exp_table = cache.get_ophys_experiment_table()
...
local_ids = sorted(int(f.split('_')[-1].split('.')[0])
                   for f in os.listdir(nwb_dir) if f.endswith('.nwb'))
et = exp_table.loc[exp_table.index.intersection(local_ids)]
...
et = et[~et.passive]
...
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
```

iii. In `CONVERSION_NOTES.md`, the AI says it used `from_local_cache` because the provided data are an offline local cache, restricted itself to locally present NWBs, and excluded passive sessions because the target task was the behaving Visual Behavior task.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. After session conversion, the AI builds `subjects` as the sorted set of session-level `mouse_id` values and constructs `subject_idx` from that list.

ii.
```python
result = {
    ...
    'mouse_id': str(meta_rows.iloc[0]['mouse_id']),
    ...
}
...
subjects = sorted({r['mouse_id'] for r in good})
subject_idx = np.array([subjects.index(r['mouse_id']) for r in good], dtype=np.int64)
```

iii. The notes say `mouse_id` is the canonical subject identifier in the Allen metadata, so it is used directly as the mouse split.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `ophys_session_id`. All experiment-table rows with the same `ophys_session_id` are grouped into one job and treated as one converted session.

ii.
```python
session_ids = sorted(et.ophys_session_id.unique())
...
for sid in session_ids:
    rows = et[et.ophys_session_id == sid].sort_index()
    jobs.append((int(sid), list(rows.index), rows,
                 image_names, args.show_processing, '/app', args.neural_signal))
```

iii. The notes justify this as the natural session unit because experiments from the same `ophys_session_id` share behavior, stimulus timing, and clock.

## 1-d. How are the data split into trials?

i. Trials are taken from `d0.trials`, but only `go` and `catch` rows are kept. Each kept row becomes one fixed-length trial window centered on `change_time`, not the full `start_time` to `stop_time` trial.

ii.
```python
trials = d0.trials
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
...
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
```

iii. In the notes, the AI says it defines a trial as one go or catch trial aligned to the image change, using a `[-2, +4] s` window because that change event is what the decoder should focus on.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials where `go` or `catch` is true and `change_time` is present. It skips sessions with fewer than 2 such trials, skips sessions with no eye tracking, skips sessions with all-NaN pupil diameter, and skips sessions with zero neurons.

ii.
```python
eye = d0.eye_tracking
if eye is None or len(eye) == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}
...
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
if len(tr) < 2:
    return {'session_id': ophys_session_id, 'skipped': 'fewer than 2 go/catch trials'}
...
if pupil_diam is None:
    return {'session_id': ophys_session_id, 'skipped': 'pupil all NaN'}
...
if n_neurons == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no neurons'}
```

iii. The notes say go/catch matches the task specification, that passive sessions were removed earlier, and that eye-tracking-less sessions were dropped because pupil diameter is a required decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default the neural data come from `dff_traces.dff`. The script also exposes optional alternatives: `events.filtered_events` and `events.events`.

ii.
```python
for d, eid in zip(datasets, exp_ids):
    if neural_signal == 'dff':
        ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
    elif neural_signal == 'filtered_events':
        ev = np.vstack(d.events.filtered_events.values).astype(np.float32)
    else:
        ev = np.vstack(d.events.events.values).astype(np.float32)
```

iii. The notes explain that the paper used events, but the AI chose dF/F as the default because it empirically decoded better in its sample experiments.

## 2-b. How is the `neural` data processed?

i. Neural traces from all planes in a session are binned into 100 ms bins within each trial window by averaging samples inside each bin, and then vertically stacked across planes.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    ...
    parts = []
    for ev, ts in zip(plane_events, plane_ts):
        b, counts = bin_average(ev, ts, edges)
        ...
        parts.append(b)
    neural_trials.append(np.vstack(parts).astype(np.float32))
```

iii. The notes justify this as using one common time bin size across rigs while avoiding upsampling the slower 11 Hz sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional per-neuron quality filtering beyond the Allen release. The only neural-related skip is dropping sessions with zero neurons.

ii.
```python
regions_per_neuron = np.concatenate(region_idx_parts) if region_idx_parts else np.array([])
n_neurons = sum(e.shape[0] for e in plane_events)
if n_neurons == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no neurons'}
```

iii. The notes say the released Allen data already include upstream ROI validity filtering and standard pipeline processing, so no extra neuron QC was added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to `trials.change_time` and uses a fixed `[-2.0, +4.0] s` window around that event. Catch trials use the SDK’s sham change time because that is what appears in `change_time`.

ii.
```python
BIN_SIZE = 0.1
OFF_START = -2.0
OFF_END = 4.0
...
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. The notes say the image change is the defining event for the decoder outputs, so all signals were centered on that event rather than on trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 100 ms bins (`NBINS = 60` over a 6 s window). Yes, temporal rebinning is applied through bin averaging.

ii.
```python
BIN_SIZE = 0.1
OFF_START = -2.0
OFF_END = 4.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
...
b, counts = bin_average(ev, ts, edges)
```

iii. The notes justify 100 ms as a common grid that works for both 11 Hz and 31 Hz sessions without inventing higher-frequency samples for slower recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the session’s `stimulus_presentations` table, specifically `image_name`, after restricting that table to the change-detection stimulus block.

ii.
```python
sp = d0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
sp = sp.sort_values('start_time')
stim_start = sp.start_time.values.astype(np.float64)
stim_img = sp.image_name.values.astype(str)
```

iii. The notes say the AI chose `stimulus_presentations` because it directly describes the flashed images and omissions over time.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI constructs a fixed global class list of the 16 natural images plus `omitted`, maps each flashed image name to an integer code, and then assigns each 100 ms bin the image whose flash interval contains that bin center.

ii.
```python
image_names = sorted({
    'im000', 'im031', 'im035', 'im045', 'im054', 'im073', 'im075', 'im106',
    'im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im077', 'im085'})
image_names = image_names + ['omitted']
...
img_to_code = {name: i for i, name in enumerate(image_names)}
stim_code = np.array([img_to_code[n] for n in stim_img], dtype=np.int64)
...
j = np.searchsorted(stim_start, centers, side='right') - 1
j = np.clip(j, 0, len(stim_start) - 1)
within = centers < stim_end[j]
img = np.where(within, stim_code[j], img_to_code['omitted'])
```

iii. The notes justify this with the paper’s “image presentation interval” framing and say omissions should be their own class because no image is present on omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned by using the same 100 ms trial bins and bin centers used for the neural data. Each bin gets the image code for the flash interval containing that bin center.

ii.
```python
edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
...
j = np.searchsorted(stim_start, centers, side='right') - 1
...
img_trials.append(img.astype(np.int64))
```

iii. The notes say all outputs were put on the same ophys-based time grid as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations.is_change` flag in the filtered change-detection stimulus table.

ii.
```python
stim_change = sp.is_change.values.astype(bool)
...
chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. The notes justify this as a direct readout of whether the currently assigned flash interval is the real change flash.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI assigns each 100 ms bin a binary value indicating whether its center falls inside a flash interval whose `is_change` flag is true.

ii.
```python
j = np.searchsorted(stim_start, centers, side='right') - 1
j = np.clip(j, 0, len(stim_start) - 1)
within = centers < stim_end[j]
chg = np.where(within, stim_change[j], False).astype(np.int64)
change_trials.append(chg)
```

iii. The notes say this follows the flash-interval convention used elsewhere in the script and avoids hand-coded change windows.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary. The AI casts the boolean change indicator to integer `0/1` and labels the categories `no_change` and `change`.

ii.
```python
chg = np.where(within, stim_change[j], False).astype(np.int64)
...
'output_values': [
    image_names,
    ['no_change', 'change'],
    [f'speed_quintile_{i+1}' for i in range(NQUANT)],
    [f'pupil_quintile_{i+1}' for i in range(NQUANT)],
    OUTCOMES,
],
```

iii. The notes treat image change as an inherently binary output rather than something requiring further thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same 100 ms bins and the same `change_time`-centered window as the neural data.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ...
    chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. The notes explicitly say all outputs were aligned to the change event on the ophys-based bin grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `d0.running_speed`, using the `speed` values and their native `timestamps`.

ii.
```python
rs = d0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_sp = interpolate_nans(rs.speed.values)
```

iii. The notes identify Allen’s running-speed stream as the canonical locomotion measure.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI fills any NaNs in the full running trace, averages running speed within each 100 ms trial bin, and if a bin has no samples it fills that bin by interpolating the underlying trace at the bin center.

ii.
```python
def interpolate_nans(x):
    ...
    if bad.any():
        good = ~bad
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), x[good])
    return x
...
rb, rc = bin_average(run_sp[None, :], run_ts, edges)
rb = rb[0]
if (rc == 0).any():
    rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
    filled_run += int((rc == 0).sum())
run_trials.append(rb)
```

iii. The notes say this avoids leaving empty bins at zero and keeps all behavioral outputs on the same 100 ms grid as neural data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 within-session equal-percentile bins by computing quantile thresholds over all running values from that session’s extracted trials.

ii.
```python
def quantile_bins(values, nq=NQUANT):
    qs = np.quantile(values, np.arange(1, nq) / nq)
    labels = np.searchsorted(qs, values, side='right')
    return labels.astype(np.int64), qs
...
run_mat = np.vstack(run_trials)
run_lab, run_q = quantile_bins(run_mat.ravel())
run_lab = run_lab.reshape(run_mat.shape)
```

iii. The notes justify per-session quintiles by arguing that absolute running scales differ across sessions and mice, so relative state within session is more comparable.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned by binning running speed with the same `edges` array used for the neural data in each change-centered trial.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ...
    rb, rc = bin_average(run_sp[None, :], run_ts, edges)
```

iii. The notes say all modalities are synchronized to the same ophys-based trial grid, so running is binned directly on that grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area`, converted to diameter as `2 * sqrt(area / pi)`.

ii.
```python
eye_ts = eye.timestamps.values.astype(np.float64)
pupil_area = eye.pupil_area.values.astype(np.float64)
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_nans(pupil_diam_raw)
```

iii. The notes say the AI preferred a geometric diameter computed from pupil area rather than using one width dimension directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI linearly interpolates NaNs in the session-level diameter trace, averages the interpolated diameter within each 100 ms trial bin, and fills any sample-empty bins by interpolating at the bin center.

ii.
```python
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_nans(pupil_diam_raw)
...
pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
pb = pb[0]
if (pc == 0).any():
    pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
    filled_pupil += int((pc == 0).sum())
pupil_trials.append(pb)
```

iii. The notes justify this as standard handling for blink-related gaps and dropped camera samples.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 within-session equal-percentile bins using quantiles from that session’s extracted pupil values.

ii.
```python
pupil_mat = np.vstack(pupil_trials)
pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
pupil_lab = pupil_lab.reshape(pupil_mat.shape)
```

iii. The notes argue that absolute pupil size is not directly comparable across rigs and sessions, so session-relative quintiles are more stable.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. It is aligned with the same change-centered 100 ms bins used for the neural data.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ...
    pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
```

iii. The notes say pupil was moved onto the same per-trial time grid as the other outputs and the neural signal.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` columns of the filtered `trials` table.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = np.full(len(tr), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    outcome[tr[name].values.astype(bool)] = k
```

iii. The notes say those four Allen outcome flags are the canonical trial-outcome labels for the task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans to integer codes `0..3` in `OUTCOMES` order and broadcasts each trial’s code across all 60 time bins.

ii.
```python
outcome = np.full(len(tr), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    outcome[tr[name].values.astype(bool)] = k
...
out = np.stack([img_trials[i],
                change_trials[i],
                run_lab[i],
                pupil_lab[i],
                np.full(NBINS, outcome[i], dtype=np.int64)], axis=0)
```

iii. The notes say the output should be time-varying when possible, so the static trial label is repeated across time instead of being stored once.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid data are handled mainly by skipping entire sessions or interpolating within sessions. Missing eye tracking or all-NaN pupil traces cause a session skip; NaNs inside behavioral traces are linearly interpolated; bins with no running or pupil samples are filled by interpolation at bin centers; exceptions return an error record instead of crashing the whole run.

ii.
```python
def interpolate_nans(x):
    ...
    if bad.all():
        return None
    if bad.any():
        good = ~bad
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), x[good])
...
if eye is None or len(eye) == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}
...
if pupil_diam is None:
    return {'session_id': ophys_session_id, 'skipped': 'pupil all NaN'}
...
if (rc == 0).any():
    rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
...
if (pc == 0).any():
    pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
...
except Exception as exc:
    import traceback
    return {'session_id': int(ophys_session_id), 'error': f'{exc}',
            'traceback': traceback.format_exc()}
```

iii. The notes describe these choices as keeping required outputs usable while preventing one bad session from aborting the full conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identified NWB loading as the main bottleneck. Session conversion is parallelized with a multiprocessing pool; the notes say binning is relatively cheap compared with file I/O.

ii.
```python
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
...
if args.workers > 1 and len(jobs) > 1:
    with Pool(min(args.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
            ...
```

iii. `CONVERSION_NOTES.md` says NWB load time dominates and reports per-plane/session timing, while vectorized binning keeps compute costs low.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the inner bin averaging with `searchsorted` and cumulative sums, but it still loops over trials and planes in Python. The optional plotting code also uses explicit per-bin loops.

ii.
```python
def bin_average(values, timestamps, edges):
    ...
    idx = np.searchsorted(sub_ts, edges, side='left')
    counts = np.diff(idx)
    csum = np.concatenate([np.zeros((sub.shape[0], 1), dtype=np.float64),
                           np.cumsum(sub.astype(np.float64), axis=1)], axis=1)
    sums = csum[:, idx[1:]] - csum[:, idx[:-1]]
...
for ct in change_times:
    ...
    for ev, ts in zip(plane_events, plane_ts):
        b, counts = bin_average(ev, ts, edges)
...
for b in range(NBINS):
    sel = (run_ts >= edges[b]) & (run_ts < edges[b + 1])
```

iii. The notes explicitly present vectorized binning as an optimization; they do not claim every remaining loop was removed, only that the dominant numerical part was.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats cache construction inside every session worker, repeats trial-by-trial bin construction for every output stream, and, when plotting is enabled, recomputes binned running and pupil traces again just for visualization.

ii.
```python
def process_session(args):
    ...
    cache = get_cache()
...
for ct in change_times:
    ...
    rb, rc = bin_average(run_sp[None, :], run_ts, edges)
    ...
    pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
...
for b in range(NBINS):
    sel = (run_ts >= edges[b]) & (run_ts < edges[b + 1])
...
for b in range(NBINS):
    sel = (eye_ts >= edges[b]) & (eye_ts < edges[b + 1])
```

iii. The notes emphasize avoiding repeated NWB reads, but the code still repeats some setup and diagnostic work across workers and plots.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code optionally creates detailed processing plots and always tracks timing/QC counters such as `empty_bins`, `filled_run_bins`, `filled_pupil_bins`, and `total_time`, but those diagnostics are not part of the final decoder inputs/outputs. It also stores some per-session fields in intermediate `result` dicts and then keeps only a subset in the final pickle.

ii.
```python
result = {
    ...
    'empty_bins': int(empty_bins),
    'filled_run_bins': int(filled_run),
    'filled_pupil_bins': int(filled_pupil),
    'timing': timing,
    'total_time': time.time() - t_start,
}
...
if show_processing:
    try:
        make_processing_plot(...)
...
'session_info': [
    {k: r[k] for k in ('session_id', 'experiment_ids', 'mouse_id',
                       'session_type', 'cre_line', 'experience_level',
                       'project_code', 'equipment_name', 'n_neurons',
                       'n_trials', 'n_go', 'n_catch', 'n_hit', 'n_miss',
                       'n_fa', 'n_cr', 'run_quantiles', 'pupil_quantiles',
                       'pupil_nan_frac', 'cell_specimen_ids')}
    for r in good],
```

iii. The notes frame these as validation/debugging aids rather than part of the downstream representation.
