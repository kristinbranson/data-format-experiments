# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data exclusively through `allensdk.brain_observatory.behavior.behavior_project_cache.VisualBehaviorOphysProjectCache`, but it uses `from_local_cache`, restricts to NWB files that are already present locally, filters to `project_code == 'VisualBehavior'` and `~passive`, and then loads each selected experiment with `get_behavior_ophys_experiment`. It processes experiments in parallel with `multiprocessing.Pool`.

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

ds = get_cache().get_behavior_ophys_experiment(oeid)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as the only fully present local dataset variant, and says it used `from_local_cache` because the provided cache is on disk and does not need direct NWB access or S3 fetching. It also states NWB decompression/loading is the runtime bottleneck, so it parallelized session extraction.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the selected experiment table, then stored as sorted strings.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_idx = np.array([subjects.index(s['mouse_id']) for s in sessions], dtype=np.int64)
```

iii. The AI’s notes explicitly say subject identity comes from Allen metadata and that `mouse_id` is the animal identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each `BehaviorOphysExperiment` / NWB file / `ophys_experiment_id` as one session. It does not group multiple experiments by `ophys_session_id`.

ii.
```python
def extract_session(oeid):
    ds = get_cache().get_behavior_ophys_experiment(oeid)
    ...
    info = {
        'oeid': oeid,
        ...
        'ophys_session_id': int(md['ophys_session_id']),
        ...
    }
```

iii. The justification in the notes is that the local `VisualBehavior` subset is “single-plane,” so “session ≡ experiment ≡ NWB file.” The AI therefore decided not to reconstruct sessions by grouping experiments that share an `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are taken from `ds.trials`, restricted to `go` or `catch`, and each trial spans the experiment-defined interval `[start_time, stop_time)`. Trial frame indices are computed on the ophys timestamps with `np.searchsorted`.

ii.
```python
trials = ds.trials
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]

for k, (_, tr) in enumerate(sel.iterrows()):
    i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
    if i1 - i0 < 2:
        continue
```

iii. The AI says in its notes that it used the experiment’s own trial definition because the instructions said to segment recordings into trials “based on how they are defined in the experiment.”

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` and `catch` trials, relies on the SDK’s trial logic to exclude aborted and auto-rewarded trials, drops trials with fewer than 2 ophys frames, and later drops sessions with fewer than 2 kept trials.

ii.
```python
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
assert not sel.aborted.any(), f'{oeid}: aborted trial selected'
assert not sel.auto_rewarded.any(), f'{oeid}: auto-rewarded trial selected'

if i1 - i0 < 2:
    continue

sessions = [r for r in results if not r['skip'] and r['n_trials_kept'] >= 2]
```

iii. The notes cite AllenSDK `Trial._get_trial_data()` and argue that `go`/`catch` are already mutually exclusive with aborted and auto-rewarded trials, so `trials[go | catch]` is exactly the requested subset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default, the converted neural data come from `ds.dff_traces['dff']`. The script also exposes debug options for `events` and `filtered_events`, but the shipped conversion uses `dff`.

ii.
```python
ev = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
```

iii. The AI’s notes justify `dff` as the Allen pipeline’s published neural output and say it outperformed `events` and `filtered_events` in its decoder spot checks.

## 2-b. How is the `neural` data processed?

i. The AI stacks all cell traces within one experiment into a `(neurons, time)` matrix, casts to `float32`, and slices it into per-trial matrices. It does not merge multiple experiments from the same `ophys_session_id`.

ii.
```python
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
...
neural.append(activity[:, i0:i1].copy())
...
sess_neural.append(np.ascontiguousarray(act, dtype=np.float32))
```

iii. The notes say no extra normalization or filtering is applied because the Allen pipeline already computed the traces. The main additional processing is trial segmentation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied beyond what is already present in the AllenSDK-released data. Sessions with zero cells are skipped.

ii.
```python
if len(ev) == 0:
    return {'oeid': oeid, 'skip': 'no cells'}
```

iii. The AI’s notes say Allen ROI filtering was already applied upstream and that `cell_specimen_table.valid_roi` was all true in the released data it checked.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start by selecting ophys frames with `start_time <= t < stop_time` on `ds.ophys_timestamps`.

ii.
```python
def frame_slice(ophys_ts, t_start, t_stop):
    i0 = int(np.searchsorted(ophys_ts, t_start, side='left'))
    i1 = int(np.searchsorted(ophys_ts, t_stop, side='left'))
    return i0, i1

neural.append(activity[:, i0:i1].copy())
```

iii. The AI’s notes say the master time base is `ophys_timestamps`, and trial segmentation is done directly on that common clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native single-plane ophys frame rate, about 31 Hz / 32.3 ms per bin. It does not temporally rebin the neural data.

ii.
```python
'dt': float(np.median(np.diff(ophys_ts))),
...
'time_bin_size': float(np.mean(dt_all) * 1000.0),
'sampling_rate_hz': float(1.0 / np.mean(dt_all)),
```

iii. The notes justify this by saying the instructions asked for alignment on ophys timestamps, so the natural conversion bin is one ophys frame.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically the change-detection stimulus block, using each flash’s `image_name`, `start_time`, `end_time`, and `omitted` status.

ii.
```python
block = stim[stim.stimulus_block_name.str.contains('change_detection', na=False)]
shown = block[~block.omitted.astype(bool)]

image_names = sorted(set(shown.image_name.unique()))
starts = shown.start_time.values
ends = shown.end_time.values
codes = shown.image_name.map(code).values.astype(np.int16)
```

iii. The AI’s notes explicitly say it preferred `stimulus_presentations` so that image identity would reflect the actual flash table, including grey intervals and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a full-session per-frame image-identity time series on the ophys time base. Frames during flashed images get a nonzero code; all other frames, including inter-stimulus grey periods and omissions, remain `0` / `grey`. Those session-local codes are later mapped to a global image vocabulary.

ii.
```python
img_local = np.zeros(n, dtype=np.int16)
...
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c

image_names_global = sorted({n for s in sessions for n in s['image_names']})
image_values = [GREY_LABEL] + image_names_global
global_code = {n: i + 1 for i, n in enumerate(image_names_global)}
...
out[0] = lut[s['image_local'][k]]
```

iii. The justification in the notes is that the task asked for “image identity of the image presented during the non-grey screen,” so the AI treated the grey intervals as a separate state and explicitly counted omissions as grey.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-identity vector is first built on the same `ophys_timestamps` time base as the neural data, and then each trial gets the slice defined by the same `[start_time, stop_time)` frame indices used for neural extraction.

ii.
```python
img_local, is_change_ts, image_names = build_stimulus_timeseries(
    ds.stimulus_presentations, ophys_ts)
...
img_tr.append(img_local[i0:i1].copy())
...
out[0] = lut[s['image_local'][k]]
```

iii. The notes justify this as a common-clock alignment strategy: build all time-varying signals on `ophys_timestamps`, then slice neural and outputs with the same trial frame boundaries.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, using `is_change` together with the flash `start_time`/`end_time` boundaries within the change-detection block.

ii.
```python
changes = shown.is_change.values.astype(bool)
...
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
    if ch:
        is_change[a:bc] = 1
```

iii. The AI’s notes say this uses the flash table directly so that “image_change” marks the actual changed-image presentation rather than an arbitrary longer post-change interval.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI builds a binary per-frame session vector. With the default `CHANGE_WINDOW='flash'`, it sets `image_change = 1` only during the 250 ms changed-image flash. It also includes a debug option for a longer `presentation_interval` window but does not use it by default.

ii.
```python
CHANGE_WINDOW = 'flash'
...
if CHANGE_WINDOW == 'presentation_interval':
    ...
else:
    chg_ends = ends
...
if ch:
    is_change[a:bc] = 1
```

iii. The notes say the 250 ms flash window decoded better than the 750 ms alternative and was “more consistent with how image identity is coded.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous threshold is applied; the AI emits a binary categorical variable with codes `0` = `no_change` and `1` = `change`.

ii.
```python
is_change = np.zeros(n, dtype=np.uint8)
...
'output_values': [
    image_values,
    ['no_change', 'change'],
    [f'run_q{i + 1}' for i in range(N_QUANTILE_BINS)],
    [f'pupil_q{i + 1}' for i in range(N_QUANTILE_BINS)],
    OUTCOME_NAMES,
],
```

iii. The code and notes both treat image change as a binary event, so no extra thresholding step is described.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is built on `ophys_timestamps` and then sliced per trial using the same frame indices as the neural data.

ii.
```python
img_local, is_change_ts, image_names = build_stimulus_timeseries(
    ds.stimulus_presentations, ophys_ts)
...
chg_tr.append(is_change_ts[i0:i1].copy())
...
out[1] = s['is_change'][k]
```

iii. The AI’s notes justify this with the same common-clock alignment argument it used for the other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
rs = ds.running_speed
t = rs.timestamps.values.astype(np.float64)
v = rs.speed.values.astype(np.float64)
```

iii. The notes identify `ds.running_speed` as the AllenSDK locomotion interface and say the values are already the SDK’s processed running-speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto the ophys timestamps with `np.interp`, then computes global quintile edges across all kept sessions and digitizes each trial’s values into bin indices.

ii.
```python
def resample_running(ds, ophys_ts):
    rs = ds.running_speed
    t = rs.timestamps.values.astype(np.float64)
    v = rs.speed.values.astype(np.float64)
    return np.interp(ophys_ts, t, v).astype(np.float32)

all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
run_edges = quantile_edges(all_run)
...
out[2] = digitize(s['running'][k], run_edges)
```

iii. The notes justify global quintiles as making the bin labels comparable across sessions while producing the requested five equal-percentile bins dataset-wide.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global equal-percentile bins using dataset-wide quantile edges.

ii.
```python
def quantile_edges(values, nbins=N_QUANTILE_BINS):
    qs = np.arange(1, nbins) / nbins
    return np.quantile(values, qs)

def digitize(x, edges):
    return np.digitize(x, edges, right=False).astype(np.int64)
```

iii. The AI’s notes explicitly say the categories are global quintiles over all converted timepoints.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is resampled to `ophys_timestamps` before trialization and then sliced with the same `[start_time, stop_time)` indices as the neural data.

ii.
```python
running = resample_running(ds, ophys_ts)
...
run_tr.append(running[i0:i1].copy())
...
out[2] = digitize(s['running'][k], run_edges)
```

iii. The notes say all streams are on the SDK’s common clock, so interpolation onto the ophys time base is the alignment step.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, specifically `pupil_area` and `timestamps`, with `likely_blink` handled indirectly because blink frames appear as invalid / NaN `pupil_area`.

ii.
```python
eye = ds.eye_tracking
t = eye.timestamps.values.astype(np.float64)
area = eye.pupil_area.values.astype(np.float64)
good = np.isfinite(area) & (area > 0)
```

iii. The notes justify this by saying the task asks for pupil diameter, so it converts pupil area to diameter rather than using `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts valid pupil area samples to diameter with `2*sqrt(area/pi)`, linearly interpolates through blink gaps and onto the ophys time base with `np.interp`, drops sessions with unusable or empty eye tracking, then bins the aligned values into global quintiles.

ii.
```python
if eye is None or len(eye) == 0:
    return None
...
diam = 2.0 * np.sqrt(area[good] / np.pi)
return np.interp(ophys_ts, t[good], diam).astype(np.float32)
...
if pupil is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
```

iii. The notes justify interpolation across blink gaps as a way to keep every neural frame aligned to a pupil value while preserving trial lengths.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global equal-percentile bins using dataset-wide quantile edges and `np.digitize`.

ii.
```python
all_pup = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
pup_edges = quantile_edges(all_pup)
...
out[3] = digitize(s['pupil'][k], pup_edges)
```

iii. The notes say the same rationale as running speed applies: one consistent five-bin scale over the whole converted dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is first resampled onto `ophys_timestamps`, then trial slices are extracted using the same frame indices as the neural data.

ii.
```python
pupil = resample_pupil(ds, ophys_ts)
...
pup_tr.append(pupil[i0:i1].copy())
...
out[3] = digitize(s['pupil'][k], pup_edges)
```

iii. The AI’s notes again appeal to the common AllenSDK clock and the policy of slicing all streams with the same trial boundaries after ophys-time resampling.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `ds.trials`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome_flags = sel[OUTCOME_NAMES].values.astype(bool)
assert (outcome_flags.sum(axis=1) == 1).all()
outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)
```

iii. The notes say these are the canonical Allen outcome flags for go/catch trials and verified exactly one is true for each selected trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four mutually exclusive boolean flags are converted to integer class indices with `argmax`, and each trial’s scalar outcome code is broadcast across all time bins of that trial in the final `(5, T)` output array.

ii.
```python
outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)
...
out = np.empty((5, T), dtype=np.int64)
...
out[4] = s['outcome'][k]
```

iii. The notes justify broadcasting because the output format requires one array per trial and the trial outcome is static within a trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips sessions with no cells or no usable eye tracking, catches session-level exceptions with a safe wrapper, drops trials with fewer than 2 ophys frames, and fills blink-related pupil gaps by interpolation rather than preserving missing values.

ii.
```python
if len(ev) == 0:
    return {'oeid': oeid, 'skip': 'no cells'}
...
if eye is None or len(eye) == 0:
    return None
...
if good.sum() < 2:
    return None
...
if pupil is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
...
if i1 - i0 < 2:
    continue

def extract_session_safe(oeid):
    try:
        return extract_session(oeid)
    except Exception as exc:
        ...
        return {'oeid': oeid, 'skip': f'ERROR {exc!r}', 'traceback': traceback.format_exc()}
```

iii. The notes say pupil diameter is a required output, so sessions without eye tracking cannot be kept. They also say interpolation through blink gaps preserves alignment and fixed trial lengths relative to neural data.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies loading/decompressing experiments via `get_behavior_ophys_experiment` as the bottleneck and therefore parallelizes session extraction.

ii.
```python
ds = get_cache().get_behavior_ophys_experiment(oeid)
...
with Pool(min(args.workers, len(oeids))) as pool:
    for i, info in enumerate(pool.imap(extract_session_safe, oeids, chunksize=1)):
        ...
```

iii. In the notes, it explicitly says “NWB decompression is the bottleneck” and reports a wall-clock speedup from multiprocessing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s main efficiency decision was to avoid a trial-by-trial flash lookup by building session-wide stimulus time series once with vectorized `searchsorted` on all flash boundaries. It still retains explicit Python loops over flashes and over trials.

ii.
```python
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
i1c = np.searchsorted(ophys_ts, chg_ends, side='left')
...
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
    if ch:
        is_change[a:bc] = 1

for k, (_, tr) in enumerate(sel.iterrows()):
    i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
```

iii. The notes say this avoided an `O(trials × flashes)` design and that the remaining loops were acceptable relative to the I/O bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. In the main conversion path, the AI’s position is that it avoids repeated processing by building each session’s stimulus/running/pupil time series once and then slicing trials from those arrays. The optional `show_processing()` path does reopen a session and re-render diagnostic plots.

ii.
```python
img_local, is_change_ts, image_names = build_stimulus_timeseries(
    ds.stimulus_presentations, ophys_ts)
running = resample_running(ds, ophys_ts)
pupil = resample_pupil(ds, ophys_ts)
...
if args.show_processing:
    for s in sessions[:2]:
        show_processing(s, run_edges, pup_edges, image_values,
                        f'processing_{s["oeid"]}.png')
```

iii. The notes explicitly state that repeated re-reading of streams was avoided in the main pipeline; the only intentional repetition is optional validation/plotting work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs optional diagnostic plotting and unconditional summary/assertion checks that are not consumed by downstream decoder analyses. It also stores rich session metadata that are useful for inspection but not required by the decoder.

ii.
```python
run_summary(data)

if args.show_processing:
    for s in sessions[:2]:
        show_processing(s, run_edges, pup_edges, image_values,
                        f'processing_{s["oeid"]}.png')
...
'metadata': {
    ...
    'session_info': session_info,
},
```

iii. The AI’s notes frame these as deliberate sanity checks and documentation rather than part of the minimal decoding dataset.
