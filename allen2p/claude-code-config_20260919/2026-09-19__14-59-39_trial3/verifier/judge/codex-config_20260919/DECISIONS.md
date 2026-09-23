# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local ophys experiment metadata, finds locally present NWB files, keeps `project_code == 'VisualBehavior'` and active (`passive == False`) experiments, then loads each selected file directly as a `BehaviorOphysExperiment`. It processes experiments in parallel. Thus it does not load every metadata-listed VisualBehavior session: passive sessions and later sessions lacking usable eye data are excluded.

ii.
```python
et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
sel = et[et.ophys_experiment_id.isin(present)
         & (et.project_code == PROJECT_CODE)
         & (~et.passive.astype(bool))].copy()
...
ds = BehaviorOphysExperiment.from_nwb_path(path)
...
with ProcessPoolExecutor(nworkers) as ex:
    for n, res in enumerate(ex.map(process_experiment, jobs), start=1):
```

iii. The notes say direct NWB loading produces the same SDK object as the cache route and was needed because the local cache lacks its manifest directory. They restrict to single-plane VisualBehavior for a uniform 30.94 Hz clock and exclude passive sessions because the paper did not analyze them and their outcomes are degenerate.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs from kept experiments; a sorted global subject list and per-session index are constructed.

ii.
```python
subjects = sorted({r['subject'] for r in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject']] for r in kept], dtype=np.int64)
```

iii. The agent treats SDK metadata `mouse_id` as the animal identifier and retains it in session metadata for traceability.

## 1-c. How are the data split into sessions?

i. One selected `VisualBehavior` ophys experiment/NWB is one output session. The agent does not group experiments by `ophys_session_id` because this project is single-plane and has one experiment per session.

ii.
```python
for i, row in sel.iterrows():
    jobs.append({'ophys_experiment_id': int(row.ophys_experiment_id), ...})
...
return {'oeid': int(oeid), 'neural': neural, ...}
```

iii. The notes explicitly verified that experiment and session are equivalent for the selected single-plane VisualBehavior project; multiscope data, where this would not hold, are excluded.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. Only go or catch rows are kept, and each trial is the ophys frames in `[start_time, stop_time)`.

ii.
```python
trials = ds.trials
go = trials['go'].values.astype(bool)
catch = trials['catch'].values.astype(bool)
tr = trials[go | catch]
starts = tr['start_time'].values.astype(np.float64)
stops = tr['stop_time'].values.astype(np.float64)
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
```

iii. The agent cites the Allen tutorial, which uses the four go/catch outcomes and trial start/stop bounds. Variable natural trial lengths preserve both pre-change and response periods.

## 1-e. How are trials filtered based on quality controls?

i. The go/catch selection excludes aborted and auto-rewarded trials. Trials shorter than two ophys frames or containing no genuine non-blink pupil sample are dropped, and sessions left with fewer than two trials are skipped.

ii.
```python
keep = go | catch
...
if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
    continue
if j1[k] - j0[k] < 1:
    continue
...
if len(sel) < MIN_TRIALS_PER_SESSION:
    return {'oeid': int(oeid), 'skip': f'only {len(sel)} usable trials'}
```

iii. Go/catch is justified as exactly the union of hit, miss, false alarm, and correct reject. The extra pupil rule avoids creating an entire trial from extrapolation, and the two-trial rule is required by the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Default neural data come from `ds.dff_traces['dff']`, indexed by valid cell specimen IDs. Optional benchmarking flags can instead use raw or filtered events.

ii.
```python
cell_ids = cst.index.values[valid_roi]
signal = job.get('neural_signal', 'dff')
if signal == 'dff':
    col = ds.dff_traces['dff']
else:
    col = ds.events[signal]
```

iii. The agent benchmarked dF/F, events, and filtered events and chose un-z-scored dF/F because it decoded every requested timepoint-level output better and is a released, already processed neural product.

## 2-b. How is the `neural` data processed?

i. Selected traces are copied into a float32 neuron-by-full-session matrix, with no normalization by default, then contiguous per-trial time slices are made.

ii.
```python
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes state the SDK has already demixed, neuropil-corrected, baseline-normalized, and detrended dF/F. Z-scoring was tested but not retained.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells whose `cell_specimen_table.valid_roi` is true are retained; sessions with no such cells are skipped. No further cell-quality threshold is applied.

ii.
```python
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
if len(cell_ids) == 0:
    return {'oeid': int(oeid), 'skip': 'no valid ROIs'}
```

iii. The agent says the released dataset already passed the Allen ROI classifier and verified all 29,097 candidate cells in selected active sessions were valid, so this guard removes nothing in practice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start by searching the common ophys timestamps for trial start and stop, then slicing the neural matrix with those exact indices.

ii.
```python
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
...
a, b = i0[k], i1[k]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The agent says all timestamps share the hardware-synchronized session clock; no estimated shift is introduced. Metadata records trial start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data stay at native single-plane ophys resolution, about 32.32 ms (30.94 Hz). No neural temporal rebinning is applied; other streams are resampled onto these timestamps.

ii.
```python
'frame_period_s': float(np.median(np.diff(ts))),
...
'time_bin_size': float(np.median(frame_periods) * 1000.0)
```

iii. The agent selected the single-plane project specifically to keep one native time-bin size across sessions and asserts that frame periods agree within tolerance.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from change-detection rows of `ds.stimulus_presentations`, chiefly `start_time`, `end_time`, `image_name`, and implicitly omitted presentations through their absent/non-image spans.

ii.
```python
sp = stimulus_presentations
sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
start = sp['start_time'].values.astype(np.float64)
end = sp['end_time'].values.astype(np.float64)
names = sp['image_name'].astype(str).values
```

iii. The agent chose presentation records rather than trial-level initial/change names so identity follows actual 250 ms on-screen flashes and grey/omitted periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Sixteen globally distinct image names map to codes 1–16. Code 0 represents grey, including inter-stimulus intervals and omitted flashes. For each ophys time, the most recent presentation is found and accepted only if the frame precedes its end time.

ii.
```python
codes = np.array([IMAGE_CODE.get(n, 0) for n in names], dtype=np.int16)
idx = np.searchsorted(start, ts, side='right') - 1
on_screen = valid & (ts < np.nan_to_num(end[idx_c], nan=-np.inf))
image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
```

iii. The agent argues the requested identity is the image during the non-grey screen, so grey must be explicitly distinguishable; global names avoid conflating different A/B stimuli that share within-session indices.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. A full-session image trace is evaluated directly at every ophys timestamp, then sliced with the same trial indices as neural data.

ii.
```python
image_all, change_all, sp = stimulus_traces(ds.stimulus_presentations, ts, ...)
...
out[0] = image_all[a:b]
```

iii. The agent visually and independently checked that non-grey codes occur exactly inside presentation spans on the common ophys clock.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses `stimulus_presentations.is_change` together with presentation start/end times. Sham catch presentations are not `is_change`.

ii.
```python
is_change = sp['is_change'].values.astype(bool)
...
change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. The agent regards an actual identity-changing presentation as the change event and intentionally leaves catch/sham changes at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The default produces 1 only on ophys frames within the approximately 250 ms changed flash. A non-default `interval` option can extend it across the associated 750 ms presentation interval.

ii.
```python
if change_window == 'interval':
    change = (valid & is_change[idx_c]).astype(np.int16)
else:
    change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. The notes report benchmarking both definitions and retaining the flash-only label because it is consistent with the agent's on-screen image-identity definition.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: false/no changed flash is 0 and true/current changed flash is 1. There is no numeric threshold.

ii.
```python
change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. The two classes are documented as `['no_change', 'change']`; catch trials remain no-change because image identity does not change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is computed at all ophys timestamps and sliced by the same `[a:b]` indices used for the neural trial.

ii.
```python
image_all, change_all, sp = stimulus_traces(..., ts, ...)
...
out[1] = change_all[a:b]
```

iii. The common ophys-time evaluation makes the label rise on the same frame as the changed stimulus presentation.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `ds.running_speed['timestamps']` and `ds.running_speed['speed']`.

ii.
```python
rt = running_speed['timestamps'].values.astype(np.float64)
rv = running_speed['speed'].values.astype(np.float64)
```

iii. The notes describe this as the SDK's already unwrapped, transient-corrected, low-pass-filtered wheel speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are removed, timestamps sorted, speed linearly interpolated once onto the full ophys timebase, and retained trial samples are quantile-binned per session.

ii.
```python
good = np.isfinite(rt) & np.isfinite(rv)
rt, rv = rt[good], rv[good]
order = np.argsort(rt)
return np.interp(ts, rt[order], rv[order])
...
rb, run_edges = quantile_bins(run_all[frame_idx])
```

iii. The agent avoids repeated per-trial interpolation and uses per-session bins to make categories balanced and comparable within a recording.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the session's 20th, 40th, 60th, and 80th percentiles over exactly the kept trial frames; `searchsorted(..., side='right')` assigns classes 0–4.

ii.
```python
edges = np.quantile(values, np.arange(1, nbins) / nbins)
return np.searchsorted(edges, values, side='right').astype(np.int16), edges
```

iii. Per-session quintiles were chosen to satisfy five equal percentile bins locally and avoid letting between-session calibration differences dominate the categories.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated onto `ophys_timestamps` before segmentation and later sliced with the neural trial bounds.

ii.
```python
run_all = resample_running(ds.running_speed, ts)
...
out[2] = run_bin_all[a:b]
```

iii. The agent relies on the shared hardware clock and performs no additional time shift.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `ds.eye_tracking['timestamps']` and `pupil_area`; finite positive area samples are treated as non-blink samples.

ii.
```python
t = eye_tracking['timestamps'].values.astype(np.float64)
a = eye_tracking['pupil_area'].values.astype(np.float64)
good = np.isfinite(t) & np.isfinite(a) & (a > 0)
```

iii. The notes say pupil fields are NaN at likely blinks and choose area because an effective diameter can be derived from it.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Invalid/blink samples are dropped; area is converted to effective circular diameter `2*sqrt(area/pi)`, linearly interpolated onto the ophys clock, and quantile-binned per session.

ii.
```python
diam = 2.0 * np.sqrt(a / np.pi)
return np.interp(ts, t, diam), t
...
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
```

iii. The agent calls interpolation standard blink handling and notes that diameter is monotonic in area, so the quintile membership is unchanged by the transformation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. As for running, each session uses its kept trial values' 20/40/60/80 percentiles to produce classes 0–4.

ii.
```python
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
pup_bin_all[frame_idx] = pb
```

iii. The agent argues absolute camera-pixel pupil measures vary by rig, zoom, and eye position, making per-session percentile bins more meaningful and exactly balanced.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Blink-cleaned diameter is interpolated at every ophys timestamp, binned, and sliced with the same `[a:b]` trial frame range.

ii.
```python
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
...
out[3] = pup_bin_all[a:b]
```

iii. Alignment uses the common synchronized session clock. Trials with no actual pupil sample inside their bounds are rejected despite interpolation being available.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` after go/catch filtering.

ii.
```python
outcome_cols = np.stack([tr['hit'].values.astype(bool),
                         tr['miss'].values.astype(bool),
                         tr['false_alarm'].values.astype(bool),
                         tr['correct_reject'].values.astype(bool)], axis=1)
```

iii. The agent verified every selected go/catch trial has exactly one of these canonical outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. An assertion enforces one-hot outcomes, `argmax` maps them in fixed order to 0–3, and the static trial label is repeated across all trial frames.

ii.
```python
assert outcome_cols.sum(axis=1).min() == 1 and outcome_cols.sum(axis=1).max() == 1
outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)
...
out[4] = outcome[k]
```

iii. Repetition permits all five outputs to share a `(5, T)` representation while preserving a static per-trial value.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with absent eye tracking or no valid ROIs are skipped. Non-finite running samples and invalid pupil areas are removed before interpolation; trials without any real pupil observation or with fewer than two frames are dropped. Plot failures are warnings only. Assertions catch non-unique outcomes and inconsistent frame periods.

ii.
```python
if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
    return {'oeid': int(oeid), 'skip': 'no eye tracking data'}
...
if j1[k] - j0[k] < 1:
    n_drop_pupil += 1
    continue
...
except Exception as exc:
    print(f'  [warn] plotting failed for {oeid}: {exc!r}', flush=True)
```

iii. The agent prefers explicit exclusion where a mandatory output cannot be grounded in observed data. It records all skipped sessions and trial-drop counts and performed independent raw-NWB spot checks.

## 9-a. What are the most time-consuming steps of the code?

i. Opening roughly 0.9 GB NWB files and materializing the neuron-by-frame trace matrix dominate. Pickle writing and optional plots also cost time, but session loading/neural extraction is the central bottleneck.

ii.
```python
ds = BehaviorOphysExperiment.from_nwb_path(path)
...
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

iii. The notes report about 3–6 seconds per session for opening/materialization and reduce wall time with 16 worker processes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-cell trace-copy loop, trial-quality selection loop, trial assembly loop, and Python image-code construction could potentially be vectorized or replaced with bulk indexing. Trial boundary searches already are vectorized, and sessions are parallelized.

ii.
```python
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
...
for k in range(len(tr)):
    ...
for k in sel:
    ...
```

iii. The agent specifically identified naive per-trial interpolation/searching as avoidable and instead resamples once per session and uses vectorized `searchsorted`; it judged I/O more important than the small remaining loops.

## 9-c. What processing does the code repeat multiple times?

i. Default conversion resamples each behavior/stimulus stream only once per session, but it necessarily repeats NWB opening, trace extraction, interpolation, binning, and assembly independently for every session. Optional plotting rereads/derives some display arrays, and the agent's separate benchmark runs repeat whole conversion under alternate neural/change settings.

ii.
```python
for n, res in enumerate(ex.map(process_experiment, jobs), start=1):
    ...
run_all = resample_running(ds.running_speed, ts)
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
```

iii. The notes emphasize that work is not repeated per trial: full-session resampling cuts approximately 250 interpolation calls per session to one. Benchmark repetition was intentional validation, not part of the default full run.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default run, full-session behavior arrays, stimulus-table subsets, raw quantile edges, timing diagnostics, and rich session metadata are computed although the decoder consumes only trial slices and core format fields. The optional plotting path performs substantial visualization-only processing. Most of this is retained in metadata or logs rather than silently discarded; full-session intermediates are released after trial construction.

ii.
```python
image_all, change_all, sp = stimulus_traces(...)
...
info = {... 'running_quintile_edges': run_edges.tolist(), ...}
...
if show:
    make_processing_plot(...)
```

iii. The agent justifies these extras as auditability, sanity checking, and diagnostics. Plotting is disabled unless requested, while traceability metadata was deliberately retained.
