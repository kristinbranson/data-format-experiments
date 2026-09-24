# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local ophys experiment metadata CSV, excludes session types containing `passive`, and considers every table row whose experiment NWB file exists. It loads each retained file directly with `BehaviorOphysExperiment.from_nwb_path`. A first pass gathers global labels/bin edges and a second pass reloads each experiment for conversion.

ii.
```python
def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

exp_table = load_experiment_table()
exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
```

iii. The notes identify the release as a local Allen project-cache-style dataset and say direct NWB loading avoids a broken local-cache manifest. Passive data are excluded because trial outcome is not behaviorally meaningful there.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `mouse_id`; unique string IDs are sorted, and each converted experiment is assigned its index in that list.

ii.
```python
subject = str(meta_row.get('mouse_id', 'unknown'))
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes describe `mouse_id` as the subject identifier and preserve subject mapping across experiments.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane) is treated as a separate converted session. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges,
                              image_to_idx, blank_idx)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The notes explicitly justify this choice on the grounds that an experiment has a coherent neuron set and a single region/depth assignment, despite acknowledging that multiple experiments are nested in one ophys session.

## 1-d. How are the data split into trials?

i. The SDK `trials` table supplies behavioral trial boundaries. For each kept row, the agent selects ophys timestamps in the half-open interval `[start_time, stop_time)` and skips windows with fewer than two frames.

ii.
```python
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
    if m.sum() < 2:
        continue
```

iii. The notes say behaviorally defined SDK trials should be segmented while retaining time-varying outputs on the ophys grid.

## 1-e. How are trials filtered based on quality controls?

i. Only rows marked `go` or `catch` are retained; aborted and auto-rewarded rows are explicitly removed. Experiments with fewer than two such raw rows are rejected in the collection pass, trials with fewer than two ophys frames are skipped, and converted experiments with fewer than two final trials are omitted.

ii.
```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
...
if keep.sum() < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The notes trace this rule to the instructions and the SDK definitions of go, catch, aborted, and auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity preferentially comes from `exp.dff_traces`; `exp.events` is a fallback only if the dF/F table is empty. Trace extraction prefers columns named `events`, `filtered_events`, then `dff`.

ii.
```python
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()
    return 'events', events
```

iii. The notes say sparse events caused an all-zero trial warning and poorer decoding, so the final implementation switched to SDK-provided dF/F.

## 2-b. How is the `neural` data processed?

i. Per-cell arrays are converted to `float32`, truncated to the shortest cell trace, stacked as neuron by time, then truncated again to match the ophys timestamp count. Trial windows are sliced without further filtering or normalization.

ii.
```python
arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
n_t = min(len(a) for a in arrays)
return np.stack([a[:n_t] for a in arrays], axis=0)
...
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The agent relied on the SDK pipeline's already processed dF/F and reported a raw-versus-converted `np.allclose` check.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell/ROI quality filtering is performed; every row present in the SDK dF/F table is retained.

ii.
```python
neural = extract_neural_matrix(signal_df, signal_kind)
brain_region_idx = np.zeros((neural.shape[0],), dtype=np.int64)
```

iii. The notes considered SDK-provided traces and cell segmentation sufficient and did not add a separate ROI filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are selected on the native ophys clock between behavioral trial start and stop. Thus each variable uses the same `sample_times`, with trial start as the effective alignment event and variable trial length.

ii.
```python
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The notes repeatedly state that all streams are aligned to ophys timestamps, as required by the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment remains at its native ophys sampling interval, and metadata reports the median experiment-level interval across converted experiments in milliseconds.

ii.
```python
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
...
time_bin_size = float(np.median([s['dt_ms'] for s in sessions]))
```

iii. The agent's plan calls for native ophys timestamps and describes assignment of other streams onto that grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`, with `start_time`, a reconstructed `stop_time`, and `omitted` used to define applicable intervals.

ii.
```python
stim = ensure_stim_stop_time(stim_df)
stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
image_vals = [image_to_idx.get(str(x), blank_idx)
              for x in stim_non_omitted['image_name'].astype(str).values]
```

iii. The notes selected presentation-level variables because they directly describe the time-varying stimulus.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. All non-omitted image names encountered in the collection pass are globally sorted and mapped to integers after prepending a `blank` class. Each presentation is extended to the next presentation start, and interval membership assigns its code; gaps and omitted presentations remain `blank`.

ii.
```python
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
...
image_identity = interval_assign(sample_times,
    stim_non_omitted['start_time'].values, stim_non_omitted['stop_time'].values,
    image_vals, default=blank_idx)
```

iii. Sample validation initially found invalid negative/unknown labels. The agent added an explicit blank class and extended presentation labels over full presentation intervals to reduce unknown bins.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. For every neural `sample_time`, interval tests find the non-omitted stimulus presentation containing that timestamp; the resulting vector has exactly the neural trial length.

ii.
```python
out = np.full(sample_times.shape, default, dtype=np.int64)
for s, e, v in zip(starts, stops, values):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v
```

iii. The notes say assigning presentation data directly on the trial's ophys grid guarantees alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from the `is_change` boolean and `start_time` columns of non-omitted stimulus presentations overlapping the trial.

ii.
```python
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[
            stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
```

iii. The mapping plan chose `stimulus_presentations.is_change` as the presentation-level change marker.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The output starts as zero. For every marked presentation, `searchsorted` finds the first trial ophys sample at or after presentation start, and exactly that one sample is set to one.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. The agent describes this as a binary time series that is one “immediately after” a change and otherwise zero.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated. `False`/absence maps to category 0 (`no_change`) and a marked first sample maps to category 1 (`change`).

ii.
```python
'output_values': [
    ...
    ['no_change', 'change'],
]
```

iii. The raw SDK boolean marker was treated as already categorical.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Presentation start is located in the same `sample_times` array used to slice neural data, using the first ophys sample at or after the event.

ii.
```python
idx = np.searchsorted(sample_times, ct, side='left')
image_change[idx] = 1
```

iii. The notes require every time-varying output to be assigned on the ophys timestamp grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `exp.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
run_df = exp.running_speed.copy()
return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The notes identify the SDK running-speed stream as the wheel-derived locomotion measure.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Global percentile edges are computed from every finite raw running sample in every eligible experiment, not just retained trial samples. For each trial, the nearest running sample is assigned to each ophys timestamp and digitized. Non-finite assigned values become bin 0.

ii.
```python
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
run_edges = make_percentile_bins(np.concatenate(run_vals))
...
run_vals = nearest_assign(sample_times, run_df['timestamps'].values,
                          run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The notes proposed interpolation/assignment to the ophys grid and global equal-percentile categories for consistent, balanced decoder labels.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global 20th/40th/60th/80th percentile edges define five integer bins through `np.digitize`; duplicate edges are nudged upward and missing values are assigned category 0.

ii.
```python
qs = np.linspace(0, 100, n_bins + 1)[1:-1]
edges = np.percentile(vals, qs)
...
out = np.digitize(vals, edges, right=False).astype(np.int64)
out[~np.isfinite(vals)] = 0
```

iii. The five percentile bins directly implement the decoder specification and were chosen globally to keep labels consistent across experiments.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Each neural ophys timestamp is assigned the temporally nearest native running-speed observation, including endpoint extrapolation by clipping to the first/last source sample.

ii.
```python
idx = np.searchsorted(source_times, sample_times, side='left')
idx = np.clip(idx, 0, len(source_times) - 1)
...
return source_values[idx]
```

iii. The agent viewed timestamp-based assignment to the ophys clock as sufficient hardware-synchronized alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It comes from `exp.eye_tracking`. The first available column among `pupil_diameter`, `pupil_area`, `pupil_width`, and `pupil_radius` is used, with a numeric pupil-named fallback. The `likely_blink` flag is not used.

ii.
```python
candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
for c in candidates:
    if c in eye_tracking.columns:
        return c
```

iii. The notes state that a pupil-diameter field would be identified from the experiment object and invalid values masked, but do not justify substituting area/radius or retaining blink frames.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Percentile edges are computed globally from all finite raw values in the chosen pupil column. Within trials, rows with non-finite timestamps/values are dropped, nearest-neighbor assignment maps values to ophys times, and values are digitized. Missing columns or wholly invalid tables yield all-zero bins.

ii.
```python
pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals))
...
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values,
                            eye_valid[pupil_col].values)
```

iii. The agent intended global equal-percentile discretization and direct assignment on the ophys timebase; it also chose zero as a robust missing-data fallback.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global percentile cut points define five categories exactly as for running speed; non-finite values and missing pupil streams map to category 0.

ii.
```python
pupil_edges = make_percentile_bins(...)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. This follows the requested five equal-percentile bins, with global cut points for consistent category meanings.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The nearest valid eye-tracking sample is selected for each neural ophys timestamp, with out-of-range times clipped to an endpoint sample.

ii.
```python
pupil_vals = nearest_assign(sample_times,
    eye_valid['timestamps'].values, eye_valid[pupil_col].values)
```

iii. The notes regard all source clocks as synchronized and use the ophys timestamps as the common target grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived in priority order from the trial row's `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
if bool(trial_row.get('hit', False)):
    outcome = 0
elif bool(trial_row.get('miss', False)):
    outcome = 1
elif bool(trial_row.get('false_alarm', False)):
    outcome = 2
elif bool(trial_row.get('correct_reject', False)):
    outcome = 3
else:
    outcome = 0
```

iii. The notes identify these as the canonical, mutually exclusive SDK outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes map to integers 0–3 in the documented order, and the selected value is repeated across all time bins of the trial. An unrecognized row silently defaults to hit (0).

ii.
```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
...
['hit', 'miss', 'false_alarm', 'correct_reject']
```

iii. Replication makes a static per-trial target compatible with the time-series output matrix; the notes report spot-checking an outcome against the raw trial table.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing NWB files are skipped; non-finite values are excluded when fitting bins; missing or empty pupil data become bin 0; non-finite samples digitize to 0; trace/timestamp length mismatches are truncated; absent stimulus labels fall back to `blank`; too-short trials and experiments are omitted. There is no per-experiment exception handler, so other malformed data abort the run, and nearest-neighbor alignment extrapolates endpoints.

ii.
```python
if not nwb_path.exists():
    continue
...
n_t = min(neural.shape[1], len(ophys_timestamps))
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
...
if m.sum() < 2:
    continue
```

iii. The notes document fixes for unknown images and sparse event traces and say sample/full verification produced a valid file; most fallback behavior is defensive code rather than explicitly justified scientific imputation.

## 9-a. What are the most time-consuming steps of the code?

i. NWB/AllenSDK loading is the main cost, amplified because every eligible experiment is loaded in `collect_global_info` and then loaded again in `process_experiment`. Trial/presentation interval assignment also adds Python-loop work.

ii.
```python
for _, row in exp_table.iterrows():
    exp = load_experiment(exp_id)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, ...)
```

iii. The notes explicitly identify repeated experiment loading as slow and say the full conversion runtime was long.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. `interval_assign` loops over every stimulus interval and builds a full Boolean mask each time; image-change events are also looped. The per-trial `iterrows` loop and first-pass experiment/table loops could be reorganized or vectorized, though trial slicing naturally retains some iteration.

ii.
```python
for s, e, v in zip(...):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v
...
for ct in ...:
    idx = np.searchsorted(sample_times, ct, side='left')
```

iii. The notes specifically call out interval assignment as a vectorization opportunity and acknowledge only partial vectorization.

## 9-c. What processing does the code repeat multiple times?

i. Every selected experiment is decoded from NWB twice. Trial filtering, stimulus stop-time reconstruction, running-table extraction, eye-table extraction, and pupil-column selection are repeated in both passes. `ensure_stim_stop_time` is additionally repeated for every trial inside `build_trial_output` even though `process_experiment` already normalized the experiment's stimulus table.

ii.
```python
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
...
trial_output = build_trial_output(..., trial_stim, ...)
...
def build_trial_output(...):
    stim = ensure_stim_stop_time(stim_df)
```

iii. The agent documented repeated experiment loading as an inefficiency but did not remove it.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `signal_kind` is passed to `extract_neural_matrix` but never used there. `brain_region_idx` is first created as zeros inside each session and later used only for its length before being replaced by a constant region mapping. Stimulus stop times are reconstructed on already trial-clipped tables, and full-session raw behavior samples outside retained trial windows are processed to fit bin edges even though those samples are not outputs.

ii.
```python
def extract_neural_matrix(signal_df, signal_kind):
    preferred = ['events', 'filtered_events', 'dff']
...
'brain_region_idx': brain_region_idx,
...
np.full((len(s['brain_region_idx']),), region_to_idx[s['region']], dtype=np.int64)
```

iii. The notes do not justify these discarded intermediates; they arise from generic/fallback structure and the two-pass design.
