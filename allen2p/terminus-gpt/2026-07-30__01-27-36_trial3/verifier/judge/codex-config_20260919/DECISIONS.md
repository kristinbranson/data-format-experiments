# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-sorts every locally present NWB file (284 in the full run) and loads each with `NWBHDF5IO`, then constructs one `BehaviorOphysExperiment` per file. It does not use the project cache/experiment table and treats the local subset as the complete convertible dataset.

ii.
```python
files = sorted(NWB_DIR.glob('*.nwb'))
...
io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
nwbfile = io.read()
exp = BehaviorOphysExperiment.from_nwb(nwbfile)
```

iii. The notes say project metadata describe a larger release than the available NWBs, so conversion should operate only on locally available files. The trajectory also emphasizes avoiding unavailable remote data.

## 1-b. How are the data split into subjects?

i. After processing files, unique string-valued `mouse_id` metadata are sorted into `subjects`; each converted file receives the corresponding index.

ii.
```python
mouse_id = str(meta.get('mouse_id', 'unknown'))
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes identify `mouse_id` as the source for subject assignment and report 21 local subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file/ophys experiment is appended as a separate target session. Experiments sharing an `ophys_session_id` are not grouped, so simultaneously acquired imaging planes remain separate sessions.

ii.
```python
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, ...)
    sessions.append(sess)
```

iii. The notes call the NWBs “one file per available ophys experiment” but subsequently describe these as sessions. No justification for failing to reconstruct multi-plane ophys sessions is given.

## 1-d. How are the data split into trials?

i. The SDK `exp.trials` table defines trials. For every retained row, ophys samples satisfying `start_time <= timestamp < stop_time` are selected, producing variable-length trial windows.

ii.
```python
trials = valid_trials_table(exp.trials)
...
idx = np.flatnonzero((ts >= start) & (ts < stop))
if idx.size < 2:
    continue
```

iii. The notes say experiment-defined trial intervals reconcile the requested trial segmentation with the paper’s image-wise analyses, while preserving the ophys clock.

## 1-e. How are trials filtered based on quality controls?

i. Rows marked by any recognized aborted or auto-rewarded column are removed. When possible, only rows labeled go or catch are kept. Trials with fewer than two ophys frames are also skipped. Sessions themselves are not subsequently checked for at least two retained trials.

ii.
```python
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
...
df = df[df['go'].fillna(False) | df['catch'].fillna(False)]
```

iii. The agent explicitly cites the instruction to include Go/Catch and exclude Aborted/Auto-rewarded trials. The two-frame rule prevents empty or unusable slices.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come exclusively from the SDK `dff_traces` table’s `dff` column; detected events are not used.

ii.
```python
dff = exp.dff_traces.copy()
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
```

iii. The notes initially preferred events because the paper used detected calcium events, but record that sample validation was switched to dF/F. The final code offers no events-first fallback despite metadata claiming one.

## 2-b. How is the `neural` data processed?

i. Per-cell dF/F vectors are cast to float32, stacked as cells by time, and sliced at the trial’s ophys indices. No normalization, smoothing, or multi-plane concatenation is performed.

ii.
```python
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The agent relied on SDK-provided processed dF/F and used the ophys clock as the common alignment base.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter; every row exposed in `dff_traces` is retained.

ii.
```python
cell_ids = np.asarray(dff.index)
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
```

iii. The notes say exact extra cell-quality rules were not established and implicitly rely on AllenSDK/NWB curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are selected directly on `exp.ophys_timestamps` between each trial’s start and stop; thus trial arrays begin at the first ophys frame at/after trial start rather than a fixed change-centered window.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_neural = neural_mat[:, idx]
```

iii. The notes identify ophys timestamps as the required master clock and experiment trial boundaries as the requested segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native ophys frames are retained with no rebinning. Metadata report the median interval from the first file in milliseconds.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. The choice preserves the common ophys time base; neither notes nor code propose additional temporal aggregation.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `exp.stimulus_presentations.image_name` and presentation `start_time`; every presentation is assumed to last 0.75 s.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim_end = np.asarray(stim['start_time'], dtype=float) + 0.75
```

iii. The paper/methods defined a 750 ms image-presentation interval, and the notes planned to use the presentation table rather than trial-level initial/change names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A mapping is built from names observed in only the first eight files, plus forced `blank` and `omitted` classes. Each trial starts as blank; frames in overlapping 750 ms presentation windows receive the mapped image name, with unseen names falling back to blank.

ii.
```python
probe_files = files[:min(8, len(files))]
image_names = set(['blank', 'omitted'])
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The notes say this produced balanced image classes and an omitted fraction near 3.7%; they justify blank as a full-time-series background class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus start/end times are compared with the exact `trial_ts = ophys_timestamps[idx]` used for neural slicing, assigning labels with a Boolean time mask.

ii.
```python
trial_ts = ts[idx]
smask = (trial_ts >= s0) & (trial_ts < s1)
image_series[smask] = ...
```

iii. The notes expressly choose ophys timestamps as the common clock for stimulus, behavior, and neural streams.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from consecutive `image_name` values among stimulus presentations overlapping each trial, rather than from trial `change_time` and `go`.

ii.
```python
prev_name = None
for _, srow in stim_trial.iterrows():
    name = str(srow['image_name'])
    if prev_name is not None and name != prev_name:
        ...
    prev_name = name
```

iii. The mapping plan describes image-identity transitions as the source and says the output should mark change events within trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The series begins at zero. Whenever the current presentation name differs from the previous presentation name, only the first ophys frame in the current 750 ms mask is set to one.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
```

iii. The agent interpreted “right after” as a one-frame impulse, although its notes do not justify using every within-trial identity transition rather than the experiment’s designated go change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is inherently binary: zero is `no_change`, and a detected transition frame is one (`change`); no numeric threshold is applied.

ii.
```python
'output_values': [..., ['no_change', 'change'], ...]
```

iii. This directly follows the requested binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The one is placed at the first neural/ophys frame whose timestamp lies in the new presentation interval.

ii.
```python
hit = np.flatnonzero(smask)
change_series[hit[0]] = 1
```

iii. The ophys timestamp mask is shared with image identity and neural samples.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the timestamp and speed columns of `exp.running_speed` (or positional fallback columns).

ii.
```python
tcol = 'timestamps' if 'timestamps' in cols else cols[0]
vcol = 'speed' if 'speed' in cols else cols[1]
```

iii. The notes identify the SDK running-speed stream as the canonical locomotion source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Raw running values from only the first eight full files determine global 0/20/40/60/80/100% quantiles. Trial values are linearly interpolated to ophys timestamps with `np.interp` and digitized against the four interior edges.

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False)
```

iii. The notes justify percentile bins as approximately balanced categories. The code comment calls the eight-file probe an efficiency optimization to avoid loading every NWB twice.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four interior quintile edges create integer categories 0–4 using left-closed `np.digitize` behavior (`right=False`).

ii.
```python
np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. Five equal-percentile bins are explicitly required; the agent approximates their global thresholds from the probe subset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated from its own timestamps directly at each trial’s ophys timestamps, matching one value to each neural column.

ii.
```python
trial_ts = ts[idx]
run_interp = np.interp(trial_ts, run_t, run_v)
```

iii. The notes specify interpolation/alignment to the ophys master clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The agent searches `exp.eye_tracking` for timestamps and the first available pupil-related numeric column, prioritizing `pupil_diameter`, then `pupil_area`, then width, height, radius, or size. It does not consult `likely_blink`.

ii.
```python
candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width',
                  'pupil_height', 'pupil_radius', 'pupil_size']
value_col = next((c for c in candidate_cols if c in cols), None)
```

iii. The notes planned to use a diameter-like or area-equivalent measure when necessary, but also recognized that missing values needed careful handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Quantiles are estimated from finite raw values in the first eight files. Within a trial, finite timestamp/value pairs are linearly interpolated to ophys times and digitized. No blink removal is done; absent or insufficient data become all-zero bins.

ii.
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid])
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1])
```

iii. The notes justify interpolation to the ophys clock and percentile discretization, but do not justify retaining blink-contaminated samples or sampling only eight files.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four interior quintile edges produce integer categories 0–4. Missing/insufficient pupil streams are assigned category zero throughout.

ii.
```python
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False)
...
pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. Five percentile bins follow the task; zero filling is a pragmatic fallback documented only generally as handling missing data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Valid eye-tracking samples are interpolated at the same trial ophys timestamps indexing neural columns.

ii.
```python
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid])
```

iii. The notes identify the ophys clock as the shared temporal alignment base.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the Boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`, in that priority order, with an `other` fallback.

ii.
```python
for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
    if key in row.index and bool(row[key]):
        return key
return 'other'
```

iii. The notes call these the experiment’s trial outcome fields and plan a static per-trial category.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The five labels, including `other`, are mapped to integers. Although conceptually static, the selected code is repeated across every time point in the trial output matrix.

ii.
```python
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
```

iii. Repeating the label makes it compatible with the shared time-varying output matrix; the notes describe it as static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Flexible column fallbacks handle schema variation; trials with fewer than two frames are skipped; missing/short pupil streams become bin zero; nonfinite pupil samples are removed before interpolation; unseen image names become blank; missing metadata become `unknown`. There is no per-session exception recovery, running NaN filtering before interpolation, or explicit file-handle close.

ii.
```python
if idx.size < 2: continue
...
if valid.sum() >= 2: ...
else: pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
image_to_idx.get(name, image_to_idx['blank'])
```

iii. The agent aimed for a robust local conversion, but the notes contain little specific validation of these fallbacks and leave critical-review steps unfinished.

## 9-a. What are the most time-consuming steps of the code?

i. Reading and decoding large NWB files/SDK experiment objects and materializing full dF/F matrices dominate. Trial-wise stimulus-table filtering and nested presentation iteration add CPU cost, while serializing the reported 14 GB pickle is also substantial.

ii.
```python
exp = load_experiment(p)
neural_mat, cell_ids, neural_source = get_cell_matrix(exp)
...
pickle.dump(data, f)
```

iii. The code’s explicit optimization comment says repeated NWB loading is expensive; the trajectory’s long full conversion and 14 GB output support that assessment.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over trials and over stimulus rows within each trial could be replaced by interval/searchsorted indexing. Image-name collection, output assembly, and per-session region arrays also permit vectorized or indexed constructions, though file loading itself remains serial and I/O-heavy.

ii.
```python
for _, tr in trials.iterrows():
    ...
    for _, srow in stim_trial.iterrows():
```

iii. The agent’s notes left the “code inefficiencies” section blank, so no explicit vectorization rationale was documented.

## 9-c. What processing does the code repeat multiple times?

i. The first eight NWBs are loaded once to estimate categories and again for conversion; the first file is loaded a third time to compute `dt_ms`. Stimulus rows are repeatedly filtered from the full table for every trial, and full trial outputs duplicate a static outcome value at every frame.

ii.
```python
collect_global_info(files)
...
for p in files: exp = load_experiment(p)
...
dt_ms = ... load_experiment(files[0]) ...
```

iii. The agent intentionally accepted the eight-file reload as a compromise to avoid reloading *all* files twice, but did not document the third load or repeated table scans.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `image_names_seen`, `trial_outcomes_seen`, `neural_source`, and `pupil_col` are computed/returned but do not affect the saved dataset (apart from `neural_source` being printed). Cell IDs are retained only to determine counts. `stim['_end_time']` and per-session name lists are intermediate conveniences. The full static outcome series also wastes storage compared with a scalar static output.

ii.
```python
image_names_seen.append(name)
trial_outcomes_seen.append(outcome)
...
'image_names_seen': sorted(set(image_names_seen)),
'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),
```

iii. No justification is supplied; the notes’ inefficiency and speedup fields were left blank.
