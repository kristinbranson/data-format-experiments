# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a local Allen release directly from disk, not through the project cache. It reads `project_metadata/ophys_experiment_table.csv`, excludes rows whose `session_type` contains `passive`, then iterates over `ophys_experiment_id` values. Each kept experiment is loaded from its NWB file with `BehaviorOphysExperiment.from_nwb_path()`. The code does this twice: once in `collect_global_info()` to build global image/bin metadata and choose eligible experiments, and again in `process_experiment()` to build the final dataset.

ii. 
```python
def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def get_nwb_path(exp_id: int) -> Path:
    return EXPT_DIR / f'behavior_ophys_experiment_{int(exp_id)}.nwb'

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))
...
exp_table = load_experiment_table()
exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
...
for _, row in exp_table.iterrows():
    exp_id = int(row['ophys_experiment_id'])
    nwb_path = get_nwb_path(exp_id)
    if not nwb_path.exists():
        continue
    exp = load_experiment(exp_id)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the agent justified moving to per-experiment NWB loading because it viewed each ophys experiment as one coherent recording unit and thought passive sessions were incompatible with the required trial-outcome output. The trajectory also shows an earlier Allen cache approach was replaced with direct NWB loading after schema/runtime issues.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `mouse_id` attached to each kept experiment metadata row. After processing, the code builds a sorted unique subject list and maps each converted session to that subject index.

ii. 
```python
subject = str(meta_row.get('mouse_id', 'unknown'))
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes repeatedly state that subject identity comes from experiment metadata and should be preserved across converted sessions.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as one converted session. It does not group multiple experiments by `ophys_session_id`.

ii. 
```python
for _, row in exp_table.iterrows():
    exp_id = int(row['ophys_experiment_id'])
    ...
    chosen_ids.append(exp_id)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. In Step 4 and Step 5 of `CONVERSION_NOTES.md`, the agent explicitly decided to use “ophys experiment as one converted session,” arguing that each experiment had one neuron set and one region assignment.

## 1-d. How are the data split into trials?

i. Trials come from `exp.trials`. For each retained row, the code uses the trial’s `start_time` and `stop_time`, selects all ophys timestamps in `[start_time, stop_time)`, and stores that variable-length slice as one trial.

ii. 
```python
trials = exp.trials.copy()
...
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
    if m.sum() < 2:
        continue
    sample_times = ophys_timestamps[m]
    trial_neural = neural[:, m].astype(np.float32)
```

iii. The notes say the agent wanted “behaviorally defined trials” and variable-length trial windows aligned to ophys timestamps.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are labeled `go` or `catch`, not `aborted`, and not `auto_rewarded`. Trials with fewer than 2 ophys frames are skipped. At the session level, passive sessions are excluded before loading and any experiment with fewer than 2 retained trials is excluded.

ii. 
```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()
...
if m.sum() < 2:
    continue
...
if keep.sum() < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The notes say this rule was chosen to match the instruction “include go and catch, exclude aborted and auto-rewarded,” plus an additional judgment that passive sessions should be dropped because trial outcome would not be meaningful there.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are taken from `exp.dff_traces` when available, with a fallback to `exp.events`. Within whichever table is chosen, the code prefers columns named `events`, `filtered_events`, or `dff`.

ii. 
```python
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()
    return 'events', events

def extract_neural_matrix(signal_df, signal_kind):
    preferred = ['events', 'filtered_events', 'dff']
    ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
```

iii. The trajectory shows the agent first tried events, then switched to dF/F after sample validation because event-based trials were sparse and produced warnings. `CONVERSION_NOTES.md` Step 10 and Step 12 explicitly say the final choice was dF/F because it verified and decoded better.

## 2-b. How is the `neural` data processed?

i. The chosen per-cell traces are converted to `float32`, truncated so all cells share the minimum trace length, stacked into a neuron-by-time matrix, then trimmed again so the number of time bins does not exceed the number of ophys timestamps. Trial matrices are created by boolean indexing with each trial mask.

ii. 
```python
arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
n_t = min(len(a) for a in arrays)
return np.stack([a[:n_t] for a in arrays], axis=0)
...
neural = extract_neural_matrix(signal_df, signal_kind)
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
...
trial_neural = neural[:, m].astype(np.float32)
```

iii. The notes describe this as a minimal-processing choice: use SDK-provided traces directly, align them to the ophys clock, and avoid extra transformations.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron/ROI quality filter in the final script. Every row in the chosen `dff_traces` or `events` table is included.

ii. 
```python
signal_kind, signal_df = choose_signal(exp)
neural = extract_neural_matrix(signal_df, signal_kind)
```

iii. The notes mention Allen ROI curation exists in the source dataset, but the final conversion does not apply any additional valid-ROI or cell-quality filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps within each trial window, not to a separate change-time-centered window. For each trial, the code keeps frames whose timestamps lie between trial `start_time` and `stop_time`.

ii. 
```python
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
...
start = float(tr['start_time'])
stop = float(tr['stop_time'])
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. Step 4 and Step 5 in the notes say the agent wanted all streams aligned on the ophys clock and trials segmented behaviorally, rather than re-centering around one event like `change_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at native ophys frame resolution. The script computes each experiment’s median ophys frame interval in milliseconds and stores the median across sessions as `metadata['time_bin_size']`. No temporal rebinning is applied.

ii. 
```python
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
...
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
...
'time_bin_size': time_bin_size,
```

iii. The notes repeatedly say the conversion should align everything to the existing ophys timestamp grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, specifically `image_name`, `start_time`, `stop_time` (or a derived stop time), and `omitted`.

ii. 
```python
stim = ensure_stim_stop_time(stim_df)
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
...
image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(
    sample_times,
    stim_non_omitted['start_time'].values,
    stim_non_omitted['stop_time'].values,
    image_vals,
    default=blank_idx,
)
```

iii. The sample-stage notes say the agent changed this representation after finding too many “unknown” labels; it decided to assign image identity across full image-presentation intervals instead of only flash instants.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds a global image vocabulary from all non-omitted stimulus presentations in kept experiments, prepends an explicit `blank` class, maps image names to integers, and interval-assigns those labels to each ophys sample in a trial. Any uncovered time bin defaults to the `blank` class.

ii. 
```python
if 'image_name' in stim.columns:
    all_images.update(stim['image_name'].astype(str).unique().tolist())
...
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
...
image_identity = interval_assign(..., image_vals, default=blank_idx)
```

iii. The trajectory shows this was partly a bug fix: training failed when image labels were negative, so the agent added an explicit blank class instead of using `-1`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. For each trial, the code first finds stimulus presentations overlapping the trial and then labels the trial’s ophys `sample_times` by checking which stimulus interval contains each sample.

ii. 
```python
trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
trial_output = build_trial_output(tr, sample_times, trial_stim, ...)
...
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
```

iii. The notes justify this as a direct ophys-time alignment of stimulus identity within behaviorally defined trials.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the change presentation `start_time`.

ii. 
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The notes frame image change as something that should come from presentation-level change markers aligned to ophys time.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes an all-zero vector and then marks a single ophys bin as `1` at the first sample on or after each change presentation start within the trial.

ii. 
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
...
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. No separate prose justification for the one-bin pulse was recorded; it appears to follow the agent’s interpretation of “1 right after a change in image identity.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary. The code uses integer labels `0` and `1`, with output names `['no_change', 'change']`.

ii. 
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
...
'output_values': [
    image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
```

iii. No extra thresholding rationale is needed because the source variable is treated as a binary indicator.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change label is computed on the same per-trial ophys `sample_times` used to slice the neural matrix.

ii. 
```python
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
trial_output = build_trial_output(tr, sample_times, trial_stim, ...)
...
idx = np.searchsorted(sample_times, ct, side='left')
```

iii. The notes consistently say all time-varying outputs should be expressed on the ophys clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed`, specifically timestamp and speed columns normalized to `timestamps` and `speed`.

ii. 
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    cols = {c.lower(): c for c in run_df.columns}
    tcol = cols.get('timestamps', 'timestamps')
    scol = cols.get('speed', None)
    ...
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The notes identify Allen’s running wheel stream as the source for this output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code collects speed values from each kept experiment to compute global percentile bin edges, then within each trial assigns running speed to ophys sample times by nearest-neighbor matching and digitizes the values into bins.

ii. 
```python
run_df = get_running_df(exp)
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
...
run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
...
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. Step 5 in the notes says the agent wanted global percentile bins for consistency. There is no note defending nearest-neighbor interpolation specifically; that appears to be an implementation choice.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile bins using globally computed edges. Non-finite values are forced into bin `0`. If percentile edges are not strictly increasing, later edges are nudged upward with `np.nextafter`.

ii. 
```python
def make_percentile_bins(values, n_bins=5):
    ...
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def digitize(values, edges):
    ...
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
```

iii. The notes explicitly planned 5 equal-percentile bins for continuous outputs.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned trial-by-trial to the same ophys `sample_times` used for neural data, using nearest-neighbor time assignment.

ii. 
```python
sample_times = ophys_timestamps[m]
...
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The notes say all outputs should live on the ophys clock; the code implements that with nearest-neighbor assignment rather than interpolation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is taken from `exp.eye_tracking`, but the exact column is chosen heuristically: `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_radius`, or any numeric column whose name contains `pupil`.

ii. 
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
    for c in eye_tracking.columns:
        if 'pupil' in c.lower() and pd.api.types.is_numeric_dtype(eye_tracking[c]):
            return c
    return None
```

iii. The notes say the agent needed to inspect experiment objects to determine the exact pupil column; the final implementation keeps that decision flexible instead of hard-coding one field.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code gathers raw pupil values across kept experiments to compute global percentile edges. During trial construction it drops only `NaN`/`inf`, then nearest-neighbor assigns pupil values to ophys sample times and digitizes them. There is no blink filtering.

ii. 
```python
eye_df = get_eye_df(exp)
pupil_col = pick_pupil_column(eye_df)
if pupil_col is not None:
    pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
...
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
...
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The notes planned global percentile binning for pupil. They do not record a justification for omitting blink filtering; this appears to be an implementation simplification.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil values are digitized into 5 global percentile bins with the same helper used for running speed. Missing or non-finite values become bin `0`. If no pupil column exists, the entire trial is set to zeros.

ii. 
```python
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
...
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The notes explicitly say continuous outputs should be discretized into 5 percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are aligned to the same per-trial ophys `sample_times` used by neural data, via nearest-neighbor timestamp matching.

ii. 
```python
sample_times = ophys_timestamps[m]
...
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. As with running speed, the notes say the common target clock should be ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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
```

iii. The notes identify these as the canonical SDK outcome fields for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the four boolean outcome fields to integer classes `0` through `3` and then repeats the resulting class across every time bin of the trial. If none of the booleans is true, it falls back to `0`.

ii. 
```python
if bool(trial_row.get('hit', False)):
    outcome = 0
...
else:
    outcome = 0
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
...
'output_values': [
    ...,
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
],
```

iii. Step 5 in the notes says the trial outcome should be a static per-trial category, and the implementation makes it time-constant by broadcasting it across the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The final script handles several edge cases with defaults rather than exceptions: missing NWB files are skipped; experiments with fewer than 2 retained trials are dropped; neural traces are truncated to the shortest common length and to the available ophys timestamps; trials with fewer than 2 frames are skipped; missing/non-finite running or pupil values become bin `0`; absent pupil data become all zeros; missing stimulus stop times are synthesized from the next stimulus onset or a fallback duration; and non-stimulus periods are assigned to an explicit `blank` image class.

ii. 
```python
if not nwb_path.exists():
    continue
...
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
...
if m.sum() < 2:
    continue
...
out[~np.isfinite(vals)] = 0
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
...
default_interval = float(np.nanmedian(diffs)) if len(diffs) else 0.75
...
image_names = ['blank'] + sorted(all_images)
```

iii. Some of these choices are justified in the notes and trajectory: the blank image class was added to avoid invalid labels during training, and dF/F truncation/alignment was used to keep traces synchronized with timestamps. Other defaults, like all-zero pupil bins when no usable pupil column exists, are not explicitly justified in prose.

## 9-a. What are the most time-consuming steps of the code?

i. The obvious bottleneck is experiment loading from NWB files, especially because every kept experiment is loaded once in `collect_global_info()` and again in `process_experiment()`. Secondary cost comes from per-trial and per-presentation labeling loops.

ii. 
```python
for _, row in exp_table.iterrows():
    ...
    exp = load_experiment(exp_id)
...
for i, exp_id in enumerate(chosen_ids, 1):
    ...
    sess = process_experiment(exp_id, ...)
...
for _, tr in trials.iterrows():
    ...
    trial_output = build_trial_output(...)
...
for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
    m = (sample_times >= s) & (sample_times < e)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly flags repeated experiment loading during global percentile/image collection as a likely full-run bottleneck, and the trajectory shows the agent repeatedly monitored full-conversion runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain vectorizable: the per-stimulus `interval_assign()` loop, the per-change loop used to mark `image_change`, the per-trial loop over `trials.iterrows()`, and repeated per-trial pupil preprocessing inside `build_trial_output()`.

ii. 
```python
def interval_assign(...):
    ...
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v

for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
    idx = np.searchsorted(sample_times, ct, side='left')
    ...

for _, tr in trials.iterrows():
    ...

eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
```

iii. The notes only mention one of these explicitly: Step 6 says “interval assignment currently loops over stimulus presentations per trial.” The rest are not separately justified.

## 9-c. What processing does the code repeat multiple times?

i. The code reloads each experiment twice, computes/filters stimulus-presentation stop times more than once, and re-derives pupil preprocessing inside every trial rather than once per experiment.

ii. 
```python
exp = load_experiment(exp_id)        # in collect_global_info
...
sess = process_experiment(exp_id, ...)  # loads again inside process_experiment
...
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
...
stim = ensure_stim_stop_time(stim_df)
...
pupil_col = pick_pupil_column(eye_df)
...
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
```

iii. Step 6 in the notes explicitly acknowledges repeated experiment loading. The other repeated work appears to be an implementation artifact rather than a documented design choice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is creating a zero-filled `brain_region_idx` array inside `process_experiment()` that is not actually used downstream except for its length; the final `brain_region_idx` values are rebuilt later from `region_to_idx`. The code also stores `signal_kind` per session mainly for logging, and repeatedly builds empty trial input arrays even though decoder inputs are always absent.

ii. 
```python
brain_region_idx = np.zeros((neural.shape[0],), dtype=np.int64)
return {
    ...
    'brain_region_idx': brain_region_idx,
    'signal_kind': signal_kind,
    ...
}
...
'brain_region_idx': [np.full((len(s['brain_region_idx']),), region_to_idx[s['region']], dtype=np.int64) for s in sessions],
...
trial_input = np.zeros((0, len(sample_times)), dtype=np.float32)
sess_input.append(trial_input)
```

iii. No explicit justification for this extra work was recorded in the notes or trajectory; it appears incidental to how the script was assembled.
