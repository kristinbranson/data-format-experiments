# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local `ophys_experiment_table.csv`, filters out passive sessions at the table level, then loads each retained `ophys_experiment_id` directly from its NWB file with `BehaviorOphysExperiment.from_nwb_path()`. It does not use `VisualBehaviorOphysProjectCache`, does not filter by `project_code`, and does not reconstruct Allen `ophys_session_id` sessions from multiple experiments.

ii. 
```python
EXPT_TABLE = META_DIR / 'ophys_experiment_table.csv'
EXPT_DIR = DATASET_DIR / 'behavior_ophys_experiments'

def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def get_nwb_path(exp_id: int) -> Path:
    return EXPT_DIR / f'behavior_ophys_experiment_{int(exp_id)}.nwb'

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

exp_table = load_experiment_table()
exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
```

iii. The justification in `CONVERSION_NOTES.md` and the trajectory is that one `ophys_experiment` gives one coherent neuron population and one brain-region assignment, so the agent chose experiments as the basic converted unit. It also switched to direct NWB loading after cache-based loading failed in the recorded trajectory.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `mouse_id` column in the experiment metadata. After processing sessions, the AI builds a sorted unique subject list and maps each converted session back to its subject index.

ii.
```python
subject = str(meta_row.get('mouse_id', 'unknown'))
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The agent’s notes identify `mouse_id` as the canonical subject identifier in the Allen metadata tables, so it uses that field directly.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` is treated as one converted session. The AI does not group multiple experiments that share the same `ophys_session_id`.

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

iii. In `CONVERSION_NOTES.md`, the agent explicitly resolved the “session unit” discrepancy by deciding to “Use ophys experiment as one converted session because each experiment has one coherent neuron set and one brain region/depth assignment.”

## 1-d. How are the data split into trials?

i. For each experiment, the AI uses `exp.trials`, keeps go/catch trials, and segments each trial by taking all ophys frames with timestamps in `[start_time, stop_time)`. Trial length is therefore variable.

ii.
```python
trials = exp.trials.copy()
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()

for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
    if m.sum() < 2:
        continue
```

iii. The agent’s notes say “Segment by trial” because trials are behaviorally defined by the SDK and then all time-varying outputs can be assigned within each trial on the ophys timeline.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `go` or `catch`, with `aborted` and `auto_rewarded` removed. Trials with fewer than 2 ophys frames are skipped, and experiments with fewer than 2 retained trials are excluded from the final dataset.

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
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The notes justify this as matching the task’s instruction to include go/catch and exclude aborted/auto-rewarded trials, plus a minimum-trial requirement so the decoder has enough data per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final code prefers `exp.dff_traces`; if that DataFrame is empty, it falls back to `exp.events`. Within the chosen table it picks the first matching trace column from `events`, `filtered_events`, or `dff`.

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

iii. The trajectory shows the agent originally used events, saw sparse/all-zero trial issues, then deliberately switched to “prefer dff over events” while keeping an events fallback for robustness.

## 2-b. How is the `neural` data processed?

i. The chosen trace arrays are converted to `float32`, truncated to a common minimum length, stacked into a neuron-by-time matrix, and then sliced by trial. No additional normalization, denoising, or cross-plane merge is performed.

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

iii. The justification is mostly implicit: the agent treated each experiment as a self-contained recording, so it only needed to stack traces within that experiment and trim them to the available ophys timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no additional neuron-level quality control in the final script. All rows present in the chosen AllenSDK trace table are kept.

ii.
```python
signal_kind, signal_df = choose_signal(exp)
neural = extract_neural_matrix(signal_df, signal_kind)
```

iii. The notes discuss possible SDK ROI curation hooks but the final code applies none of them, effectively trusting the experiment-provided traces as already curated enough.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps within each behaviorally defined trial. For each trial, the script selects all frames whose ophys timestamps fall between the trial `start_time` and `stop_time`.

ii.
```python
start = float(tr['start_time'])
stop = float(tr['stop_time'])
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The notes explicitly say “Align everything on ophys timestamps” because that is required by the task and is the common time base exposed by the AllenSDK objects.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native ophys frame resolution. No temporal rebinning is applied. `time_bin_size` is reported as the median inter-frame interval, summarized across sessions.

ii.
```python
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
...
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
...
'time_bin_size': time_bin_size,
```

iii. The justification in the notes is that all outputs are aligned directly onto the ophys timestamp grid, so native frame timing is the natural bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`, using stimulus interval timing rather than the trial table’s `initial_image_name` and `change_image_name`.

ii.
```python
stim = ensure_stim_stop_time(stim_df)
...
image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(
    sample_times,
    stim_non_omitted['start_time'].values,
    stim_non_omitted['stop_time'].values,
    image_vals,
    default=blank_idx
)
```

iii. The trajectory says the agent moved to presentation-level image identity because the earlier representation produced too many unknown bins, and the paper discusses 750 ms image-presentation intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI computes stimulus stop times from successive presentation starts, drops omitted flashes from the interval list, maps image names to global integer codes, and uses a `blank` category for time bins not covered by a non-omitted presentation.

ii.
```python
def ensure_stim_stop_time(stim, use_full_interval=True):
    ...
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim

if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
...
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
...
default=blank_idx
```

iii. The agent’s explicit justification was that carrying image identity over the full image-presentation interval “better reflect[s] the task” and greatly reduced invalid `-1` labels; it then added an explicit `blank` class so training would not crash on negative labels.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned on the exact `sample_times` used for the trial’s neural frames, via interval membership on the ophys timestamp grid.

ii.
```python
sample_times = ophys_timestamps[m]
...
image_identity = interval_assign(sample_times, ..., default=blank_idx)
...
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```

iii. The notes repeatedly state that all time-varying outputs are aligned to ophys timestamps inside each trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the corresponding presentation `start_time`, not from `trials.change_time`.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The trajectory says the agent wanted image-change timing to follow presentation-level intervals rather than only the trial metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI initializes an all-zero vector and sets a single time bin to 1 at the first ophys frame at or after each change presentation onset within the trial.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
...
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. The trajectory indicates this was paired with the shift to full image-presentation intervals; the agent describes setting image change “at the corresponding interval onset.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is encoded as a binary categorical variable: `0` for `no_change`, `1` for `change`.

ii.
```python
'output_values': [
    image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
],
```

iii. This follows the task specification that image change should be binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned to the trial’s neural sample times by searching for the first ophys frame at or after each change-presentation start.

ii.
```python
for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
    idx = np.searchsorted(sample_times, ct, side='left')
    if idx < len(image_change):
        image_change[idx] = 1
```

iii. The same general alignment rule is used for all outputs: project them onto the trial’s ophys timestamps.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed`, specifically its timestamp and speed columns.

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    ...
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The agent’s notes identify `BehaviorOphysExperiment.running_speed` as the standard locomotion stream to use.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI gathers running values globally from retained experiments to compute 5 percentile bin edges. Within each trial, it assigns running speed to ophys frames by nearest-neighbor timestamp matching and digitizes the matched values into bins. Non-finite values are assigned to bin 0.

ii.
```python
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
...
run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
...
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)

def digitize(values, edges):
    vals = np.asarray(values, dtype=float)
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out
```

iii. The notes justify the global percentile discretization as matching the decoder spec; there is no explicit defense of nearest-neighbor assignment versus interpolation in the final notes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five percentile-based bins, labeled `bin_0` through `bin_4`.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    ...
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, qs)
...
[f'bin_{i}' for i in range(5)]
```

iii. This directly follows the task requirement for five equal-percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by nearest-neighbor sampling onto each trial’s ophys timestamps.

ii.
```python
sample_times = ophys_timestamps[m]
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The general justification is the same as elsewhere in the notes: all time-varying streams are mapped onto the ophys timestamp grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, using whichever numeric pupil-related column is found first among `pupil_diameter`, `pupil_area`, `pupil_width`, or `pupil_radius`.

ii.
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
```

iii. The code is written defensively for schema variation. The notes mention that the exact pupil column had to be identified from the experiment object.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI gathers all retained pupil values globally to build 5 percentile bin edges. For each trial, it drops NaN/Inf rows from the chosen pupil column, samples the nearest eye-tracking value at each ophys timestamp, and digitizes the result. If there is no usable pupil column or no valid rows, it fills the trial with zeros.

ii.
```python
pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
...
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
...
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
if len(eye_valid) == 0:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
else:
    pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
    pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The main stated rationale is robustness to missing columns and missing values. The final code does not mention blink filtering.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five percentile-based bins, labeled `bin_0` through `bin_4`.

ii.
```python
pupil_edges = make_percentile_bins(...)
...
[f'bin_{i}' for i in range(5)]
```

iii. This matches the decoder specification for five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned to neural data by nearest-neighbor assignment onto the per-trial ophys timestamps.

ii.
```python
sample_times = ophys_timestamps[m]
...
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. As with the other continuous outputs, the notes say the ophys timestamp grid is the shared alignment target.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes identify these as the AllenSDK’s canonical outcome labels for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome labels to integer codes 0 to 3 and then repeats the trial’s outcome code across every time bin in the trial output matrix.

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
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
```

iii. The justification is implicit: the task calls trial outcome a static per-trial variable, but the output format used by the script is uniformly time-varying, so the code broadcasts the label across the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases by substitution or skipping:
- Missing NWB files are skipped.
- Missing/short trials are skipped.
- Missing or non-finite running and pupil values are digitized to bin 0.
- Missing pupil columns or empty valid pupil rows produce all-zero pupil bins.
- Time bins not covered by a non-omitted image get the explicit `blank` image class.
- Neural traces and timestamps are truncated to the shorter length if they disagree.

ii.
```python
if not nwb_path.exists():
    continue
...
if m.sum() < 2:
    continue
...
out[~np.isfinite(vals)] = 0
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
...
default=blank_idx
...
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
```

iii. The justifications given in notes and trajectory are pragmatic: keep the pipeline running, avoid invalid labels, and be robust to Allen file-schema quirks encountered during debugging.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeated experiment loading from NWB files. The code loads every retained experiment once in `collect_global_info()` and again in `process_experiment()`. Trial-wise interval assignment is a smaller but repeated per-trial cost.

ii.
```python
for _, row in exp_table.iterrows():
    ...
    exp = load_experiment(exp_id)
    ...

for i, exp_id in enumerate(chosen_ids, 1):
    ...
    sess = process_experiment(exp_id, ...)
```

iii. `CONVERSION_NOTES.md` explicitly flags “Repeated experiment loading during global percentile/image collection may be slow for full conversion.”

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-interval loop in `interval_assign()`, the per-change loop for `image_change`, and the repeated Python-level iteration over trials and experiments.

ii.
```python
for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v
...
for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
    idx = np.searchsorted(sample_times, ct, side='left')
    ...
...
for _, tr in trials.iterrows():
    ...
for _, row in exp_table.iterrows():
    ...
```

iii. The notes specifically call out interval assignment as a loop-based hotspot and acknowledge repeated experiment-level processing.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work:
- It loads the same experiment twice, once for global bin/image collection and once for actual conversion.
- It filters the same trial table twice, in `collect_global_info()` and again in `process_experiment()`.
- It reconstructs stimulus stop times globally and then again per trial slice.

ii.
```python
exp = load_experiment(exp_id)
trials = exp.trials.copy()
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
...
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
...
sess = process_experiment(exp_id, ...)
...
trials = exp.trials.copy()
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
...
trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
trial_output = build_trial_output(...)
```

iii. The agent documented repeated experiment loading as a known inefficiency in Step 6 of `CONVERSION_NOTES.md`.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some work that is not needed by the final saved dataset:
- It stores session-local fields like `exp_id`, `signal_kind`, `n_trials`, and a placeholder `brain_region_idx`, then drops them when building the final output.
- It computes a placeholder all-zero `brain_region_idx` array inside `process_experiment()` but later ignores it and rebuilds region indices from `region`.
- It defines a `--show-processing` flag but never uses it.

ii.
```python
brain_region_idx = np.zeros((neural.shape[0],), dtype=np.int64)
return {
    'exp_id': int(exp_id),
    ...
    'brain_region_idx': brain_region_idx,
    'signal_kind': signal_kind,
    'n_trials': len(sess_neural),
    'dt_ms': ...
}
...
'brain_region_idx': [np.full((len(s['brain_region_idx']),), region_to_idx[s['region']], dtype=np.int64) for s in sessions],
```

iii. There is no explicit justification for these extra fields in the notes; they appear to be leftovers from internal bookkeeping during development.
