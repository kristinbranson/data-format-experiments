# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script does not use `VisualBehaviorOphysProjectCache`. It reads the local experiment metadata CSV, filters out passive sessions, then iterates over `ophys_experiment_id` rows. Each retained experiment NWB is loaded directly with `BehaviorOphysExperiment.from_nwb_path()`. The code does this twice: once in `collect_global_info()` to gather global image/bin metadata and pick experiments with at least 2 kept trials, and again in `process_experiment()` to build the final dataset.

ii.
```python
def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

def collect_global_info(exp_table):
    for _, row in exp_table.iterrows():
        exp_id = int(row['ophys_experiment_id'])
        nwb_path = get_nwb_path(exp_id)
        if not nwb_path.exists():
            continue
        exp = load_experiment(exp_id)
```

```python
exp_table = load_experiment_table()
exp_table = exp_table[
    ~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)
].copy()
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
```

iii. The notes say direct NWB loading was chosen because it “avoided broken local-cache manifest path,” and later note that repeated experiment loading is a known runtime cost.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by `mouse_id` from experiment metadata. After all retained experiments are processed, the script creates a sorted unique subject list and maps each converted session to that subject index.

ii.
```python
region = str(meta_row.get('targeted_structure', 'unknown'))
subject = str(meta_row.get('mouse_id', 'unknown'))
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes explicitly map `metadata.mouse_id` to `subjects / subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI chose to treat each `ophys_experiment_id` as one converted session, not each `ophys_session_id`. A converted “session” is therefore one NWB experiment file, after passive-session exclusion and later trial-count filtering.

ii.
```python
for _, row in exp_table.iterrows():
    exp_id = int(row['ophys_experiment_id'])
    ...
    chosen_ids.append(exp_id)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The notes and README state this decision directly: “Use ophys experiments as sessions” / “one ophys experiment per converted session,” justified by one experiment having one coherent neuron population and one region assignment.

## 1-d. How are the data split into trials?

i. Trials are taken from `exp.trials`. Within each retained experiment, the script iterates over trial rows and uses the trial’s `start_time` and `stop_time` to slice ophys timestamps and neural/activity-aligned outputs. Trials are variable length.

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

iii. The notes say to “Segment by trial” and to align time-varying outputs “within each trial” on the ophys timestamp grid.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if `go` or `catch` is true and both `aborted` and `auto_rewarded` are false. Trials with fewer than 2 ophys frames are skipped. At the experiment level, only experiments with at least 2 such kept trials are processed into the final dataset. The code does not explicitly require non-null `change_time`.

ii.
```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()
...
if m.sum() < 2:
    continue
```

```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
if keep.sum() < 2:
    continue
```

iii. The notes justify go/catch-only inclusion as matching the task instructions and say passive sessions were excluded because trial outcome would not be meaningful there.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final documentation says the neural signal was switched to dF/F traces, but the actual script still implements a fallback: use `exp.dff_traces` if present and non-empty, otherwise use `exp.events`. `extract_neural_matrix()` then pulls whichever trace column is available from `['events', 'filtered_events', 'dff']`.

ii.
```python
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()
    return 'events', events
```

```python
preferred = ['events', 'filtered_events', 'dff']
for c in preferred:
    if c in signal_df.columns:
        col = c
        break
```

iii. The notes record a late-stage decision that “Events-based neural representation produced sparse/all-zero trial warning ... resolved by switching to dF/F traces,” but the code and metadata still say `dff_preferred_else_events`.

## 2-b. How is the `neural` data processed?

i. After choosing a signal table, the script finds one trace column, converts each row to `float32`, truncates all traces to the minimum shared length, stacks them into a neuron-by-time matrix, then truncates again to the length of `ophys_timestamps`. No further normalization is applied, and no multi-plane/session merging is done because the unit is one experiment.

ii.
```python
arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
n_t = min(len(a) for a in arrays)
return np.stack([a[:n_t] for a in arrays], axis=0)
```

```python
signal_kind, signal_df = choose_signal(exp)
neural = extract_neural_matrix(signal_df, signal_kind)
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
```

iii. The notes frame this as using SDK-provided processed traces and later say the event-based version was replaced by dF/F because events were too sparse.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality control is applied in `convert_data.py`. The script uses whichever rows are present in the chosen dF/F or events table.

ii.
```python
signal_kind, signal_df = choose_signal(exp)
neural = extract_neural_matrix(signal_df, signal_kind)
```

iii. The notes mention possible ROI-validity hooks in AllenSDK exploration, but no explicit neuron filtering rule was implemented in the final script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps inside each behaviorally defined trial. For each trial, the code takes frames whose `ophys_timestamps` satisfy `start_time <= t < stop_time`.

ii.
```python
start = float(tr['start_time'])
stop = float(tr['stop_time'])
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The notes explicitly say “Align everything on ophys timestamps” and “Segment by trial.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native ophys frame rate. It does not temporally rebin neural data. `time_bin_size` is reported as the median inter-frame interval in milliseconds, aggregated across retained sessions by median of per-session `dt_ms`.

ii.
```python
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
...
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
...
'time_bin_size': time_bin_size,
```

iii. The notes describe “native ophys timestamps” as the temporal unit and do not mention any rebinning step.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`, not from the trial table’s `initial_image_name` and `change_image_name`. The code gathers image names globally from non-omitted stimulus presentations and then assigns them to ophys sample times inside each trial.

ii.
```python
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
...
if 'image_name' in stim.columns:
    all_images.update(stim['image_name'].astype(str).unique().tolist())
```

```python
image_vals = [image_to_idx.get(str(x), blank_idx)
              for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(
    sample_times,
    stim_non_omitted['start_time'].values,
    stim_non_omitted['stop_time'].values,
    image_vals,
    default=blank_idx)
```

iii. The notes justify this by saying image identity should be assigned from `stimulus_presentations` over “full image-presentation intervals,” and later mention adding a `blank` class after validation failures.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code fills or reconstructs stimulus `stop_time`, removes omitted flashes, builds a global image vocabulary from all retained experiments, prepends a `blank` category, maps names to integers, and then assigns those category IDs across each trial’s ophys time bins using interval membership.

ii.
```python
def ensure_stim_stop_time(stim, use_full_interval=True):
    ...
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim
```

```python
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
...
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
```

iii. The notes say this changed after “invalid negative image labels caused decoder training failure,” which was resolved by adding an explicit `blank` class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The code aligns image identity on the same `sample_times` array used for per-trial neural slices. It uses interval assignment over the stimulus presentations that overlap the trial window, so each ophys time bin receives the image category active at that timestamp.

ii.
```python
sample_times = ophys_timestamps[m]
trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
trial_output = build_trial_output(
    tr, sample_times, trial_stim, run_df, eye_df,
    run_edges, pupil_edges, image_to_idx, blank_idx)
```

```python
image_identity = interval_assign(sample_times, ..., default=blank_idx)
```

iii. The notes say all time-varying streams should be “assigned onto the ophys timestamp grid within each trial.”

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the presentation `start_time` values within the trial, not from the trial table’s `change_time` and `go` columns.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[
        stim_non_omitted['is_change'].fillna(False), 'start_time'
    ].values:
        idx = np.searchsorted(sample_times, ct, side='left')
```

iii. The mapping plan in the notes explicitly ties image change to `stimulus_presentations.is_change` and image-identity transitions.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes an all-zero vector and, for each presentation flagged `is_change`, sets exactly one ophys frame to 1: the first frame at or after that presentation’s `start_time`. It does not mark a 750 ms change window.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The notes describe this output as a binary time series with 1 “immediately after an image identity change,” but do not give a separate justification for using a one-frame pulse.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous thresholding is used. The variable is directly encoded as a binary category vector with `0 = no_change` and `1 = change`.

ii.
```python
'output_values': [
    image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
```

iii. This follows the notes’ plan to represent image change as a binary time series.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The code aligns the change indicator on trial-local `sample_times`, using `np.searchsorted()` to place the change at the first ophys frame at or after each `is_change` presentation start time.

ii.
```python
sample_times = ophys_timestamps[m]
...
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. The notes repeatedly state that all outputs are aligned to the ophys timestamp grid within each trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, specifically a timestamps column plus a speed column.

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    ...
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The notes map `BehaviorOphysExperiment.running_speed` to the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script collects raw running-speed samples globally to compute percentile bin edges, then within each trial assigns running speed to ophys frames by nearest-neighbor timestamp matching and digitizes those values into 5 bins. It does not linearly interpolate onto the full ophys timebase.

ii.
```python
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
```

```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The notes originally planned to “Interpolate/assign onto ophys timestamps,” but no explicit final rationale for nearest-neighbor assignment appears in the notes or trajectory.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five global percentile bins. Non-finite values are assigned to category 0.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    ...
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, qs)
```

```python
def digitize(values, edges):
    vals = np.asarray(values, dtype=float)
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out
```

iii. The notes explicitly say to “Discretize running speed ... into 5 percentile bins” and compute cut points on valid retained samples.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to neural data by nearest timestamp assignment from `running_speed.timestamps` to each trial’s `sample_times` (the ophys frames inside that trial).

ii.
```python
sample_times = ophys_timestamps[m]
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The notes say all time-varying outputs are aligned onto ophys timestamps, but they do not justify the specific nearest-neighbor choice over interpolation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, but the exact measurement column is chosen heuristically from `['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']` or any numeric column containing “pupil”. The code does not explicitly use `likely_blink`.

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

iii. The notes initially said the exact pupil field still needed to be identified; no later justification was given beyond choosing a workable pupil column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script gathers global raw pupil values from the chosen pupil column to compute percentile bin edges. For each trial, it drops non-finite pupil samples, assigns the nearest remaining eye-tracking value to each ophys frame, and digitizes into 5 bins. If no pupil column exists or all values are invalid, it fills the trial with zeros. It does not exclude likely blinks.

ii.
```python
eye_df = get_eye_df(exp)
pupil_col = pick_pupil_column(eye_df)
if pupil_col is not None:
    pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
```

```python
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
if len(eye_valid) == 0:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
else:
    pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
    pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The notes planned “mask invalid values” and percentile-bin pupil size, but do not mention or justify omission of blink removal in the final script.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global percentile bins, with non-finite values mapped to category 0.

ii.
```python
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
...
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The notes explicitly planned 5 percentile bins for pupil diameter.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is aligned by nearest-neighbor timestamp assignment from eye-tracking timestamps to the trial’s ophys `sample_times`.

ii.
```python
sample_times = ophys_timestamps[m]
...
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The notes say the stream should be assigned onto the ophys timestamp grid within each trial, but do not justify why nearest-neighbor was used instead of interpolation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes map exactly these four trial outcome fields from `Trials / Trial`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the four boolean outcome fields to integer categories `0..3`, with a default of 0 if none are set, then repeats that category across every time bin in the trial.

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

iii. The notes explicitly planned a “Static per-trial categorical label replicated across timepoints.”

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script skips missing NWB files, skips experiments with fewer than 2 kept trials, skips trials with fewer than 2 ophys frames, maps non-finite running/pupil values to bin 0, uses all-zero pupil bins if no usable pupil data exist, and introduced an explicit `blank` image class to avoid invalid image labels during decoder validation.

ii.
```python
if not nwb_path.exists():
    continue
...
if keep.sum() < 2:
    continue
...
if m.sum() < 2:
    continue
```

```python
out = np.digitize(vals, edges, right=False).astype(np.int64)
out[~np.isfinite(vals)] = 0
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
```

iii. The notes explicitly mention fixing “invalid negative image labels” by adding a `blank` image class and treating runtime warning spam as non-fatal.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeatedly loading experiment NWB files through `BehaviorOphysExperiment.from_nwb_path()`. The script loads each retained experiment once in `collect_global_info()` and again in `process_experiment()`, so I/O dominates runtime.

ii.
```python
for _, row in exp_table.iterrows():
    ...
    exp = load_experiment(exp_id)
```

```python
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
```

iii. The notes explicitly call out “Repeated experiment loading during global percentile/image collection” as slow and say full conversion runtime was long.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial `iterrows()` loop, the per-stimulus interval loop inside `interval_assign()`, and the per-change loop for `image_change`. The global `exp_table.iterrows()` pass is also pure Python and reloads experiments serially.

ii.
```python
for _, tr in trials.iterrows():
    ...
```

```python
for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v
```

```python
for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
    idx = np.searchsorted(sample_times, ct, side='left')
```

iii. The notes explicitly mention that “Interval assignment currently loops over stimulus presentations per trial.”

## 9-c. What processing does the code repeat multiple times?

i. The code repeats full experiment loading in two passes, computes stimulus stop times globally and then again on each trial subset, and repeats trial-local stimulus subsetting/assignment for every trial.

ii.
```python
exp = load_experiment(exp_id)
...
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
...
sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
```

```python
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
...
trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
trial_output = build_trial_output(tr, sample_times, trial_stim, ...)
```

iii. The notes directly acknowledge repeated experiment loading as an inefficiency; they do not justify the repeated stimulus interval reconstruction.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores temporary per-session fields such as `exp_id`, `signal_kind`, and an all-zero `brain_region_idx` in the intermediate `sess` dict even though the final output rebuilds region indices from session-level region names and drops the other fields. It also computes global/per-trial stimulus stop-time expansions and blank-interval labeling that are only needed because of the chosen image-identity representation, not by downstream decoder format itself.

ii.
```python
return {
    'exp_id': int(exp_id),
    'subject': subject,
    'region': region,
    ...
    'brain_region_idx': brain_region_idx,
    'signal_kind': signal_kind,
    'n_trials': len(sess_neural),
    'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
}
```

```python
'brain_region_idx': [
    np.full((len(s['brain_region_idx']),), region_to_idx[s['region']], dtype=np.int64)
    for s in sessions
],
```

iii. The notes do not justify these extra intermediate fields; they mainly acknowledge runtime cost and validation-driven additions like the `blank` class.
