# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading the experiment table CSV from `project_metadata/ophys_experiment_table.csv`, filtering out passive sessions, then loading each experiment individually via `BehaviorOphysExperiment.from_nwb_path()`. A first pass (`collect_global_info`) loads all experiments to gather global image names and percentile bin edges for running speed and pupil. A second pass (`process_experiment`) loads each experiment again to extract neural, behavioral, and trial data.

ii.
```python
exp_table = load_experiment_table()  # pd.read_csv(EXPT_TABLE)
exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
...
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
```

iii. The AI chose to load from local NWB files via `BehaviorOphysExperiment.from_nwb_path()` rather than the S3 cache API, because the local cache manifest paths were broken. Passive sessions were excluded because trial outcomes are not behaviorally meaningful for passive viewing.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values extracted from the experiment metadata table. Each experiment's `mouse_id` is stored and unique subjects are collected after all sessions are processed.

ii.
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field from the experiment table is the canonical subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each **ophys experiment** (single imaging plane) as one session. This means a single behavioral session with multiple imaging planes produces multiple "sessions" in the output.

ii.
```python
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The AI justified this by noting that each experiment has one coherent neuron set and one brain region/depth assignment (from CONVERSION_NOTES.md Step 4). The reference solution instead groups experiments by `ophys_session_id`, combining neurons from all imaging planes into one session.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's built-in `trials` table. For each experiment, the AI filters to go and catch trials (excluding aborted and auto-rewarded), then extracts ophys frames from `start_time` to `stop_time` using a boolean mask on ophys timestamps.

ii.
```python
trials = exp.trials.copy()
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()
...
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
```

iii. The filtering matches the instructions to include go and catch trials and exclude aborted and auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch, (2) not aborted, (3) not auto-rewarded, (4) must have at least 2 ophys frames in the trial window, (5) sessions with fewer than 2 valid trials are excluded.

ii.
```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
...
if m.sum() < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The AI noted that aborted trials lack a change stimulus, auto-rewarded trials bias behavioral response, and sessions with fewer than 2 trials cannot be used for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces). The AI initially tried `events` but switched to `dff_traces` due to sparse/all-zero issues.

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
    col = None
    for c in preferred:
        if c in signal_df.columns:
            col = c
            break
    ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
```

iii. The AI chose dF/F because it is the standard measure for two-photon calcium imaging and events-based representation produced sparse trials. However, the `choose_signal` function prefers `dff_traces` first but `extract_neural_matrix` prefers the `events` column first if present in the DataFrame.

## 2-b. How is the `neural` data processed?

i. The neural data from dff_traces is extracted into a matrix of shape (n_neurons, n_timepoints). No additional processing (filtering, normalization, neuropil correction) is applied beyond what the Allen SDK provides. Since each experiment is treated as a separate session, there is no merging of neurons across planes.

ii.
```python
def extract_neural_matrix(signal_df, signal_kind):
    ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)
```

iii. The SDK pipeline already handles motion correction, neuropil subtraction, and dF/F normalization. Truncating to `min(len(a))` handles potential length mismatches across neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neural data. All neurons present in the SDK's traces are included.

ii. N/A - no filtering code present.

iii. The AI noted the Allen SDK pipeline already applies quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, ophys frames falling within the trial's `start_time` to `stop_time` window are selected using a boolean mask.

ii.
```python
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
...
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The boolean mask `>=start` and `<stop` extracts ophys frames within the trial window. This is effectively aligned to trial start (the first included frame).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native ophys frame rate (~11 Hz). The time bin size is computed as the median inter-frame interval across sessions.

ii.
```python
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
```

iii. The ophys timestamps are already at a consistent frame rate from the microscope scanning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column. The AI uses `interval_assign` to map each ophys frame to the image being displayed at that time, based on the stimulus presentation `start_time` and `stop_time` intervals.

ii.
```python
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values, stim_non_omitted['stop_time'].values, image_vals, default=blank_idx)
```

iii. The AI used stimulus_presentations for more precise temporal assignment of image identity across the full trial window.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names from stimulus_presentations are mapped to integer codes via a global mapping built from all unique image names across all experiments. A 'blank' category is added for grey-screen periods (when no stimulus is displayed). Omitted flashes are excluded. The `ensure_stim_stop_time` function computes stop times as the start of the next stimulus presentation.

ii.
```python
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
...
def ensure_stim_stop_time(stim, use_full_interval=True):
    ...
    next_starts = np.r_[starts[1:], starts[-1] + default_interval]
    stim['stop_time'] = next_starts
    return stim
```

iii. The 'blank' category handles grey-screen periods between flashes. Stop times are computed to cover the full inter-stimulus interval.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned per ophys frame using `interval_assign`, which checks whether each ophys timestamp falls within a stimulus presentation interval. Frames outside any presentation interval get the 'blank' label.

ii.
```python
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. Using the same ophys timestamps for both neural data and image identity ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table.

ii.
```python
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The AI used `is_change` from stimulus_presentations rather than deriving it from `change_time` and `go` columns in the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created, with value 1 set at the single ophys frame closest to each change event's start time. No temporal window is applied.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The AI marks only a single frame as the change event, rather than a window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1). No thresholding beyond the binary assignment is applied.

ii. See 4-b above.

iii. The instructions specify a binary variable with value 1 "right after a change in image identity."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change frame index is determined by `np.searchsorted` on the trial's ophys timestamps, using the same timestamps as the neural data.

ii. See 4-b above.

iii. Using the same ophys timestamps ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, accessing the `speed` and `timestamps` columns.

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    ...
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is assigned to ophys timepoints using **nearest-neighbor** assignment (not linear interpolation). It is then discretized into 5 percentile-based bins computed across all experiments.

ii.
```python
def nearest_assign(sample_times, source_times, source_values):
    idx = np.searchsorted(source_times, sample_times, side='left')
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(sample_times - source_times[prev_idx]) < np.abs(sample_times - source_times[idx])
    idx[choose_prev] = prev_idx[choose_prev]
    return source_values[idx]

run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. Nearest-neighbor was used instead of linear interpolation. Bin edges are computed globally using inner percentile edges.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges. The edges are computed from the inner percentiles (20th, 40th, 60th, 80th) across all valid running speed values from all experiments. `np.digitize` with `right=False` maps values to bins.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    edges = np.percentile(vals, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def digitize(values, edges):
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out
```

iii. Percentile-based binning ensures roughly equal counts per bin. Non-finite values are mapped to bin 0.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is assigned to ophys timestamps using nearest-neighbor assignment within each trial, using the same ophys timestamps as the neural data.

ii.
```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
```

iii. Using the same `sample_times` (ophys timestamps within the trial) for both neural and running speed data ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI uses a priority search for pupil columns: `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_radius` (in that order).

ii.
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
```

iii. The priority order tries `pupil_diameter` first, which may differ from the reference solution's explicit use of `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil values are assigned to ophys timepoints using **nearest-neighbor** assignment after dropping NaN and infinite values. Then discretized into 5 percentile-based bins globally.

ii.
```python
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. Dropping NaN/inf serves a similar purpose to blink removal but doesn't explicitly use the `likely_blink` flag.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins using `make_percentile_bins` and `digitize`.

ii. See 5-c above.

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: nearest-neighbor assignment to ophys timestamps within each trial.

ii.
```python
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
```

iii. Same rationale as running speed alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

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
    outcome = 0  # fallback to 'hit'
```

iii. These four columns are the SDK's canonical trial outcome labels. The fallback defaults to 0 (hit) if none match.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The integer code is replicated across all time bins within the trial.

ii.
```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
```

iii. The mapping order matches the reference. The fallback to 0 (hit) when no outcome matches is a notable difference from the reference (which uses 'other' / -1).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- Failed experiments: If processing raises an exception, the experiment is skipped (implicit via the for-loop structure, though no explicit try/except in `main`).
- Short trials: Trials with fewer than 2 ophys frames are skipped.
- Missing pupil data: If no pupil column is found, zeros are used. NaN/inf values are dropped before nearest-neighbor assignment.
- Missing running speed: Non-finite values mapped to bin 0 during digitization.
- Edge cases in percentile bins: Ties in bin edges are resolved by nudging with `np.nextafter`.
- Sessions with fewer than 2 trials: Excluded.

ii.
```python
if m.sum() < 2:
    continue
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
...
out[~np.isfinite(vals)] = 0
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. These handling strategies prevent crashes from edge cases in the data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) `collect_global_info` which loads every experiment once to gather image names and compute percentile bin edges, and (2) `process_experiment` which loads each experiment a second time for the actual data extraction. This means every experiment is loaded twice.

ii.
```python
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, ...)
```

iii. Loading NWB files is I/O bound and dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interval_assign` function loops over each stimulus presentation interval to assign image identities. This O(n_stim * n_timepoints) loop could be vectorized using `np.searchsorted` or similar.

ii.
```python
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. The per-stimulus loop creates a boolean mask for each presentation interval, which is inefficient for large numbers of presentations.

## 9-c. What processing does the code repeat multiple times?

i. Every experiment is loaded twice: once in `collect_global_info` to gather global statistics (image names, running/pupil distributions for binning) and once in `process_experiment` for actual data extraction.

ii.
```python
# First pass:
def collect_global_info(exp_table):
    for _, row in exp_table.iterrows():
        exp = load_experiment(exp_id)
        ...

# Second pass:
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, ...)
```

iii. This doubles the I/O time. The reference solution avoids this by collecting running/pupil values during the first (and only) data extraction pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `ensure_stim_stop_time` function recomputes stop times for all stimulus presentations even though the SDK may already provide them. The `choose_signal` function checks for events even though dff is always preferred. The entire `collect_global_info` pass loads trials, stimuli, running, and eye data just to compute bin edges, when this could be done during the main processing pass.

ii. See 9-c above.

iii. The duplicated loading is the main source of unnecessary processing.
