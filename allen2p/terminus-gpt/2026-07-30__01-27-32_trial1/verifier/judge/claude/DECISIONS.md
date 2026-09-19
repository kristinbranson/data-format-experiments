# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `BehaviorOphysExperiment.from_nwb_path()` to load individual NWB files from disk. It first reads the experiment table CSV, filters out passive sessions, then does a two-pass approach: (1) `collect_global_info` loads every experiment to gather global image names and compute running/pupil percentile bin edges, (2) `process_experiment` loads each experiment again to extract neural and behavioral data. This means every experiment is loaded twice.

ii.
```python
EXPT_TABLE = META_DIR / 'ophys_experiment_table.csv'

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

# First pass: collect_global_info loads all experiments
def collect_global_info(exp_table):
    for _, row in exp_table.iterrows():
        exp = load_experiment(exp_id)
        ...

# Second pass: process each experiment again
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
```

iii. The AI chose to load from NWB files directly rather than using the SDK cache, noting that the local-cache manifest path was broken. The two-pass approach was used to first compute global percentile bin edges and image mappings, then process experiments individually.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from the experiment metadata table. Unique mouse IDs are collected from all processed sessions.

ii.
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
subject = str(meta_row.get('mouse_id', 'unknown'))
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each **ophys experiment** (one imaging plane) as a separate session. This differs from the reference, which groups experiments by `ophys_session_id` to merge multiple imaging planes recorded simultaneously into a single session with combined neurons.

ii.
```python
def process_experiment(exp_id, meta_row, run_edges, pupil_edges, image_to_idx, blank_idx):
    exp = load_experiment(int(exp_id))
    # ... processes one experiment as one session
    return {
        'exp_id': int(exp_id),
        'neural': sess_neural,
        ...
    }

# Each experiment becomes a session
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The AI documented this decision in CONVERSION_NOTES.md Step 4: "Use ophys experiments as sessions: Each experiment has one coherent neuron population and region assignment, whereas one ophys session may contain multiple experiments/planes." This yields 202 sessions from 38 subjects, with one subject (457841) having 34 sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `exp.trials` table. Each non-aborted, non-auto-rewarded go or catch trial is included. The trial window spans from `start_time` to `stop_time`, yielding variable-length trials. Ophys frames are selected using a boolean mask `(ophys_timestamps >= start) & (ophys_timestamps < stop)`.

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

iii. The AI followed the instructions to include go and catch trials and exclude aborted and auto-rewarded trials. The full trial window (start_time to stop_time) is used to capture time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 ophys frames within the window are skipped. Sessions with fewer than 2 valid trials are excluded. Failed experiment loads are caught by try/except and skipped. The AI also explicitly filters `go | catch` and excludes `aborted | auto_rewarded`.

ii.
```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
# ...
if m.sum() < 2:
    continue
# ...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. These filters follow the instructions. The reference also requires `change_time.notna()`, which the AI does not explicitly check but is implicitly handled since go/catch trials generally have valid change times.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from dF/F traces (`exp.dff_traces`). The AI initially considered using event traces but switched to dF/F after finding that events produced sparse/all-zero trials.

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
    # ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
```

iii. The AI documented in CONVERSION_NOTES.md Step 10: "Events-based neural representation produced sparse/all-zero trial warning and poor sample decoding: resolved by switching to dF/F traces." Note: the `extract_neural_matrix` function prefers `events` column if available in the dff DataFrame, but `dff_traces` DataFrames have a `dff` column, so in practice it uses `dff`.

## 2-b. How is the `neural` data processed?

i. The neural data is extracted as a matrix of shape (n_neurons, n_timepoints) per experiment. Since each experiment is treated as a separate session (single imaging plane), there is no merging of neurons across planes.

ii.
```python
def extract_neural_matrix(signal_df, signal_kind):
    # ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)
```

iii. The arrays are truncated to the minimum length across neurons, then stacked. This handles any minor length mismatches between neuron traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neural data. All neurons in the experiment's dff_traces are included.

ii. N/A - no filtering code present.

iii. The AI relied on the SDK's built-in quality control. The reference solution also does not apply additional neural filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. Ophys frames are selected using a boolean mask on timestamps within [start_time, stop_time). This gives variable-length trial windows.

ii.
```python
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
# ...
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The boolean mask approach is functionally similar to the reference's `np.searchsorted` approach but uses `>=` and `<` for boundary handling.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Data is kept at the native ophys frame rate (~11 Hz). The time bin size is computed as the median of inter-frame intervals across all sessions.

ii.
```python
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
```

iii. The native frame rate is preserved, matching the reference approach.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name` via the `interval_assign` function. The AI assigns the image displayed during each stimulus presentation interval to the corresponding ophys frames, with a "blank" category (index 0) for grey/inter-stimulus periods and omitted flashes.

ii.
```python
stim = ensure_stim_stop_time(stim_df)
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
else:
    stim_non_omitted = stim

image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values,
                                  stim_non_omitted['stop_time'].values, image_vals, default=blank_idx)
```

iii. The AI uses the stimulus_presentations table for fine-grained image identity assignment, including handling of omitted flashes and grey periods. This differs from the reference which uses `initial_image_name` and `change_image_name` from the trials table with a switch at `change_time`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected globally across all experiments, sorted, and a "blank" entry is prepended. Each name is mapped to an integer index. The `interval_assign` function assigns image codes to ophys frames based on stimulus presentation start/stop times. Frames outside any presentation interval receive the "blank" code. The `ensure_stim_stop_time` function computes stop times as the start time of the next presentation if not available.

ii.
```python
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}

def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. The "blank" category captures grey inter-stimulus intervals. The reference instead assigns continuous image identity (initial image before change, change image after change) with no blank periods, which is a simpler but different approach.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned to the same ophys timestamp grid as the neural data, using the `interval_assign` function that operates on the trial's `sample_times` (the ophys timestamps within the trial window).

ii.
```python
trial_output = build_trial_output(tr, sample_times, trial_stim, ...)
# sample_times are the same ophys_timestamps used for trial_neural
```

iii. Both neural and output data use the same `sample_times` array, guaranteeing alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `stimulus_presentations`. For each stimulus presentation where `is_change` is True, the ophys frame at the presentation's start_time is set to 1.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The `is_change` flag from stimulus_presentations marks presentations where the image identity actually changed, which naturally excludes catch trials (where no real change occurs).

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created with 1 set at a single ophys frame (the frame at or just after the change stimulus start time). Only presentations with `is_change == True` are marked.

ii. See 4-a above.

iii. The AI marks only a single frame, while the reference marks a 750ms window (one flash + grey period). The instructions say "Have value of 1 right after a change in image identity, otherwise 0" which is ambiguous regarding duration. The single-frame approach results in image_change being 1 for only ~0.4% of all timepoints, which makes it very sparse and harder to decode.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1), no thresholding needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change uses the same `sample_times` (ophys timestamps) as the neural data. `np.searchsorted` maps the change time to the corresponding frame index.

ii.
```python
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. Aligned via the same ophys timestamp grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, specifically the `speed` column with `timestamps`.

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    # ... column name normalization
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The SDK's `running_speed` attribute provides wheel-encoder-derived running speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is assigned to ophys timestamps using **nearest-neighbor** assignment (not linear interpolation). It is then discretized into 5 percentile-based bins using global percentile edges computed from all raw running speed values across all experiments.

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

iii. Nearest-neighbor was chosen over interpolation. The reference uses linear interpolation (`scipy.interpolate.interp1d`). Percentile bin edges are computed from raw values at native timestamps rather than from values already interpolated to ophys timestamps.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins (0-4). Bin edges are the 20th, 40th, 60th, and 80th percentiles of all valid running speed values. Non-finite values are mapped to bin 0.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, qs)
    # Handle ties
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def digitize(values, edges):
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out
```

iii. The AI's approach to computing inner edges and using `np.digitize` produces 5 bins (0-4), matching the reference's approach functionally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is assigned to ophys timestamps via nearest-neighbor within each trial, using the same `sample_times` as neural data.

ii.
```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
```

iii. Uses nearest-neighbor rather than linear interpolation, but both methods align to the same ophys timestamp grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI uses `pick_pupil_column` which tries columns in order: `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_radius`.

ii.
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
    # fallback: any column with 'pupil' in name
```

iii. The reference specifically uses `pupil_width`. The AI's preference order puts `pupil_diameter` first, which would be selected if available. This could result in different values since pupil_diameter and pupil_width are different measurements.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil values are assigned to ophys timestamps using nearest-neighbor. Invalid values (NaN, inf) are dropped before assignment. No blink removal is performed. The values are then discretized into 5 percentile bins using global edges.

ii.
```python
pupil_col = pick_pupil_column(eye_df)
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The AI drops NaN/inf values but does NOT use the `likely_blink` flag to remove blink artifacts. The reference explicitly removes blink frames before interpolation: `eye_clean = eye[~eye['likely_blink']]`. This is a notable omission.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins with global edges.

ii. Same as running speed discretization.

iii. Same percentile-based approach as reference.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same nearest-neighbor approach as running speed, using the trial's ophys timestamps.

ii.
```python
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
```

iii. Uses nearest-neighbor instead of linear interpolation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

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

iii. The same four outcome categories as the reference. The fallback to outcome=0 (hit) differs from the reference which uses 'other'.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integers 0-3. The outcome is replicated across all timepoints in the trial as a static per-trial variable.

ii.
```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
```

iii. Same approach as reference -- a constant value across all frames in the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing pupil data**: If no pupil column is found, all pupil bins default to 0. Invalid values (NaN, inf) are dropped before nearest-neighbor assignment. Non-finite values after assignment are mapped to bin 0.
- **Missing NWB files**: `collect_global_info` checks `nwb_path.exists()` and skips missing files.
- **Short trials**: Trials with fewer than 2 ophys frames are skipped.
- **Short sessions**: Sessions with fewer than 2 trials are excluded.
- **Fallback trial outcome**: If no outcome boolean is True, defaults to hit (0).

ii.
```python
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
# ...
out[~np.isfinite(vals)] = 0
# ...
if m.sum() < 2:
    continue
```

iii. These are reasonable defaults, though the fallback to hit (0) for unknown outcomes could silently misclassify trials.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB experiment files, which happens TWICE -- once in `collect_global_info` and again in `process_experiment`. Each experiment is loaded via `BehaviorOphysExperiment.from_nwb_path()`.

ii.
```python
# First load in collect_global_info:
exp = load_experiment(exp_id)
# Second load in process_experiment:
exp = load_experiment(int(exp_id))
```

iii. Loading each experiment twice doubles the I/O time, which is the primary bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interval_assign` function loops over each stimulus presentation interval, which could be vectorized using `np.searchsorted` and array operations.

ii.
```python
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. Each call to `interval_assign` iterates over all stimulus presentations, applying a boolean mask per presentation. This could be done with vectorized searchsorted operations.

## 9-c. What processing does the code repeat multiple times?

i. The code loads every experiment twice: once during `collect_global_info` to compute global bin edges and image mappings, and again during the main processing loop in `process_experiment`. This doubles all NWB file I/O.

ii.
```python
# Pass 1: collect_global_info
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)

# Pass 2: process each experiment again
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, ...)
```

iii. The reference avoids this by loading each session once, storing trial-level data, then computing bin edges from the stored data before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `ensure_stim_stop_time` for each trial's stimulus presentations, which derives stop times from next-presentation start times. This is used for image identity assignment but the full stop-time computation adds overhead. The "blank" image category adds an extra output class that may not be useful for decoding if it's a small fraction of timepoints (3.4% of frames).

ii.
```python
def ensure_stim_stop_time(stim, use_full_interval=True):
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim
```

iii. The reference avoids this by using a simpler approach with initial_image_name and change_image_name from the trials table.
