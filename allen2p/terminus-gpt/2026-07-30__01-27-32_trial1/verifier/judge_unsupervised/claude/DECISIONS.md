# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading the ophys experiment table CSV (`project_metadata/ophys_experiment_table.csv`) to get experiment IDs and metadata. It filters out passive sessions by excluding rows where `session_type` contains "passive". It then iterates through the remaining experiment IDs, loading each experiment via `BehaviorOphysExperiment.from_nwb_path()` from individual NWB files. Data is loaded in two passes: first in `collect_global_info()` to gather global image names, running speed values, and pupil diameter values for percentile bin computation; then again in `process_experiment()` to extract neural, stimulus, behavioral, and trial data per experiment.

ii.
```python
DATASET_DIR = Path('data/visual-behavior-ophys-1.1.0')
META_DIR = DATASET_DIR / 'project_metadata'
EXPT_TABLE = META_DIR / 'ophys_experiment_table.csv'
EXPT_DIR = DATASET_DIR / 'behavior_ophys_experiments'

def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

# In main():
exp_table = load_experiment_table()
exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
```

iii. The agent noted in CONVERSION_NOTES.md Step 4 that it chose to use `BehaviorOphysExperiment.from_nwb_path` to bypass a broken project-cache manifest path. The trajectory (Step 20) confirms this was a pragmatic workaround after the standard `VisualBehaviorOphysProjectCache.from_local_cache()` approach failed.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the `mouse_id` column in the experiment metadata table. After processing all experiments, unique subjects are collected from the processed sessions and sorted. A `subject_idx` array maps each session to its subject index.

ii.
```python
# In process_experiment():
subject = str(meta_row.get('mouse_id', 'unknown'))

# In main():
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
data['subjects'] = subjects
data['subject_idx'] = np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64)
```

iii. The agent documented in Step 5 of CONVERSION_NOTES.md that subject identity comes from `metadata.mouse_id`, and session order follows the converted experiment order.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (not ophys session) is treated as one "session" in the converted data. This means a single behavioral session with multiple imaging planes (experiments) produces multiple converted sessions. Only experiments with available NWB files and at least 2 valid trials are included as sessions.

ii.
```python
# In main():
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The agent documented in CONVERSION_NOTES.md Step 4: "Use ophys experiment as one converted session because each experiment has one coherent neuron set and one brain region/depth assignment." This is a reasonable choice since multiscope data have multiple planes per session.

## 1-d. How are the data split into trials?

i. Trials are defined from the SDK's `exp.trials` table. Each trial has `start_time` and `stop_time`. Neural data is sliced by selecting ophys timestamps within `[start_time, stop_time)`. Only trials with at least 2 timepoints in this window are kept.

ii.
```python
# In process_experiment():
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

iii. The agent noted in CONVERSION_NOTES.md Steps 1 and 4 that trial segmentation follows the SDK's trial table with behaviorally defined start/stop times, and trials are aligned to ophys timestamps within each trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only `go` or `catch` trials, (2) excluding `aborted` trials, (3) excluding `auto_rewarded` trials, (4) requiring at least 2 ophys timepoints within the trial window. Passive sessions are excluded at the session level.

ii.
```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()
# ...
if m.sum() < 2:
    continue
```

iii. The agent justified this in CONVERSION_NOTES.md Steps 1, 3, and 4, citing the SDK trial logic that `aborted` trials should be excluded and only `go`/`catch` trials included. The instructions explicitly state "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `exp.dff_traces` (dF/F traces). The code first attempts to use dF/F, falling back to `exp.events` if dF/F is unavailable. In practice, all sessions used dF/F.

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
    # ... extracts array from the identified column
```

iii. The agent initially tried using `events` but found it produced sparse/all-zero trials and poor decoder performance (trajectory Step 26-27). The switch to dF/F resolved these issues. CONVERSION_NOTES.md Step 10 documents: "Events-based neural representation produced sparse/all-zero trial warning and poor sample decoding: resolved by switching to dF/F traces."

## 2-b. How is the `neural` data processed?

i. The dF/F traces are extracted from the SDK's `dff_traces` DataFrame. The `dff` column contains per-cell trace arrays. These are stacked into a (n_neurons, n_timepoints) matrix. No additional processing (smoothing, normalization, z-scoring) is applied. The data is cast to float32.

ii.
```python
def extract_neural_matrix(signal_df, signal_kind):
    # ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)

# In process_experiment():
trial_neural = neural[:, m].astype(np.float32)
```

iii. The agent's trajectory and CONVERSION_NOTES.md indicate the dF/F data is used as-is from the SDK, which already provides preprocessed dF/F values. No further normalization or filtering is performed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level quality filtering is applied. The code does not check the `valid_roi` column in the cell specimen table. All neurons present in the dF/F traces are included. The minimum trace length across neurons is used to truncate all traces to equal length.

ii.
```python
def extract_neural_matrix(signal_df, signal_kind):
    # ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)
```

iii. The agent mentioned neuron curation in CONVERSION_NOTES.md Steps 1 and 3 (noting valid_roi metadata and ROI exclusion from the whitepaper) but did not implement explicit filtering. The trajectory does not show any explicit reasoning about skipping valid_roi filtering. For the available data, all 12 neurons in a sample experiment had `valid_roi=True`, so this may not have impacted results.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, the code selects ophys timestamps within `[trial_start_time, trial_stop_time)` and extracts the corresponding neural data columns. This means neural data is aligned to the start of each behaviorally-defined trial.

ii.
```python
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
# ...
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The instructions specify "Temporally align based on ophys timestamp." The agent documented this in CONVERSION_NOTES.md Step 5: "Align everything on ophys timestamps" and in the metadata: `temporal_alignment_event: 'ophys timestamps within each behaviorally defined trial'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native ophys frame rate (~93.2 ms per frame, approximately 10.7 Hz). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval across sessions.

ii.
```python
# In process_experiment():
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),

# In main():
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
```

iii. The agent documented that the native ophys timestamps are used without rebinning. CONVERSION_NOTES.md Step 3 notes: "neural data time bin: native ophys timestamps."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name` and the associated timing information (`start_time`). The stop time is computed by `ensure_stim_stop_time()` which uses the next image's start time rather than the actual `end_time` column in the data.

ii.
```python
def ensure_stim_stop_time(stim, use_full_interval=True):
    stim = stim.sort_values('start_time').copy()
    starts = stim['start_time'].to_numpy(dtype=float)
    # ...
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim
```

iii. The agent's trajectory (Step 23) documents that after initial issues with too many "unknown" image bins, the approach was changed to assign image labels over full image-presentation intervals (the full ~750ms between flashes) rather than only the ~250ms flash duration.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each unique image name is mapped to a categorical index. A "blank" category (index 0) is added for non-image periods. Omitted stimulus presentations are excluded. The `interval_assign` function assigns the image index to each ophys timestamp that falls within a stimulus presentation interval. Timestamps not covered by any presentation get the "blank" default.

ii.
```python
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}

# In build_trial_output():
stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values,
                                  stim_non_omitted['stop_time'].values, image_vals, default=blank_idx)
```

iii. The agent noted this decision was made to handle grey/omitted periods explicitly. The CONVERSION_NOTES.md Step 7 discusses fixing the image identity representation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned to each ophys timestamp within a trial using `interval_assign`, which checks whether each timestamp falls within the `[start_time, stop_time)` interval of a non-omitted stimulus presentation. This produces a time-varying categorical vector aligned to the neural data's time axis.

ii.
```python
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. The agent documented this alignment in CONVERSION_NOTES.md Step 5: all streams are assigned onto the ophys timestamp grid within each trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column and `start_time` in `stimulus_presentations`.

ii.
```python
# In build_trial_output():
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The agent used the SDK-provided `is_change` flag rather than computing changes from image identity transitions, which is consistent with the reference code's stimulus presentation table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary vector of zeros is created with length matching the trial's ophys timestamps. For each stimulus presentation marked as `is_change=True`, the ophys timestamp nearest to (at or after) the change onset is set to 1.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The agent's trajectory documents this as a time-varying binary output matching the instructions: "Have value of 1 right after a change in image identity, otherwise 0."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed. The output values are labeled `['no_change', 'change']`.

ii.
```python
'output_values': [
    image_names,
    ['no_change', 'change'],
    ...
]
```

iii. This directly follows the instructions specifying image change as a "binary variable."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change event is aligned by finding the ophys timestamp index closest to (at or after) the change onset time using `np.searchsorted`. Only a single timepoint per change event is marked as 1.

ii.
```python
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. The agent designed this to mark the change "right after" the change occurs, consistent with the instruction wording.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, which provides a DataFrame with `timestamps` and `speed` columns.

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    cols = {c.lower(): c for c in run_df.columns}
    tcol = cols.get('timestamps', 'timestamps')
    scol = cols.get('speed', None)
    if scol is None:
        for c in run_df.columns:
            if 'speed' in c.lower():
                scol = c
                break
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The agent documented in CONVERSION_NOTES.md Step 5 that running speed comes from `BehaviorOphysExperiment.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are assigned to ophys timestamps using nearest-neighbor interpolation (`nearest_assign`). The raw speed values are then digitized into 5 bins using globally-computed percentile edges.

ii.
```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The agent documented in CONVERSION_NOTES.md Step 5: "Interpolate/assign onto ophys timestamps, then discretize globally into 5 equal-percentile bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Percentile bin edges are computed globally across all sessions using `make_percentile_bins`, which computes the 20th, 40th, 60th, and 80th percentiles (for 5 bins). Values are then digitized using `np.digitize` with these edges. Non-finite values are assigned to bin 0.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    edges = np.percentile(vals, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def digitize(values, edges):
    vals = np.asarray(values, dtype=float)
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out
```

iii. The instructions specify "discretized into five equal percentile bins." The agent computes global percentile edges and applies them consistently.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is assigned to ophys timestamps via nearest-neighbor interpolation. The `nearest_assign` function finds the running speed sample closest in time to each ophys timestamp.

ii.
```python
def nearest_assign(sample_times, source_times, source_values):
    source_times = np.asarray(source_times, dtype=float)
    source_values = np.asarray(source_values)
    idx = np.searchsorted(source_times, sample_times, side='left')
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(sample_times - source_times[prev_idx]) < np.abs(sample_times - source_times[idx])
    idx[choose_prev] = prev_idx[choose_prev]
    return source_values[idx]
```

iii. The agent documented this in CONVERSION_NOTES.md Step 5 as resampling onto the ophys timestamp grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, specifically the `pupil_diameter` column (identified via `pick_pupil_column`).

ii.
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
    # fallback search...
```

iii. The agent documented in CONVERSION_NOTES.md Step 5 that pupil diameter comes from `BehaviorOphysExperiment.eye_tracking`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Invalid (inf, NaN) pupil values are dropped. Valid pupil diameter values are assigned to ophys timestamps using nearest-neighbor interpolation. Values are then digitized into 5 percentile bins using globally-computed edges. If no valid pupil data exists, zeros are used.

ii.
```python
# In build_trial_output():
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
if len(eye_valid) == 0:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
else:
    pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
    pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The agent documented handling of invalid eye tracking data in the code with appropriate fallbacks.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: global percentile bin edges (20th, 40th, 60th, 80th percentiles) computed across all sessions, then applied via `np.digitize`. Non-finite values map to bin 0.

ii.
```python
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
# ... later ...
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. Same justification as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: nearest-neighbor assignment onto ophys timestamps using `nearest_assign`.

ii.
```python
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
```

iii. Consistent with the overall temporal alignment strategy documented in CONVERSION_NOTES.md.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` columns in the SDK's `trials` table.

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
    outcome = 0  # default to hit
```

iii. The agent documented in CONVERSION_NOTES.md Steps 1 and 5 that trial outcomes come from the SDK trial labels, consistent with the reference code's trial logic.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcome columns are checked in priority order (hit > miss > false_alarm > correct_reject). The first True value determines the outcome. If none are True, the default is 0 (hit). The outcome is replicated across all timepoints in the trial as a static per-trial variable.

ii.
```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
```

iii. The instructions specify trial outcome as "Static per-trial." The agent implements this by filling all timepoints with the same value, which is compatible with the decoder framework.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- Missing `stop_time` in stimulus presentations: computed from next start times via `ensure_stim_stop_time`
- Missing pupil data: falls back to zeros
- Missing pupil column: falls back to zeros
- NaN/inf values in eye tracking: dropped before interpolation
- NaN values in trial boolean columns: filled with False via `.fillna(False)`
- Experiments with fewer than 2 valid trials: skipped
- Trials with fewer than 2 ophys timepoints: skipped
- Non-finite running/pupil values in percentile computation: excluded
- Default trial outcome when no boolean is True: defaults to hit (0)

ii.
```python
# Missing pupil data
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)

# NaN handling in trial filtering
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)

# Non-finite handling in digitize
out[~np.isfinite(vals)] = 0

# Default trial outcome
else:
    outcome = 0
```

iii. The agent documented handling of edge cases in CONVERSION_NOTES.md Step 10 and trajectory steps discussing invalid image labels and sparse neural data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()` (~4-20 seconds per experiment)
2. The double loading of all experiments: once in `collect_global_info()` and again in `process_experiment()`
3. The `interval_assign` function which loops over stimulus presentations per trial

ii.
```python
# Each experiment is loaded twice:
# First pass in collect_global_info():
exp = load_experiment(exp_id)
# Second pass in process_experiment():
exp = load_experiment(int(exp_id))
```

iii. The agent noted in CONVERSION_NOTES.md Step 6: "Repeated experiment loading during global percentile/image collection may be slow for full conversion." The trajectory (Step 37) confirms: "The main inefficiency is still the first pass loading every experiment to collect global image names and percentile bins, then loading all experiments again to process them."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two key loops could be vectorized:
1. `interval_assign` loops over each stimulus presentation with a Python for-loop rather than using vectorized interval matching
2. The image change detection loops over change presentations with a Python for-loop
3. The `build_trial_output` function is called per-trial in a for-loop over `trials.iterrows()`

ii.
```python
# interval_assign loop:
for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v

# image change loop:
for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
    idx = np.searchsorted(sample_times, ct, side='left')
    if idx < len(image_change):
        image_change[idx] = 1
```

iii. The agent noted in CONVERSION_NOTES.md Step 6: "Interval assignment currently loops over stimulus presentations per trial." The agent acknowledged inefficiencies but did not fully address them.

## 9-c. What processing does the code repeat multiple times?

i. The most significant repetition is loading each experiment twice:
1. First in `collect_global_info()` to gather image names, running speed values, and pupil values for global percentile computation
2. Then in `process_experiment()` to actually extract and convert the data

Additionally, `ensure_stim_stop_time` is called both in `collect_global_info()` and in `process_experiment()` for each experiment.

ii.
```python
# In collect_global_info():
exp = load_experiment(exp_id)
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
run_df = get_running_df(exp)
eye_df = get_eye_df(exp)

# In process_experiment():
exp = load_experiment(int(exp_id))
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
run_df = get_running_df(exp)
eye_df = get_eye_df(exp)
```

iii. The agent recognized this in trajectory Step 37 but did not refactor to a single-pass approach.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several unnecessary operations:
1. The `choose_signal` function loads both `dff_traces` and potentially `events`, but only one is used
2. Global image collection includes all images across all sessions, but images from excluded sessions are never used
3. The `ensure_stim_stop_time` function computes stop times even when the data already has `end_time` and `duration` columns
4. The `trial_input` is set to a zero-row array `np.zeros((0, n_timepoints))` for every trial since no inputs are needed, yet this array is created and stored for each trial
5. Running speed and pupil data for the full session is loaded and processed even though only trial-specific windows are needed

ii.
```python
# Unnecessary signal loading:
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()  # only loaded if dff fails
    return 'events', events

# Unnecessary input creation:
trial_input = np.zeros((0, len(sample_times)), dtype=np.float32)
```

iii. The agent noted efficiency concerns in CONVERSION_NOTES.md Step 6 but focused on correctness over optimization.
