# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading a local CSV experiment table (`ophys_experiment_table.csv`), filtering out passive sessions, then loading each experiment's NWB file individually using `BehaviorOphysExperiment.from_nwb_path()`. It performs two passes: first in `collect_global_info()` to gather image names, running speed values, and pupil values for global percentile bin computation, and second in `process_experiment()` to extract neural and behavioral data per experiment.

ii.
```python
EXPT_TABLE = META_DIR / 'ophys_experiment_table.csv'

def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

# In main():
exp_table = load_experiment_table()
exp_table = exp_table[~exp_table['session_type'].astype(str).str.contains('passive', case=False, na=False)].copy()
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
```

iii. The AI chose to load NWB files directly rather than using the S3 cache, noting in CONVERSION_NOTES.md that it "avoided broken local-cache manifest path." Passive sessions are excluded because trial outcomes are not meaningful during passive viewing. The two-pass approach ensures global discretization bins are computed before per-experiment processing.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values extracted from experiment metadata rows. They are collected as a sorted set from all processed sessions.

ii.
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}

# Per experiment:
subject = str(meta_row.get('mouse_id', 'unknown'))
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. Each **ophys experiment** (one imaging plane) is treated as a separate session. This differs from the reference approach of grouping experiments by `ophys_session_id` to combine multiple imaging planes into a single session.

ii.
```python
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The AI documented this decision in CONVERSION_NOTES.md Step 4: "Use **ophys experiment** as one converted session because each experiment has one coherent neuron set and one brain region/depth assignment; preserve subject mapping across experiments." This results in more sessions with fewer neurons each compared to the reference approach.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `exp.trials` table. Go and catch trials are included; aborted and auto-rewarded trials are excluded. The trial window spans from `start_time` to `stop_time` using boolean masking on ophys timestamps (`>= start` and `< stop`). Trials with fewer than 2 ophys frames are skipped.

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

iii. The filtering logic matches the SDK's trial type definitions. Using `go | catch` is logically equivalent to `not aborted and not auto_rewarded` given the SDK's trial type hierarchy.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Passive sessions are excluded entirely. Trials with fewer than 2 ophys frames are skipped. Sessions with fewer than 2 retained trials are excluded. No additional filtering on `change_time` validity is applied (unlike the reference which requires `change_time.notna()`).

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

iii. The AI decided to exclude passive sessions because trial outcomes are not behaviorally meaningful during passive viewing. The minimum frame count and trial count thresholds prevent degenerate data from entering the dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment. The AI initially considered using `events` but switched to dF/F after finding sparse/all-zero trials with events.

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
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)
```

iii. CONVERSION_NOTES.md documents: "Events-based neural representation produced sparse/all-zero trial warning and poor sample decoding: resolved by switching to dF/F traces."

## 2-b. How is the `neural` data processed?

i. The dF/F trace arrays are extracted from the signal DataFrame, converted to float32, and stacked into a `(n_neurons, n_timepoints)` matrix. If trace lengths differ across neurons, they are truncated to the minimum length. Since each experiment is one imaging plane, there is no cross-plane merging (unlike the reference which stacks across planes).

ii.
```python
def extract_neural_matrix(signal_df, signal_kind):
    # ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)
```

iii. No additional normalization or filtering is applied. The dF/F is pre-computed by the Allen SDK pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to the neural data. All neurons present in the SDK's `dff_traces` are included, consistent with the reference approach.

ii. N/A (no filtering code)

iii. The Allen SDK pipeline already applies quality control (cell segmentation, neuropil correction). The AI did not add further filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is extracted by boolean-masking ophys timestamps within `[start_time, stop_time)`. This aligns to the trial start time, giving variable-length windows per trial.

ii.
```python
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
# ...
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
```

iii. The alignment uses the ophys timestamps as the temporal reference, consistent with the instruction to "temporally align based on ophys timestamp."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native ophys frame rate (~11 Hz, ~93 ms per frame). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval.

ii.
```python
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
# where dt_ms per session:
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
```

iii. The ophys timestamps are already at a consistent frame rate. No resampling is needed since all data streams are aligned to the ophys timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`, using the stimulus presentation table to determine which image is on screen at each ophys timepoint. This differs from the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
stim = ensure_stim_stop_time(stim_df)
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
else:
    stim_non_omitted = stim

image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values, stim_non_omitted['stop_time'].values, image_vals, default=blank_idx)
```

iii. The AI used stimulus_presentations to get frame-accurate image identity, and constructed stop_times by taking the next presentation's start_time, thus covering the full interval including grey screens. A `blank` image category was added for timepoints not covered by any stimulus presentation.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping that includes a `blank` category (index 0) plus sorted unique image names from all sessions. Omitted stimuli are excluded before assignment. The `interval_assign` function maps each ophys timepoint to the image that was being presented at that time, with `blank` as default.

ii.
```python
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
# ...
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. The `blank` category was added after initial conversion produced negative image labels that caused decoder training failure. The AI extended stimulus intervals to cover inter-stimulus grey periods via `ensure_stim_stop_time`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys timestamp within the trial window, using the same `sample_times` array as the neural data. The `interval_assign` function maps each ophys timepoint to the corresponding stimulus presentation interval.

ii.
```python
image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values,
                                  stim_non_omitted['stop_time'].values, image_vals, default=blank_idx)
```

iii. Both neural and image identity use the same ophys timestamps, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, not from the trials table `change_time`/`go` columns as in the reference.

ii.
```python
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The AI used the stimulus_presentations table's `is_change` flag to identify change events, which is a valid alternative source for this information.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array of zeros is created for each trial. For each stimulus presentation marked as `is_change`, a single ophys frame is set to 1 at the change onset time. This is a single-frame indicator, not a window.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. The instruction says "Have value of 1 right after a change in image identity, otherwise 0." The AI interpreted this as marking only the single frame at change onset, while the reference uses a 750ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1) by construction. No thresholding is needed.

ii. See 4-b above.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change uses the same `sample_times` (ophys timestamps) as neural data, with `np.searchsorted` to find the frame at or after change onset.

ii.
```python
idx = np.searchsorted(sample_times, ct, side='left')
```

iii. Same frame-level alignment as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using the `speed` column with associated `timestamps`.

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    # ... column name normalization ...
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are assigned to ophys timepoints using **nearest-neighbor assignment** (`nearest_assign`), then discretized into 5 percentile-based bins. This differs from the reference which uses linear interpolation (`interp1d`). The bin edges are computed globally from all running speed values across all experiments (including full-session data, not just trial-extracted data).

ii.
```python
def nearest_assign(sample_times, source_times, source_values):
    idx = np.searchsorted(source_times, sample_times, side='left')
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(sample_times - source_times[prev_idx]) < np.abs(sample_times - source_times[idx])
    idx[choose_prev] = prev_idx[choose_prev]
    return source_values[idx]

# Global bin edges:
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
```

iii. Nearest-neighbor was used instead of linear interpolation, which is a simpler approach but may introduce step artifacts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed from inner percentiles (20th, 40th, 60th, 80th). Non-finite values are mapped to bin 0.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    vals = vals[np.isfinite(vals)]
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

iii. The percentile computation uses only the inner percentiles (4 edges for 5 bins), which differs slightly from the reference approach (which computes 6 percentile points including 0 and 100, then uses `bin_edges[1:-1]` for digitize). The AI also handles edge cases where percentile edges are equal by nudging them with `np.nextafter`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is assigned to ophys timestamps using nearest-neighbor assignment within the trial window, using the same `sample_times` as neural data.

ii.
```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
```

iii. Uses the same temporal grid as neural data but with nearest-neighbor rather than interpolation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, with the column chosen dynamically by searching for `pupil_diameter`, `pupil_area`, `pupil_width`, or `pupil_radius` in that priority order.

ii.
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
    # fallback to any column with 'pupil' in name
```

iii. The AI used a priority-based column search, which differs from the reference that directly uses `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil values are assigned to ophys timepoints using nearest-neighbor assignment. Invalid values (NaN, inf) are dropped before assignment but there is no explicit blink removal using `likely_blink`. Discretization follows the same approach as running speed (5 percentile bins).

ii.
```python
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
if len(eye_valid) == 0:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
else:
    pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
    pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The AI drops NaN/inf values but does not explicitly filter blink frames using the `likely_blink` flag (unlike the reference). This means blink artifacts may contaminate the pupil signal via nearest-neighbor assignment.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins using inner percentiles. Non-finite values mapped to bin 0.

ii. Same as 5-c.

iii. Same as 5-c.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: nearest-neighbor assignment to ophys timestamps within the trial window.

ii.
```python
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
```

iii. Uses same temporal grid as neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table, consistent with the reference.

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
    outcome = 0  # fallback to hit
```

iii. These four columns are the SDK's canonical trial outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) via an if-elif chain. The fallback for unrecognized outcomes is 0 (hit), unlike the reference which uses -1 or 'other'. The outcome is replicated across all timepoints in the trial.

ii.
```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
# output stack:
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```

iii. The outcome code mapping matches the reference ordering. The fallback to 0 (hit) rather than a separate 'other' or -1 category is a concerning choice since it silently maps edge cases to a valid outcome.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: No explicit try/except in `process_experiment`; errors would crash the pipeline.
- **Truncated trials**: Trials with fewer than 2 frames are skipped.
- **Missing pupil data**: If no pupil column found, zeros are used. If all pupil values are NaN/inf, zeros used.
- **NaN running/pupil**: Non-finite values mapped to bin 0 during digitization.
- **Few trials**: Sessions with fewer than 2 trials are excluded.
- **Missing NWB files**: `collect_global_info` skips experiments whose NWB file doesn't exist.

ii.
```python
if not nwb_path.exists():
    continue
# ...
if m.sum() < 2:
    continue
# ...
out[~np.isfinite(vals)] = 0
# ...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The handling is generally reasonable but less robust than the reference's try/except approach, which protects against unexpected errors during session loading.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment's NWB file, which is done **twice**: once in `collect_global_info()` to compute global statistics, and once in `process_experiment()` for actual data extraction. This doubles the I/O cost.

ii.
```python
# First pass (collect_global_info):
exp = load_experiment(exp_id)

# Second pass (process_experiment):
exp = load_experiment(int(exp_id))
```

iii. Each NWB file contains full-session dF/F traces, running speed, eye tracking, and trials data. Loading each file twice is a significant inefficiency.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interval_assign` function loops over each stimulus presentation to assign image identity, which could be vectorized using `np.searchsorted` or `pd.IntervalIndex`. The trial outcome if-elif chain iterates sequentially but is trivially fast.

ii.
```python
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(...):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out
```

iii. The loop over stimulus presentations is O(n_presentations * n_timepoints) per trial, which could be reduced with sorted interval lookup.

## 9-c. What processing does the code repeat multiple times?

i. Each experiment is loaded twice: once in `collect_global_info()` and once in `process_experiment()`. The `ensure_stim_stop_time` function is also called multiple times for the same stimulus presentations (once in `collect_global_info` and once in `build_trial_output`).

ii.
```python
# In collect_global_info:
exp = load_experiment(exp_id)
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())

# In process_experiment -> build_trial_output:
exp = load_experiment(int(exp_id))
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
```

iii. Loading each experiment twice roughly doubles the total runtime compared to a single-pass approach.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `collect_global_info` function computes bin edges from **all running speed and pupil values** across full sessions, not just trial-extracted data. This includes data outside of trial windows that is not used in the final output. Additionally, `ensure_stim_stop_time` computes extended stop times in the global pass but these computed values are discarded and recomputed per trial.

ii.
```python
# Full session running speed collected:
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
# Full session pupil collected:
pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
```

iii. Computing bin edges from full-session data rather than trial-extracted data is slightly inconsistent with what the decoder will see, since it includes inter-trial data in the percentile computation.
