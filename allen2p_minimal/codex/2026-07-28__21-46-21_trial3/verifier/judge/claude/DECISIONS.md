# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the local NWB files directory for experiment files, builds an experiment table using `VisualBehaviorOphysProjectCache.from_local_cache()`, and intersects with locally available NWB files. Each experiment is loaded individually via `BehaviorOphysExperiment.from_nwb_path()`. Only experiments with `passive == False` are kept (active sessions). Both `VisualBehavior` and `VisualBehaviorMultiscope` project codes are included.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
experiments = cache.get_ophys_experiment_table()
experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
# ...
experiments = experiments.loc[~experiments["passive"]].copy()
# ...
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. The AI loads from local NWB files rather than S3 cache. It includes both VisualBehavior and VisualBehaviorMultiscope project codes, filtering only on `passive == False`.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values encountered during processing, stored as strings.

ii.
```python
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

iii. Each unique mouse_id encountered is registered as a subject. The AI found 38 mice in its local data subset.

## 1-c. How are the data split into sessions?

i. Each individual experiment (one imaging plane, one NWB file) is treated as a separate "session" in the output. The AI does NOT merge multiple imaging planes from the same `ophys_session_id` into a single session.

ii.
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    # Each experiment becomes its own session
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    # ...
    neural_sessions.append(neural_trials)
```

iii. From CONVERSION_NOTES.md: "Each decoder 'session' is one Allen `BehaviorOphysExperiment` NWB file, not one unique behavior session. This matches the AllenSDK data model and avoids incorrectly merging different imaging planes from multiscope sessions into one neuron matrix."

## 1-d. How are the data split into trials?

i. Trials are defined using the `dataset.trials` table. Only trials where `go == True` or `catch == True` are kept, excluding `aborted` and `auto_rewarded` trials. Trial windows span `start_time` to `stop_time`, rebinned into 100ms bins.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
# ...
target_times = make_target_times(start_time, stop_time, bin_size_s)
```

iii. The AI uses the SDK's trials table with go/catch filtering, matching the instructions. The AI does not additionally require `change_time` to be non-null (unlike the reference).

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials with fewer than 2 time bins are skipped. Trials where running or pupil interpolation returns None are skipped. Trials where `outcome_to_index` returns None (no valid outcome) are skipped. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
# ...
if target_times.size < 2:
    continue
if running_trial is None or pupil_trial is None:
    continue
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
# ...
if len(neural_trials) < 2:
    # skip session
```

iii. Multiple quality gates ensure only well-formed trials enter the dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events["events"]` — the discrete calcium events detected by the Allen SDK, NOT from `dff_traces`.

ii.
```python
events_df = dataset.events
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Neural activity uses AllenSDK discrete calcium events from `dataset.events['events']`. This matches the paper's statement that neural analyses were performed on detected calcium events rather than raw fluorescence."

## 2-b. How is the `neural` data processed?

i. The events are stacked into a (n_neurons, T) matrix. For each trial, neural data is resampled to 100ms time bins by finding the nearest ophys timestamp to each bin center.

ii.
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
# ...
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The AI uses nearest-neighbor resampling rather than averaging within bins. The 100ms bin size was chosen to accommodate both single-plane (~93ms native) and multiscope (~32ms native) frame rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering beyond the SDK's default ROI validity filtering. Experiments with no valid event traces (empty `events` DataFrame) are skipped.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    continue
```

iii. From CONVERSION_NOTES.md: "AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically when loading each BehaviorOphysExperiment."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `trial.start_time`. Within each trial, a grid of 100ms-spaced time bin centers is created from `start_time` to `stop_time`. Neural data at each bin center is taken from the nearest ophys timestamp.

ii.
```python
start_time = float(trial_row["start_time"])
stop_time = float(trial_row["stop_time"])
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The alignment is to trial start, consistent with the instructions ("temporally align based on ophys timestamp").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning to a fixed 100ms bin size across all sessions. This is NOT the native ophys frame rate.

ii.
```python
TIME_BIN_MS_DEFAULT = 100.0
# ...
def make_target_times(start_time, stop_time, bin_size_s):
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]
```

iii. From CONVERSION_NOTES.md: "Using 100 ms bins preserves compatibility across both acquisition modes while staying close to the slower multiscope sampling interval."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, filtered to rows where `stimulus_block_name` contains "change_detection". The `image_name` column from each stimulus presentation is used, along with `start_time` and `end_time` to determine when each image is on screen. Non-image periods are labeled as "gray".

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains("change_detection", na=False)
].copy()
# ...
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
    image_idx = image_to_idx[stim_row.image_name]
    mask = (target_times >= float(stim_row.start_time)) & (
        target_times < float(stim_row.end_time)
    )
    image_labels[mask] = image_idx
```

iii. The AI uses stimulus_presentations for fine-grained per-flash image identity, including explicit "gray" labeling for inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a dynamic mapping. A "gray" label (index 0) is used for all non-image times. Omitted stimuli remain gray. The mapping is built incrementally as new image names are encountered.

ii.
```python
image_values = [GRAY_LABEL]
image_to_idx = {GRAY_LABEL: 0}
# ...
if stim_row.image_name not in image_to_idx:
    image_to_idx[stim_row.image_name] = len(image_values)
    image_values.append(stim_row.image_name)
image_idx = image_to_idx[stim_row.image_name]
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
```

iii. The AI includes "gray" as a distinct category in image identity, whereas the reference derives image identity from `initial_image_name` and `change_image_name` in the trials table (no gray label).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels are computed at the same 100ms bin centers as the neural data using `target_times`. Each bin is labeled based on whether it falls within a stimulus presentation epoch.

ii.
```python
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
# ...
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
```

iii. Alignment is guaranteed because both neural and image labels use the same target_times grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `stimulus_presentations`, filtered to the change_detection block.

ii.
```python
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The AI uses `is_change` from stimulus_presentations rather than `go` from the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary label is created: 1 during stimulus flash epochs where `is_change == True`, 0 everywhere else.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
# ...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The change label spans only the stimulus flash duration (250ms image + gray until next flash), not a fixed 750ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed.

ii. N/A — binary variable.

iii. N/A.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity — computed at the same 100ms bin centers using target_times and stimulus presentation start/end times.

ii. See 3-c code above.

iii. Same target_times grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
running_df = dataset.running_speed
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. Same source as the reference.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the 100ms target time bins, then discretized into 5 equal-percentile bins using global percentile edges computed across all sessions.

ii.
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
# ...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. Percentile edges are [20, 40, 60, 80], giving 4 edges and 5 bins. NaN/infinite values are filtered during interpolation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized using `np.digitize(values, edges, right=False)` with 4 percentile edges at [20, 40, 60, 80].

ii.
```python
def remap_to_bins(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int16)
```

iii. This produces bins 0-4 (5 bins).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same `target_times` grid used for neural data, ensuring alignment.

ii.
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. Same target_times grid as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_area"]`, converted to equivalent diameter via `2 * sqrt(area / pi)`.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. From CONVERSION_NOTES.md: "Pupil diameter is computed as equivalent diameter from AllenSDK `pupil_area`: `diameter = 2 * sqrt(area / pi)`". The reference uses `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter, then linearly interpolated to the 100ms target time bins. NaN/infinite values are filtered during interpolation. Discretized into 5 global percentile bins.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi)
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
# ...
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. Unlike the reference, the AI does NOT explicitly remove blink frames before interpolation. However, the `interp_signal` function filters out non-finite values (which would include NaN from blinks).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: `np.digitize(values, edges, right=False)` with percentile edges at [20, 40, 60, 80].

ii. Same `remap_to_bins` function as running speed.

iii. Produces 5 bins (0-4).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same `target_times` grid as neural data.

ii.
```python
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. Same target_times grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def outcome_to_index(trial_row):
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
    return None
```

iii. Same source variables as the reference. Trials where none of these is True return None and are excluded.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The outcome is constant across all time bins within a trial. Trials with no valid outcome are excluded.

ii.
```python
outcome = np.full(t, outcome_idx, dtype=np.int16)
```

iii. Same mapping order as the reference.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loads**: Experiments that fail to load are skipped with logging.
- **Missing ophys timestamps**: Experiments with fewer than 2 ophys timestamps are skipped.
- **No events**: Experiments with empty events DataFrame are skipped.
- **Missing running/pupil**: Trials where interpolation returns None are skipped.
- **Invalid pupil data**: Experiments with no finite pupil values are skipped.
- **Too few trials**: Sessions with fewer than 2 valid trials are excluded.
- **NaN handling**: `interp_signal` filters out non-finite values before interpolation; extrapolation uses edge values rather than NaN.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
except Exception as exc:
    excluded_sessions.append(...)
    continue
# ...
if ophys_timestamps.size < 2:
    excluded_sessions.append(...)
    continue
# ...
if running_trial is None or pupil_trial is None:
    continue
```

iii. The AI's `interp_signal` uses `np.interp` with `left=values[0], right=values[-1]` for extrapolation (constant edge value), whereas the reference uses `fill_value=np.nan` (NaN outside bounds, then mapped to bin 0).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `BehaviorOphysExperiment.from_nwb_path()`, which reads large NWB files from disk. This is I/O bound.

ii. N/A

iii. Each NWB file contains full-session neural traces, behavioral data, and metadata.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within each experiment iterates over trials sequentially, performing interpolation and nearest-neighbor lookups for each trial individually. These could potentially be vectorized by computing all trial boundaries at once.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    # ... interpolation and nearest-neighbor for each trial
```

iii. The per-trial loop is not the bottleneck compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil diameter interpolation are performed per-trial rather than once per session. `interp_signal` is called for each trial separately, recomputing the interpolation each time.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    running_trial = interp_signal(
        running_df["timestamps"].to_numpy(dtype=np.float64),
        running_df["speed"].to_numpy(dtype=np.float64),
        target_times,
    )
    pupil_trial = interp_signal(
        eye_df["timestamps"].to_numpy(dtype=np.float64),
        pupil_diameter,
        target_times,
    )
```

iii. The reference interpolates running and pupil to the full ophys timebase once per session, then slices per trial. The AI re-interpolates per trial.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `excluded_sessions` metadata and detailed `session_info` for every session, which are stored in metadata but not directly used by the decoder. The `native_ophys_frame_interval_ms_summary` is also computed but not used downstream.

ii.
```python
"excluded_sessions": excluded_sessions,
"native_ophys_frame_interval_ms_summary": {...},
"session_info": session_info,
```

iii. These are informational and add negligible overhead.
