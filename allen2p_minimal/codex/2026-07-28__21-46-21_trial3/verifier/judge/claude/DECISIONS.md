# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_local_cache()` to get the experiment table, then filters to only experiments with NWB files physically present on disk by scanning the `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` directory. Each experiment is loaded individually using `BehaviorOphysExperiment.from_nwb_path()`. Passive sessions are explicitly filtered out (`~experiments["passive"]`).

ii.
```python
def available_experiment_table(data_dir: Path) -> pd.DataFrame:
    nwb_root = data_dir / NWB_DIRNAME
    local_paths = {}
    for path in sorted(nwb_root.glob("behavior_ophys_experiment_*.nwb")):
        match = re.search(r"(\d+)\.nwb$", path.name)
        if match:
            local_paths[int(match.group(1))] = path
    cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
    experiments = cache.get_ophys_experiment_table()
    experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
    ...

experiments = experiments.loc[~experiments["passive"]].copy()

dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. The agent noted: "The local cache is incomplete: the project table lists release metadata for more experiments than there are NWB files on disk. That's a real constraint." It chose to scan for available NWB files and intersect with the experiment table. Passive sessions were filtered because the whitepaper distinguishes active change-detection sessions from passive viewing.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment table, stored as strings.

ii.
```python
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. Subjects are registered as they are encountered during iteration over experiments.

## 1-c. How are the data split into sessions?

i. Each decoder "session" corresponds to one Allen `BehaviorOphysExperiment` NWB file (one imaging plane), NOT one unique behavior session. This means multiscope sessions with multiple imaging planes produce multiple decoder sessions. The AI iterates over experiments in the experiment table, not grouping by `ophys_session_id`.

ii.
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    neural_sessions.append(neural_trials)
```

iii. The agent stated: "One detail I'm calling out in the notes is that each decoder 'session' corresponds to one Allen BehaviorOphysExperiment NWB file, which is the right unit for multiscope data because a single behavior session can contain multiple imaging planes." The result was 199 converted sessions from 171 unique `ophys_session_id` values.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `dataset.trials` table. Only `go` or `catch` trials are kept; `aborted` and `auto_rewarded` trials are excluded. Trial windows are from `start_time` to `stop_time`. The AI then creates a common time grid within each trial at 100ms resolution.

ii.
```python
trials = dataset.trials.copy()
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
...
for trial_id, trial_row in trials.iterrows():
    start_time = float(trial_row["start_time"])
    stop_time = float(trial_row["stop_time"])
    target_times = make_target_times(start_time, stop_time, bin_size_s)
```

iii. The agent confirmed: "The trial table confirms the key filter logic: `go` and `catch` are the behavior-defined trial types, while aborted and auto-rewarded trials are explicitly labeled and can be removed."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch (not aborted, not auto-rewarded), (2) must have at least 2 time bins in the trial window, (3) running and pupil interpolation must succeed, (4) trial outcome must be classifiable (hit/miss/false_alarm/correct_reject). Sessions with fewer than 2 valid trials are excluded. Additionally, sessions are excluded if they have too few ophys timestamps, no event traces, no running signal, or no valid pupil signal.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
...
if target_times.size < 2:
    continue
if running_trial is None or pupil_trial is None:
    continue
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
if len(neural_trials) < 2:
    ...continue
```

iii. The agent noted 3 sessions were dropped for missing pupil data and described the approach as robust to partial data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events` — specifically `events_df["events"]`, the AllenSDK discrete calcium events. This is NOT dF/F traces.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append(...)
    continue
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. The agent chose calcium events over dF/F because "the paper's statement that neural analyses were performed on detected calcium events rather than raw fluorescence." From CONVERSION_NOTES: "The converter intentionally does not use dF/F traces and does not use filtered_events."

## 2-b. How is the `neural` data processed?

i. The events traces are stacked into a `(n_neurons, n_timepoints)` matrix. For each trial, the neural data is resampled to a common 100ms time grid using nearest-neighbor indexing into ophys timestamps. No additional smoothing or normalization is applied.

ii.
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The agent did not apply additional processing beyond what the AllenSDK provides. The nearest-neighbor resampling was chosen for the discrete event signal to avoid interpolation artifacts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AllenSDK's default ROI validity filtering is relied upon (invalid ROIs are excluded at load time). No additional neural quality filtering is applied. Sessions with no valid event traces (empty `events` DataFrame) are skipped.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    continue
```

iii. The agent confirmed: "I've confirmed the SDK excludes invalid ROIs when loading experiments, which is the right baseline for the quality-control requirement." All-zero neural trials were kept as plausible for sparse event representations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start time (`start_time` from the trials table). A common time grid is created from `start_time` to `stop_time` at 100ms resolution, and neural data is sampled at these points using nearest-neighbor indexing into ophys timestamps.

ii.
```python
start_time = float(trial_row["start_time"])
stop_time = float(trial_row["stop_time"])
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The metadata records `temporal_alignment_event: "trial start time from AllenSDK trials table"` and `off_start: 0.0, off_end: None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins data to a common 100ms time bin size. This is applied uniformly across all sessions. The native ophys frame rate varies (32ms for single-plane, 93ms for multiscope).

ii.
```python
TIME_BIN_MS_DEFAULT = 100.0
...
def make_target_times(start_time: float, stop_time: float, bin_size_s: float) -> np.ndarray:
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]
```

iii. The agent explained: "The local dataset mixes single-plane and multiscope experiments. Their native frame intervals differ... Using 100 ms bins preserves compatibility across both acquisition modes while staying close to the slower multiscope sampling interval."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, filtered to the `change_detection` stimulus block. For each trial, stimulus rows are matched via `trials_id`. The `image_name`, `start_time`, and `end_time` fields from stimulus presentations are used. A special "gray" label is used for time bins outside stimulus flash periods.

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains("change_detection", na=False)
].copy()
...
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
    image_idx = image_to_idx[stim_row.image_name]
    mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
    image_labels[mask] = image_idx
```

iii. The agent stated: "each trial is already linked to the sequence of flashed image intervals via `trials_id`, and the changed image row is marked with `is_change`."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built incrementally as new image names are encountered. The default label is "gray" (index 0) for time bins outside stimulus flash periods. Omitted stimuli are skipped (remain gray).

ii.
```python
image_values = [GRAY_LABEL]
image_to_idx = {GRAY_LABEL: 0}
...
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
...
if stim_row.image_name not in image_to_idx:
    image_to_idx[stim_row.image_name] = len(image_values)
    image_values.append(stim_row.image_name)
image_idx = image_to_idx[stim_row.image_name]
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
```

iii. A global mapping ensures consistent integer codes across sessions. The gray label handles inter-stimulus intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each target time point (100ms bins) within the trial, using the same `target_times` array used for neural data alignment. For each stimulus presentation, a mask identifies which target times fall within the stimulus window.

ii.
```python
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
```

iii. Alignment is ensured by computing all variables (neural, image identity, running, pupil) on the same `target_times` grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is a binary time-varying variable derived from the `is_change` field of `stimulus_presentations` filtered to the `change_detection` block. It is 1 during stimulus epochs where `is_change == True`, 0 otherwise.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The agent used `is_change` from the stimulus presentations table to identify change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. No processing beyond computing the binary indicator from the `is_change` flag and the stimulus epoch timing (`start_time` to `end_time`).

ii. See 4-a.

iii. N/A

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). The 1 label is active only during the stimulus presentation epoch where `is_change` is True (i.e., during the 250ms flash of the changed image).

ii.
```python
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The duration of the change signal corresponds to the stimulus presentation duration rather than a fixed window.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same approach as image identity — computed on the same `target_times` grid as neural data, with the change label mask based on stimulus start/end times.

ii. See 4-a.

iii. Same frame-level alignment as all other variables via the common time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
running_df = dataset.running_speed
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 100ms target time grid using `np.interp`. Invalid (NaN/Inf) values are excluded before interpolation. Edge values are used for extrapolation. The interpolated values are then discretized into 5 global quintile bins using percentiles [20, 40, 60, 80].

ii.
```python
def interp_signal(source_times, source_values, target_times):
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    ...
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)

running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. Linear interpolation preserves signal shape. Quintile-based binning ensures roughly equal class counts. Bin edges are computed globally across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.digitize` with edges at global quintile percentiles [20, 40, 60, 80]. This produces bins 0-4.

ii.
```python
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
def remap_to_bins(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int16)
```

iii. Global quintile edges ensure balanced bin counts across the entire dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same `target_times` grid as the neural data, ensuring temporal alignment.

ii.
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. By computing all variables on the same common time grid, alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_area"]`. The equivalent diameter is computed as `2 * sqrt(pupil_area / pi)`. Blink frames are NOT explicitly filtered before interpolation.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. The agent derived diameter from pupil_area assuming a circular pupil, rather than using the directly available `pupil_width` field.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to equivalent diameter, then linearly interpolated to the 100ms target time grid. Invalid values (NaN/Inf, which include blink artifacts from the area computation) are excluded during interpolation. The interpolated values are then discretized into 5 global quintile bins.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
...
pupil_trial = interp_signal(eye_df["timestamps"].to_numpy(dtype=np.float64), pupil_diameter, target_times)
...
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. The `interp_signal` function filters out NaN/Inf values (which includes NaN pupil_area during blinks), so blinks are implicitly handled but not via the SDK's `likely_blink` flag.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 bins using `np.digitize` with global quintile edges at percentiles [20, 40, 60, 80].

ii.
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. Global quintile edges ensure balanced bin counts.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed — pupil diameter is interpolated to the same `target_times` grid as the neural data.

ii.
```python
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. Same common time grid alignment as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def outcome_to_index(trial_row: pd.Series) -> int | None:
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

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task. Trials where none match are skipped (return None).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The outcome is static per trial and broadcast to fill all time bins.

ii.
```python
def build_output_array(..., outcome_idx: int) -> np.ndarray:
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
    return np.vstack([image_labels, change_labels, running_bins, pupil_bins, outcome])
```

iii. The mapping is consistent with the output_values ordering: `["hit", "miss", "false_alarm", "correct_reject"]`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loads**: If loading an NWB file throws an exception, the session is skipped.
- **Insufficient timestamps**: Sessions with < 2 ophys timestamps are skipped.
- **No event traces**: Sessions with empty events DataFrame are skipped.
- **Missing running signal**: Sessions where running interpolation returns None are skipped.
- **Missing pupil data**: Sessions with no finite pupil values are skipped (3 sessions excluded).
- **Short trials**: Trials with < 2 time bins are skipped.
- **Missing behavioral signals per trial**: Trials where running or pupil interpolation returns None are skipped.
- **Unclassifiable outcomes**: Trials where none of hit/miss/false_alarm/correct_reject is True are skipped.
- **Few trials**: Sessions with fewer than 2 valid trials are excluded.
- **Interpolation edge values**: `np.interp` extrapolates using first/last valid values rather than NaN.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
except Exception as exc:
    ...continue
if ophys_timestamps.size < 2: ...continue
if len(events_df) == 0: ...continue
if running_trial is None or pupil_trial is None: continue
if outcome_idx is None: continue
if len(neural_trials) < 2: ...continue
```

iii. The agent designed the pipeline to be robust, skipping problematic data at multiple levels (session, trial) rather than crashing.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `BehaviorOphysExperiment.from_nwb_path()`, which reads large NWB files from disk containing neural traces, behavioral data, and metadata.

ii. N/A

iii. Each NWB file contains full-session neural traces for all neurons plus behavioral data. This is I/O bound.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop iterates over each valid trial sequentially, performing interpolation and nearest-neighbor lookups separately for each trial. The stimulus presentation matching (inner loop per trial) could also be vectorized.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
    for stim_row in trial_stim.itertuples():
        ...
```

iii. The per-trial loop is not the bottleneck compared to data loading. Running and pupil interpolation are called per-trial rather than once per session, which is redundant work.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil diameter interpolation are repeated for every trial individually, even though the underlying signal is the same across the entire session. The code calls `interp_signal()` once per trial with per-trial `target_times`, rather than interpolating once to the full ophys timebase and then slicing per trial.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
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

iii. This repeats the sorting and NaN-filtering logic of `interp_signal` for every trial. It would be more efficient to interpolate once per session.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed `session_info` and `excluded_sessions` metadata that is not used by the downstream decoder. The `native_ophys_frame_interval_ms_summary` statistics are computed but not used for any processing decisions.

ii.
```python
session_info.append({
    "experiment_id": int(experiment_id),
    "ophys_session_id": int(row["ophys_session_id"]),
    ...
})
excluded_sessions.append(...)
native_dt_ms.append(float(np.median(np.diff(ophys_timestamps)) * 1000.0))
```

iii. This metadata is useful for debugging but is not used by the decoder.
