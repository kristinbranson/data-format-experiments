# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the Allen experiment table from a local cache, intersects it with NWB files physically present on disk, sorts those rows, filters out passive experiments, and then loads each surviving NWB file one at a time with `BehaviorOphysExperiment.from_nwb_path(...)`. It does not use the SDK S3 cache path from the reference solution.

ii. 
```python
cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
experiments = cache.get_ophys_experiment_table()
experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
...
experiments = experiments.loc[~experiments["passive"]].copy()
...
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. In `CONVERSION_NOTES.md`, the AI says the local cache is incomplete, so it intentionally restricts itself to NWB files present under `/app/data/.../behavior_ophys_experiments`. In the trajectory it also says it is using “available NWB files only” and “active task sessions” so the converter matches the local subset rather than the full paper cohort.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id`, converted to strings and stored in first-seen order as active experiment-level sessions are accepted.

ii. 
```python
subjects: list[str] = []
subject_to_idx: dict[str, int] = {}
...
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
...
subject_idx.append(subject_to_idx[subject])
```

iii. The code uses the Allen experiment-table `mouse_id` field as the animal identifier. `CONVERSION_NOTES.md` describes the final output in terms of numbers of “subjects,” matching this choice.

## 1-c. How are the data split into sessions?

i. Each kept NWB experiment file is treated as one decoder session. The AI does not group multiple experiments with the same `ophys_session_id` into a single session.

ii. 
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    neural_sessions.append(neural_trials)
    ...
    session_info.append(
        {
            "experiment_id": int(experiment_id),
            "ophys_session_id": int(row["ophys_session_id"]),
            ...
        }
    )
```

iii. `CONVERSION_NOTES.md` explicitly says each decoder “session” is one Allen `BehaviorOphysExperiment` NWB file and defends that choice as avoiding incorrect merging of multiscope imaging planes. The trajectory repeats that the dataset is “experiment-plane based rather than merged across behavior sessions.”

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. After filtering, each row is treated as one trial, and the trial window runs from `start_time` to `stop_time`. Within that window, the AI creates a new regular 100 ms time grid and samples all variables onto that grid.

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

iii. The notes say trials are “segmented using the AllenSDK `trials` table,” and the trajectory says the AI confirmed that `go` and `catch` are the behavior-defined trial types to keep.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are `go` or `catch`, not `aborted`, and not `auto_rewarded`. Trials are then skipped if the 100 ms grid has fewer than 2 bins, if running or pupil interpolation produces no valid signal, or if the trial has no recognized outcome. Entire experiment-level sessions are skipped if fewer than 2 trials survive.

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
...
if running_trial is None or pupil_trial is None:
    continue
...
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
if len(neural_trials) < 2:
    excluded_sessions.append(
        {"experiment_id": int(experiment_id), "reason": "fewer_than_two_kept_trials"}
    )
    continue
```

iii. `CONVERSION_NOTES.md` emphasizes the go/catch, non-aborted, non-auto-rewarded filter. The trajectory says the AI relied on the SDK’s default ROI filtering and also chose to skip sessions with missing signals or too few valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are derived from AllenSDK event traces, specifically `dataset.events["events"]`.

ii. 
```python
events_df = dataset.events
...
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says this intentionally uses “AllenSDK discrete calcium events” because the paper describes analyses on detected calcium events, and says it intentionally does not use `dF/F` or `filtered_events`.

## 2-b. How is the `neural` data processed?

i. The AI stacks the event traces from the single loaded NWB experiment into a neuron-by-time matrix, then for each trial uses nearest-neighbor sampling from ophys timestamps onto a new common 100 ms grid. No additional normalization is applied.

ii. 
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
...
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The notes justify the common 100 ms grid as a way to combine single-plane and multiscope experiments with different native frame intervals, and the trajectory states that the AI “resample[s] to a common 100 ms grid so single-plane and multiscope sessions can coexist in one decoder dataset.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI relies on the SDK-loaded events table as already quality filtered at the ROI level. It skips an experiment only if the events table is empty.

ii. 
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    print(f"  skip {experiment_id}: no valid event traces")
    continue
```

iii. The notes say “AllenSDK default ROI validity filtering is left intact,” and the trajectory says the AI confirmed the SDK excludes invalid ROIs when loading experiments.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Within each trial’s `start_time` to `stop_time` window, the AI constructs bin-center timestamps and aligns neural activity by choosing the nearest ophys timestamp to each bin center. Metadata describes the alignment event as trial start time from the trials table.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
"temporal_alignment_event": "trial start time from AllenSDK trials table",
"off_start": 0.0,
```

iii. `CONVERSION_NOTES.md` says temporal alignment is based on ophys timestamps, with bin timestamps defined as bin centers. The trajectory says the AI chose “Allen trial start alignment.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms time bin for every trial and session. Yes, temporal rebinning is applied: the original streams are resampled onto that common grid.

ii. 
```python
TIME_BIN_MS_DEFAULT = 100.0
...
def make_target_times(start_time: float, stop_time: float, bin_size_s: float) -> np.ndarray:
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]
...
"time_bin_size": float(time_bin_ms),
```

iii. The notes explicitly justify 100 ms bins because the local data mix single-plane and multiscope recordings with different native frame intervals, and the trajectory repeats that the AI intentionally resampled to a common 100 ms grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`. The code uses `trials_id`, `image_name`, `start_time`, `end_time`, and omission flags.

ii. 
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains(
        "change_detection", na=False
    )
].copy()
...
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
if bool(stim_row.omitted) or stim_row.image_name == "omitted":
    continue
```

iii. The trajectory says the AI confirmed each trial is linked to flashed image intervals via `trials_id`, and `CONVERSION_NOTES.md` says stimulus labels come only from the change-detection rows of `stimulus_presentations`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each trial starts with all bins labeled as `"gray"`. For each non-omitted flashed stimulus interval within that trial, bins whose centers fall inside `[start_time, end_time)` are overwritten with that stimulus’s `image_name`. A global integer mapping is built incrementally, with `"gray"` fixed as code 0.

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
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. The notes justify this as matching the flashed-task structure: 250 ms image flashes separated by 500 ms gray periods, so non-image periods are labeled gray and omissions also remain gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same 100 ms `target_times` used for the trial’s neural matrix, and bins are labeled by checking which stimulus interval contains each bin center.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
...
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The notes state that all outputs are aligned to the same trial-centered timestamps, and the trajectory says the AI wanted “all outputs to the same trial-centered timestamps.”

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the filtered `stimulus_presentations` rows for the trial, specifically the `is_change` flag on each flashed interval.

ii. 
```python
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The trajectory says the AI confirmed the changed-image row is marked with `is_change`, and the notes say `image_change` is 1 only during flashed epochs with `is_change == True`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI initializes a binary all-zero vector for the trial, then sets bins to 1 only for time bins falling within flashed stimulus intervals marked `is_change`. Gray periods and omissions stay 0.

ii. 
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.omitted) or stim_row.image_name == "omitted":
    continue
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. `CONVERSION_NOTES.md` explicitly justifies this with the flashed-image task structure: only the changed flash itself is marked as change, while all other periods remain non-change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous threshold is applied. The variable is built directly as binary categories: 0 for `no_change`, 1 for `change`.

ii. 
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
...
"output_values": [
    image_values,
    ["no_change", "change"],
```

iii. The notes describe `image_change` as a binary decoder output with values `no_change` and `change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same 100 ms `target_times` grid as the neural data, using interval membership on each trial’s flashed stimulus rows.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
if bool(stim_row.is_change):
    change_labels[mask] = 1
...
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The notes say all outputs are aligned by timestamp to the common trial-centered bins, and the trajectory says the AI aligned all outputs to the same timestamps as neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `timestamps` and `speed` columns.

ii. 
```python
running_df = dataset.running_speed
...
running_df["timestamps"].to_numpy(dtype=np.float64),
running_df["speed"].to_numpy(dtype=np.float64),
```

iii. The notes describe this as “AllenSDK processed running speed,” and no alternate source is used.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to each trial’s 100 ms bin centers, pooled globally across all kept trial bins, and discretized into five percentile bins using 20/40/60/80 percentile edges.

ii. 
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
...
all_running.append(running_trial)
...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. `CONVERSION_NOTES.md` says running speed is aligned by timestamp and discretized into global quintiles over all kept trial time bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI thresholds running speed into five equal-percentile bins by computing global quintile boundaries and then applying `np.digitize`.

ii. 
```python
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
def remap_to_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int16)
...
[f"bin_{i}" for i in range(5)]
```

iii. The notes explicitly say running is discretized into global quintiles.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 100 ms `target_times` grid that defines each trial’s neural matrix.

ii. 
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
...
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The trajectory says the AI chose to align “all outputs to the same trial-centered timestamps,” and the notes say running speed is aligned by timestamp to the common bin grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_area"]`, not from `pupil_width`.

ii. 
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. `CONVERSION_NOTES.md` says the output uses “equivalent diameter from AllenSDK `pupil_area`.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code converts pupil area to equivalent circular diameter, linearly interpolates that signal to the 100 ms trial bin centers, pools values globally across all kept trial bins, and bins them into global quintiles. It does not remove likely blinks before interpolation.

ii. 
```python
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
...
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
...
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. The notes justify the equivalent-diameter transform and global quintile binning, and the trajectory focuses on keeping all outputs on the same resampled time base.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins using 20/40/60/80 percentile cut points and `np.digitize`.

ii. 
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
def remap_to_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int16)
...
[f"bin_{i}" for i in range(5)]
```

iii. `CONVERSION_NOTES.md` explicitly says pupil diameter is discretized into global quintiles.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same per-trial 100 ms `target_times` grid used for the neural matrices.

ii. 
```python
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
...
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The notes say pupil is “aligned by timestamp” to the common bin grid, and the trajectory says the AI aligned all outputs to the same timestamps as neural activity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean `hit`, `miss`, `false_alarm`, and `correct_reject` columns in `dataset.trials`.

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
```

iii. The notes describe exactly this 4-way encoding for static trial outcome.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps those four mutually exclusive booleans to integer codes 0-3 and then repeats the selected code across all time bins in the trial.

ii. 
```python
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
def build_output_array(
    image_labels: np.ndarray,
    change_labels: np.ndarray,
    running_bins: np.ndarray,
    pupil_bins: np.ndarray,
    outcome_idx: int,
) -> np.ndarray:
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` says `trial_outcome` is static within each trial and encoded as 0 hit, 1 miss, 2 false alarm, 3 correct reject.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data mainly by skipping. It skips failed loads, sessions with too few ophys timestamps, empty event tables, missing running, entirely invalid pupil signals, sessions with too few valid trials, trials with fewer than two 100 ms bins, trials with no valid interpolated running or pupil signal, and trials with no recognized outcome. For interpolation, `interp_signal` drops non-finite samples and extrapolates constant edge values rather than leaving NaNs.

ii. 
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
except Exception as exc:
    ...
    continue
...
if ophys_timestamps.size < 2:
    ...
    continue
...
if len(events_df) == 0:
    ...
    continue
...
if running_interp_source is None:
    ...
    continue
...
if not np.isfinite(pupil_diameter).any():
    ...
    continue
...
if target_times.size < 2:
    continue
...
valid = np.isfinite(source_times) & np.isfinite(source_values)
if valid.sum() == 0:
    return None
...
interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
```

iii. The notes frame this as working with an incomplete local release and excluding a few experiments with entirely invalid pupil data. The trajectory repeatedly justifies “available NWB files only” and skipping broken sessions instead of trying to recover them.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are loading each NWB experiment file through AllenSDK and then iterating through all accepted trials to build resampled neural and output arrays. The code is organized around sequential per-experiment loading.

ii. 
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    for trial_id, trial_row in trials.iterrows():
        ...
        neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The trajectory explicitly says the full run was CPU-bound while the SDK parsed hundreds of NWB files, and that the loader was the long-running part of the pipeline.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-trial loop over `trials.iterrows()` and the nested per-stimulus loop over `trial_stim.itertuples()`. The code also repeatedly calls interpolation helpers inside the trial loop.

ii. 
```python
for trial_id, trial_row in trials.iterrows():
    ...
    running_trial = interp_signal(...)
    pupil_trial = interp_signal(...)
    ...
    for stim_row in trial_stim.itertuples():
        ...
        image_labels[mask] = image_idx
        if bool(stim_row.is_change):
            change_labels[mask] = 1
```

iii. The AI does not explicitly justify leaving these loops unvectorized. The trajectory instead emphasizes correctness and compatibility across the mixed local subset, not performance optimization.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of processing per trial: converting the same running and eye-tracking timestamp columns to NumPy arrays inside the loop, re-running interpolation for each trial, filtering `change_detection` by `trials_id` and sorting it for every trial, and only later doing a second pass over all trials to bin running and pupil.

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
    ...
    trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
    trial_stim = trial_stim.sort_values("start_time")
...
for output_temp_trials in output_temp_sessions:
    for trial in output_temp_trials:
        running_bins = remap_to_bins(trial["running"], running_edges)
        pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. There is no explicit justification for these repeated passes in the notes. The only clear rationale is implicit: the AI first stores continuous running and pupil so it can compute global quintile edges before creating final categorical outputs.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra bookkeeping and preprocessing that the downstream decoder does not use: it computes `running_interp_source` only as a presence check, accumulates rich `session_info` metadata including `local_path`, tracks `excluded_sessions`, and builds summary statistics such as native frame-interval summaries. It also stores continuous running and pupil in temporary structures purely to support later binning, then discards those continuous arrays from the final saved outputs.

ii. 
```python
running_interp_source = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    np.array([ophys_timestamps[0]], dtype=np.float64),
)
if running_interp_source is None:
    ...
...
output_temp_sessions.append(
    {
        "image_labels": image_labels,
        "change_labels": change_labels,
        "running": running_trial,
        "pupil": pupil_trial,
        "outcome_idx": outcome_idx,
    }
)
...
"session_info": session_info,
"native_ophys_frame_interval_ms_summary": {
    "mean": float(np.mean(native_dt_ms)),
    ...
},
```

iii. The notes justify some of this as validation and provenance tracking for the delivered artifacts, but not as something required by the decoder itself.
