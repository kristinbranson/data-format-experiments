# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds an experiment table from the Allen local cache, intersects it with NWB files physically present on disk, and then loads each `BehaviorOphysExperiment` file directly from its local NWB path. It does not use the reference approach of filtering to `project_code == "VisualBehavior"` and then grouping multiple experiments into one ophys session.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
experiments = cache.get_ophys_experiment_table()
experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
experiments["local_path"] = [str(local_paths[int(idx)]) for idx in experiments.index]
...
experiments = experiments.loc[~experiments["passive"]].copy()
...
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. `CONVERSION_NOTES.md` says the converter intentionally used only the NWB files already present under `/app/data/.../behavior_ophys_experiments`, and that each decoder session was defined as one `BehaviorOphysExperiment` file. The notes justify this as matching the local data subset and avoiding cross-plane merging in multiscope sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by unique `mouse_id` values from the experiment table. The script converts each `mouse_id` to a string and stores a `subject_to_idx` mapping used when sessions are appended.

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

iii. The notes describe the converted dataset in terms of subjects/mice and report counts by unique `mouse_id`, so the justification is simply that Allen experiment metadata already identifies each mouse.

## 1-c. How are the data split into sessions?

i. The agent treats each experiment file as one decoder session. It does not group multiple experiments that share the same `ophys_session_id`; instead, every retained row in the experiment table becomes its own session entry.

ii.
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_temp_sessions.append(output_temp_trials)
    session_info.append(
        {
            "experiment_id": int(experiment_id),
            "ophys_session_id": int(row["ophys_session_id"]),
            ...
        }
    )
```

iii. `CONVERSION_NOTES.md` explicitly says each decoder session is one Allen `BehaviorOphysExperiment` NWB file, not one unique behavior/ophys session, and argues this avoids merging different imaging planes from multiscope recordings.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. For each retained trial row, the script uses `start_time` and `stop_time` as boundaries and creates regularly spaced bin centers (`target_times`) inside that interval. One converted trial is produced per qualifying trials-table row.

ii.
```python
trials = dataset.trials.copy()
...
for trial_id, trial_row in trials.iterrows():
    start_time = float(trial_row["start_time"])
    stop_time = float(trial_row["stop_time"])
    target_times = make_target_times(start_time, stop_time, bin_size_s)
    if target_times.size < 2:
        continue
```

iii. The notes say trial segmentation is based on the AllenSDK `trials` table and uses the Allen trial start/stop window, with temporal alignment on ophys timestamps after constructing a common 100 ms grid.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `go` or `catch` trials, excluding `aborted` and `auto_rewarded` trials. Trials with fewer than two 100 ms bins are dropped, and trials are also skipped if running or pupil interpolation fails or if the outcome cannot be assigned. Sessions with fewer than two surviving trials are skipped.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
if len(trials) < 2:
    ...

target_times = make_target_times(start_time, stop_time, bin_size_s)
if target_times.size < 2:
    continue
...
if running_trial is None or pupil_trial is None:
    continue
...
if outcome_idx is None:
    continue
...
if len(neural_trials) < 2:
    ...
```

iii. The notes justify keeping only experimentally meaningful `go`/`catch` trials and excluding premature-lick resets and free-reward trials. They also note that sessions lacking usable pupil data were excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives neural data from AllenSDK event traces, specifically `dataset.events["events"]`, not from `dff_traces`.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    ...
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. The notes explicitly state that neural activity uses AllenSDK discrete calcium events and say this was chosen because the paper’s analyses were interpreted as being based on detected events rather than raw fluorescence.

## 2-b. How is the `neural` data processed?

i. Within each experiment file, the agent stacks all neuron event traces into one `neurons x time` matrix, then samples that matrix at the nearest ophys frame to each 100 ms trial bin center. It does not apply further normalization, denoising, or cross-plane merging.

ii.
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` says the converter intentionally kept AllenSDK event traces, aligned them by nearest ophys timestamp, and imposed a common 100 ms bin size to accommodate mixed single-plane and multiscope sampling rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code skips whole sessions if `dataset.events` is empty. Within loaded sessions, it performs no explicit neuron-by-neuron QC beyond whatever filtering is already embodied in the AllenSDK `BehaviorOphysExperiment` object.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    print(f"  skip {experiment_id}: no valid event traces")
    continue
```

iii. The notes claim AllenSDK default ROI validity filtering is left intact and that invalid ROIs are therefore excluded automatically when each experiment is loaded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to each trial’s `start_time`/`stop_time` window from the Allen trials table. The actual samples are taken at the centers of uniform bins within that trial window, and each bin is assigned the nearest ophys timestamp.

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The notes and metadata both say temporal alignment is based on Allen trial start times and ophys timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms bin size for all sessions and trials. Yes, temporal rebinning is applied: native ophys sampling is replaced with a uniform 100 ms grid.

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

iii. The notes justify 100 ms bins as a compromise that works across locally available single-plane and multiscope experiments, whose native frame intervals differ substantially.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations` restricted to rows whose `stimulus_block_name` contains `"change_detection"`, using each stimulus row’s `image_name`, `start_time`, `end_time`, and omission status.

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains(
        "change_detection", na=False
    )
].copy()
...
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
```

iii. The notes say stimulus labels come only from the `change_detection` stimulus block and are meant to track the flashed image during non-gray epochs, with omissions left gray.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each trial starts with all time bins labeled `"gray"`. The script then overwrites bins that fall inside each flashed image epoch with a global integer code for that image. Image codes are created lazily as new image names are encountered across sessions.

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

iii. The notes justify this by the task structure: image flashes occupy only the non-gray periods, gray intervals should remain a separate category, and omissions should not be assigned an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same `target_times` grid used for neural data. For each stimulus flash in the trial, bins whose centers fall between that flash’s `start_time` and `end_time` receive the corresponding image label.

ii.
```python
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. The notes say all outputs were aligned to the same ophys-based 100 ms trial grid so the decoder sees neural and label streams on the same time axis.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` flag in the same `change_detection` rows of `dataset.stimulus_presentations`.

ii.
```python
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The notes say `image_change` should be `1` only during flashed epochs with `is_change == True`, which the agent treats as the explicit change indicator in the stimulus table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script initializes `image_change` to all zeros and then sets it to `1` only for time bins that overlap a flashed stimulus presentation row marked `is_change == True`. It does not extend the label through the following gray period.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. `CONVERSION_NOTES.md` states that `image_change` is `1` only during flashed epochs with `is_change == True`, otherwise `0`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is encoded as a binary categorical output with categories `["no_change", "change"]`. There is no numeric threshold beyond the boolean `is_change` flag from the stimulus table.

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
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    ["hit", "miss", "false_alarm", "correct_reject"],
],
```

iii. The notes describe `image_change` as a binary output and do not mention any further thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, `image_change` is aligned on the shared `target_times` grid. Any bin whose center falls inside a change flash is labeled `1`; other bins are `0`.

ii.
```python
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The notes say all time-varying outputs are placed on the same ophys-aligned 100 ms trial grid as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
running_df = dataset.running_speed
...
running_df["timestamps"].to_numpy(dtype=np.float64)
running_df["speed"].to_numpy(dtype=np.float64)
```

iii. The notes describe this as AllenSDK processed running speed aligned by timestamp.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code filters to finite samples, linearly interpolates running speed onto each trial’s 100 ms `target_times`, concatenates all kept trial bins across the dataset, computes global quintile cutpoints, and later digitizes each trial against those cutpoints.

ii.
```python
def interp_signal(source_times, source_values, target_times):
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    ...
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)
...
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
...
all_running.append(running_trial)
...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
```

iii. The notes justify this as timestamp alignment followed by global quintile binning over all kept trial time bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five bins using the 20th, 40th, 60th, and 80th percentiles computed globally across all retained running samples, then `np.digitize` assigns bin IDs `0` to `4`.

ii.
```python
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
def remap_to_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int16)
...
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. The notes explicitly say running is discretized into global quintiles over all kept trial time bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same `target_times` array used to build each trial’s neural matrix, so both streams share the same per-trial time axis.

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. The notes say running speed is aligned by timestamp to the common 100 ms ophys-based trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using `pupil_area` and `timestamps`, not `pupil_width`.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. The notes justify this as computing an equivalent diameter from pupil area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script converts `pupil_area` to equivalent diameter, requires at least one finite value per session, interpolates the diameter onto each trial’s `target_times`, pools all retained trial bins, computes global quintile cutpoints, and digitizes each trial later.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
if not np.isfinite(pupil_diameter).any():
    ...
...
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
...
all_pupil.append(pupil_trial)
...
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
```

iii. The notes say pupil diameter is an equivalent diameter derived from `pupil_area`, aligned by timestamp, and binned into global quintiles. They also note that three sessions were excluded because pupil data were entirely invalid.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five bins using global 20/40/60/80 percentile cutpoints and `np.digitize`, yielding bin IDs `0` to `4`.

ii.
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. The notes explicitly say pupil diameter is discretized into global quintiles over all kept trial time bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same per-trial `target_times` grid used for the neural data, so both streams share one time base.

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. The notes say pupil is aligned by timestamp to the same ophys-based 100 ms grid as neural and other outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes describe these four labels as the categorical trial outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script maps the four boolean outcome labels to fixed integers `0..3`. That integer is then repeated across every time bin of the trial when the final `output` array is built.

ii.
```python
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
def build_output_array(..., outcome_idx: int) -> np.ndarray:
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
    return np.vstack([... , outcome])
```

iii. `CONVERSION_NOTES.md` lists the fixed encoding `hit=0`, `miss=1`, `false_alarm=2`, `correct_reject=3` and says trial outcome is static within each trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses several skip-or-fill rules. Experiments missing local NWB files are ignored by intersecting with on-disk file names. Entire sessions are skipped if loading fails, if ophys timestamps are too short, if events are absent, if running cannot be interpolated at all, if pupil is entirely invalid, or if too few valid trials remain. Within signals, `interp_signal` drops non-finite source samples and extrapolates with edge values rather than producing NaNs. Trials are skipped if they yield fewer than two bins, missing interpolated signals, or no recognized outcome.

ii.
```python
experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
...
except Exception as exc:
    excluded_sessions.append(
        {"experiment_id": int(experiment_id), "reason": f"load_failed:{type(exc).__name__}"}
    )
    continue
...
if ophys_timestamps.size < 2:
    ...
if len(events_df) == 0:
    ...
if running_interp_source is None:
    ...
if not np.isfinite(pupil_diameter).any():
    ...
...
valid = np.isfinite(source_times) & np.isfinite(source_values)
...
interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
...
if target_times.size < 2:
    continue
if running_trial is None or pupil_trial is None:
    continue
if outcome_idx is None:
    continue
```

iii. The notes highlight the use of the local subset, the exclusion of three experiments with entirely invalid pupil data, and the decision to keep the pipeline robust to sparse event traces and incomplete sessions.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading every experiment NWB file and then iterating trial-by-trial to build rebinned neural/behavioral arrays. The code structure makes experiment loading, per-trial interpolation, and repeated per-trial stimulus subsetting the dominant costs.

ii.
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    for trial_id, trial_row in trials.iterrows():
        ...
        running_trial = interp_signal(...)
        pupil_trial = interp_signal(...)
        trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
        for stim_row in trial_stim.itertuples():
            ...
```

iii. This is inferred from the code and the trajectory logs that repeatedly ran full conversion/training jobs; the notes do not call out a separate optimization beyond describing the dataset scale and full-run validation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop over `trials.iterrows()`, the repeated `change_detection.loc[...]` filtering plus inner `for stim_row in trial_stim.itertuples()` loop, and the second pass that digitizes and stacks outputs trial-by-trial after all sessions were collected.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
    trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
    ...
    for stim_row in trial_stim.itertuples():
        ...
...
for output_temp_trials in output_temp_sessions:
    for trial in output_temp_trials:
        running_bins = remap_to_bins(trial["running"], running_edges)
        pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
        output_trials.append(build_output_array(...))
```

iii. This is an inference from the implementation. The agent’s notes do not justify these loops as deliberate performance tradeoffs.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly interpolates running and pupil inside the per-trial loop even though both signals are session-wide. It also repeatedly filters the session-wide `change_detection` table by `trials_id` for each trial and then performs a second complete pass over all temporary outputs to bin continuous values and build the final stacked arrays.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
    running_trial = interp_signal(..., target_times)
    pupil_trial = interp_signal(..., target_times)
    ...
    trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
for output_temp_trials in output_temp_sessions:
    for trial in output_temp_trials:
        running_bins = remap_to_bins(trial["running"], running_edges)
        pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
        output_trials.append(build_output_array(...))
```

iii. This is again inferred from the code. The notes emphasize correctness and validation, not reuse of precomputed per-session intermediates.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the one-point `running_interp_source` interpolation, which is computed only to check that running data exist and is never used afterward. The code also makes many `.copy()` DataFrame copies and stores detailed `session_info` metadata such as `local_path`, even though decoder training uses only the final neural/output arrays and categorical metadata.

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
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
session_info.append(
    {
        ...
        "local_path": str(row["local_path"]),
    }
)
```

iii. This is inferred from the code plus the fact that downstream validation/training reads only the converted dataset fields required by `train_decoder.py`. The notes do not present these as intentional optimizations.
