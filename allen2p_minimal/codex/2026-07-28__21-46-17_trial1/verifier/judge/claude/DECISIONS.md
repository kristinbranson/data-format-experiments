# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers available experiments by scanning NWB files on disk in the `behavior_ophys_experiments` directory, then cross-references with a CSV experiment table (`ophys_experiment_table.csv`). It filters to `active_behavior` experiments. Each experiment is loaded individually using `BehaviorOphysExperiment.from_nwb_path()` with `exclude_invalid_rois=True`.

ii.
```python
def get_available_experiment_ids(data_root: Path):
    experiment_dir = data_root / "behavior_ophys_experiments"
    pattern = re.compile(r"behavior_ophys_experiment_(\d+)\.nwb$")
    experiment_ids = []
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb")):
        match = pattern.match(path.name)
        if match is not None:
            experiment_ids.append(int(match.group(1)))
    return experiment_ids

def load_experiment_table(data_root: Path):
    exp_table = pd.read_csv(data_root / "project_metadata" / "ophys_experiment_table.csv")
    ...

def select_experiments(exp_table, available_ids, max_sessions=None):
    ...
    selected = selected[selected["behavior_type"] == "active_behavior"].copy()
    ...

dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
```

iii. The AI loads data directly from local NWB files rather than using the SDK's S3 cache. It filters to `active_behavior` experiments to focus on sessions where the mouse was actively performing the task, as described in the whitepaper.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table. Each experiment's `mouse_id` is tracked and collected into a sorted unique list.

ii.
```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
```

iii. The `mouse_id` field from the Allen SDK metadata table uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` is treated as a separate session. This means each imaging plane is its own session, rather than grouping multiple planes from the same `ophys_session_id` together.

ii.
```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    ...
    session, stats = load_session(experiment_id=experiment_id, nwb_path=nwb_path, meta_row=meta_row)
    ...
    sessions.append(session)
```

iii. From CONVERSION_NOTES.md: "One decoder session is one `ophys_experiment_id`. An `ophys_experiment_id` is one imaging plane with one ophys timestamp stream. This keeps neural timestamps native and avoids merging planes with different neuron sets."

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's built-in `trials` table. For each trial, the AI extracts stimulus presentation intervals from the `stimulus_presentations` table that fall within the trial's `start_time` to `stop_time`. Each trial becomes a variable-length sequence of 750ms image-presentation intervals, not individual ophys frames.

ii.
```python
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()

for trial_id, row in valid_trials.iterrows():
    trial_start = float(row["start_time"])
    trial_stop = float(row["stop_time"])
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
    ...
    stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
    bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
    neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. From CONVERSION_NOTES.md: The paper's behavioral processing explicitly assigns events to each 750ms image-presentation interval, so the AI chose interval-based binning as the closest match to the references.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only `go` or `catch` trials that are not `aborted`, not `auto_rewarded`, and have a finite `change_time`. Additional filtering: trials with no stimulus presentations are skipped; trials with non-finite binned neural/running/pupil values are skipped; trials with all-zero neural activity are skipped; go trials must have exactly 1 change flag and catch trials must have 0. Sessions with fewer than 2 valid trials are excluded. Sessions missing eye tracking data or with no valid ROIs are also excluded.

ii.
```python
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
...
if np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial)) or np.any(~np.isfinite(pupil_trial)):
    continue
if np.all(neural_trial == 0):
    continue
...
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
...
if len(session["neural_trials"]) < 2:
    return None, {"skip_reason": "too_few_binned_trials", ...}
```

iii. The AI applies extensive quality controls including sanity checks on change flag consistency. Sessions without eye tracking are skipped entirely rather than handling missing data gracefully.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events["events"]` — the AllenSDK's inferred calcium events (deconvolved spike-like events), not `dff_traces` (dF/F fluorescence traces).

ii.
```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "This follows the paper's use of inferred/discrete calcium events rather than raw fluorescence."

## 2-b. How is the `neural` data processed?

i. The events matrix is extracted per experiment (single imaging plane). Neural data is then temporally rebinned into 750ms image-presentation intervals by averaging (`nanmean`) all ophys frames within each bin. Invalid ROIs are excluded at load time via `exclude_invalid_rois=True`.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
...
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

Where `reduce_to_bins` averages values within each bin:
```python
def reduce_to_bins(values, timestamps, bin_edges):
    ...
    reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
    ...
```

iii. The AI chose to use events (deconvolved signals) following the paper's methodology, and rebins to 750ms intervals to match the image presentation schedule described in the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are excluded at load time via `exclude_invalid_rois=True`. Sessions with no valid ROIs (`len(dataset.events) == 0`) are skipped entirely. Trials where all neural bins are zero are dropped. Trials with non-finite neural values are dropped.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
...
if np.all(neural_trial == 0):
    continue
if np.any(~np.isfinite(neural_trial)):
    continue
```

iii. The `exclude_invalid_rois=True` flag leverages the Allen SDK's built-in quality control. Additional checks prevent degenerate trials from entering the dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation intervals, not to ophys timestamps directly. For each trial, stimulus onset times from `stimulus_presentations` define bin edges, and ophys frames are averaged within each 750ms bin using `reduce_to_bins`.

ii.
```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The AI's temporal alignment is based on stimulus presentation onsets rather than the ophys timestamp stream. This was motivated by the paper's description of 750ms image-presentation intervals. The instructions say "temporally align based on ophys timestamp."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins data from the native ophys frame rate (~11 Hz, ~93ms) to 750ms image-presentation intervals. The time bin size is hardcoded as 750ms.

ii.
```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
...
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
```

iii. From CONVERSION_NOTES.md: "The whitepaper and paper describe the task as 250ms flashed images plus 500ms gray. The paper's behavioral processing explicitly assigns events to each 750ms image-presentation interval."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column and the `omitted` flag. For omitted intervals, the label is "gray".

ii.
```python
stimulus_presentations = dataset.stimulus_presentations.copy()
stimulus_presentations = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].fillna("").str.contains("change_detection")
].copy()
...
session["interval_image_names"].append([
    NO_IMAGE_LABEL if omitted else str(image_name)
    for image_name, omitted in zip(
        trial_stim["image_name"].tolist(), omitted_flags.tolist()
    )
])
```

iii. The AI uses the stimulus_presentations table for per-interval image names, which provides the actual presented image at each 750ms interval. Omitted intervals (where no image was shown) are labeled "gray".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping built from all unique image names across all sessions (including "gray" for omitted intervals). The mapping is applied per interval.

ii.
```python
image_names = {NO_IMAGE_LABEL}  # starts with "gray"
...
image_names.update(trial_image_names)
...
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
...
image_identity = np.asarray(
    [image_name_to_idx[name] for name in session["interval_image_names"][trial_idx]],
    dtype=np.int64,
)
```

iii. A global sorted mapping ensures consistent integer codes. The inclusion of "gray" as an image category accounts for omitted stimulus intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because each time bin corresponds to one stimulus presentation interval. The same bin edges used for neural data define the image identity labels.

ii.
```python
# Each interval in trial_stim corresponds to one time bin
session["interval_image_names"].append([
    NO_IMAGE_LABEL if omitted else str(image_name)
    for image_name, omitted in zip(trial_stim["image_name"].tolist(), omitted_flags.tolist())
])
```

iii. Since time bins are defined by stimulus presentation onsets, each bin naturally has one image label.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
...
image_change = np.asarray(session["interval_change_flags"][trial_idx], dtype=np.int64)
```

iii. The AI uses the SDK's pre-computed `is_change` flag from the stimulus presentations table rather than computing it from `change_time` and the `go` column.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean from stimulus_presentations is directly converted to 0/1 integer values. For go trials, exactly one interval should have `is_change=True`; for catch trials, zero intervals should. Trials that violate this are excluded.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
...
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
```

iii. The AI validates that change flags are consistent with trial type, which serves as a data integrity check.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1) from the `is_change` flag. No thresholding is needed.

ii.
```python
image_change = np.asarray(session["interval_change_flags"][trial_idx], dtype=np.int64)
```

iii. The binary nature of `is_change` directly produces the two categories: `no_change` (0) and `change` (1).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — each stimulus presentation interval corresponds to one time bin, and the change flag is a per-interval value.

ii. See 4-a.

iii. Alignment is inherent in the interval-based binning approach.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed["speed"]` and its associated timestamps.

ii.
```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
```

iii. The AllenSDK's `running_speed` attribute provides locomotion data from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are filtered to remove non-finite entries, then binned into 750ms stimulus intervals by averaging (`nanmean`) all samples within each bin via `reduce_to_bins`. The resulting values are then discretized into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
...
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
...
running_edges = compute_quantile_edges(all_running_values, nbins=5)
running_bins = digitize_with_edges(session["running_cont"][trial_idx], running_edges)
```

iii. The AI bins running speed by averaging within each 750ms interval before discretization. Quantile-based binning ensures roughly equal class counts across bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized using global quantile edges computed via `compute_quantile_edges` which finds the 20th, 40th, 60th, and 80th percentiles. Values are then assigned to bins 0-4 using `searchsorted`.

ii.
```python
def compute_quantile_edges(values, nbins):
    percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, percentiles)
    ...
    return edges

def digitize_with_edges(values, edges):
    return np.searchsorted(edges, values, side="right").astype(np.int64)
```

iii. Quantile edges ensure roughly equal bin occupancy across the full dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned into the same 750ms stimulus intervals as the neural data using `reduce_to_bins` with the same bin edges.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```

iii. By using the same bin edges for both neural and running data, temporal alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_width"]` and its timestamps.

ii.
```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(), eye_timestamps
)
```

iii. `pupil_width` from the Allen SDK eye tracking module is used as the measure of pupil size.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width NaN values are filled by linear interpolation in timestamp space (via `fill_nan_by_time`), then binned into 750ms stimulus intervals by averaging within each bin. Finally, values are discretized into 5 global quantile bins.

ii.
```python
def fill_nan_by_time(values, timestamps):
    valid = np.isfinite(values) & np.isfinite(timestamps)
    ...
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)

pupil_width = fill_nan_by_time(dataset.eye_tracking["pupil_width"].to_numpy(), eye_timestamps)
...
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. NaN filling by interpolation handles missing pupil data (e.g., blinks) before binning. Note: unlike the reference, the AI does not explicitly filter out blink frames using the `likely_blink` flag before interpolation — it relies on NaN values to implicitly handle blinks.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — global quantile edges with `searchsorted`.

ii. See 5-c (same `compute_quantile_edges` and `digitize_with_edges` functions).

iii. Same rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — binned into the same 750ms stimulus intervals using `reduce_to_bins` with the same bin edges.

ii.
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. By using the same bin edges, alignment is guaranteed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def get_trial_outcome(row):
    if bool(row["hit"]):
        return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]):
        return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]):
        return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]):
        return TRIAL_OUTCOME_TO_INT["correct_reject"]
    raise ValueError("Trial does not have a valid decoder outcome.")
```

iii. These four boolean columns are the SDK's canonical trial outcomes for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping and repeated across all time bins within a trial.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
trial_outcome = np.full(T, session["trial_outcomes"][trial_idx], dtype=np.int64)
```

iii. The mapping order matches `TRIAL_OUTCOME_VALUES`. The outcome is static per trial but repeated across time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions with empty eye tracking data are skipped entirely.
- **Missing running data**: Sessions where all running values are non-finite are skipped.
- **Missing pupil data**: NaN values are linearly interpolated via `fill_nan_by_time`.
- **All-zero neural**: Trials with all-zero neural activity after binning are dropped.
- **Non-finite values**: Trials with non-finite neural/running/pupil bins are dropped.
- **Change flag mismatches**: Trials where go/catch type doesn't match expected change flag count are dropped.
- **Few trials**: Sessions with fewer than 2 valid trials after all filtering are excluded.
- **Omitted stimuli**: Labeled as "gray" rather than excluded.

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
...
pupil_width = fill_nan_by_time(dataset.eye_tracking["pupil_width"].to_numpy(), eye_timestamps)
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
...
if np.any(~np.isfinite(neural_trial)) or ...:
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The AI takes a more aggressive filtering approach than the reference — skipping entire sessions for missing eye tracking rather than handling it gracefully, and dropping individual trials with any non-finite values.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `BehaviorOphysExperiment.from_nwb_path()`, which reads large NWB files containing neural, behavioral, and stimulus data.

ii. N/A

iii. Each NWB file contains full-session data arrays. The AI processes experiments sequentially.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that calls `reduce_to_bins` for neural, running, and pupil data separately for each trial could potentially be vectorized. The `reduce_to_bins` function itself contains a per-bin loop.

ii.
```python
for trial_id, row in valid_trials.iterrows():
    ...
    neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
    running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
    pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

And within `reduce_to_bins`:
```python
for i in range(n_bins):
    lo = start_idx[i]
    hi = end_idx[i]
    if hi > lo:
        reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The inner loop in `reduce_to_bins` iterates over each bin and computes nanmean, which could be vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The `reduce_to_bins` function is called three times per trial (neural, running, pupil) with the same bin edges but different data streams. The searchsorted within each call recomputes the same temporal indices. Additionally, `convert_sessions_to_dataset` is called twice — once for the full dataset and once for the sample dataset, reprocessing shared sessions.

ii.
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
full_data = convert_sessions_to_dataset(sessions=sessions, ...)
...
sample_data = convert_sessions_to_dataset(sessions=sample_sessions, ...)
```

iii. The repeated `searchsorted` calls within `reduce_to_bins` compute overlapping index lookups.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects extensive sanity check statistics (omission counts, change flag mismatch counts) per session that are stored in session metadata but not used by the decoder. The `summarize_sessions` function builds a full DataFrame of per-session statistics. The sample dataset is a redundant subset of the full dataset.

ii.
```python
session["sanity_total_omitted_intervals"] += int(omitted_flags.sum())
session["sanity_change_interval_omission_count"] += ...
session["sanity_pre_change_omission_count"] += ...
session["sanity_change_flag_mismatch_count"] += ...
...
session_summary = summarize_sessions(session_stats, sessions)
full_data["metadata"]["session_summary"] = session_summary.to_dict(orient="records")
```

iii. While useful for validation, these statistics are not used by the downstream decoder. The sample dataset conversion is extra processing.
