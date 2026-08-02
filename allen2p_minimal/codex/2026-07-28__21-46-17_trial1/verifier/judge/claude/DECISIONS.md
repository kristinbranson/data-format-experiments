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

iii. The AI loads NWB files directly rather than using the `VisualBehaviorOphysProjectCache`. It filters to `active_behavior` to select the relevant behavioral sessions. The `exclude_invalid_rois=True` flag pre-filters low-quality ROIs at load time.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. They are sorted alphabetically.

ii.
```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
```

iii. The `mouse_id` field from the experiment metadata uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (a single imaging plane) is treated as a separate decoder session. Multiple planes from the same `ophys_session_id` are NOT merged — they become separate sessions.

ii.
```python
session = {
    "experiment_id": int(experiment_id),
    ...
    "ophys_session_id": int(meta_row["ophys_session_id"]),
    "n_neurons": int(events_matrix.shape[0]),
    ...
}
```

iii. From CONVERSION_NOTES.md: "One decoder session is one `ophys_experiment_id`. An `ophys_experiment_id` is one imaging plane with one ophys timestamp stream. This keeps neural timestamps native and avoids merging planes with different neuron sets."

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. Each trial corresponds to one stimulus change event (go or catch). Within each trial, the AI uses the `stimulus_presentations` table to identify the image-presentation intervals, and bins data into those 750ms intervals. The trial spans from `start_time` to `stop_time`.

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
    bin_edges = np.concatenate(
        [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
    )
    neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The AI uses the stimulus_presentations table to define the temporal bins within each trial, matching the paper's description of 750ms image-presentation intervals. This results in variable-length trials (10-17 intervals per trial).

## 1-e. How are trials filtered based on quality controls?

i. Trials must be `go` or `catch`, not `aborted`, not `auto_rewarded`, and have a finite `change_time`. Additional filters: trials with no stimulus presentations are skipped; trials with non-finite binned neural/running/pupil values are skipped; trials with all-zero neural activity are skipped; go trials must have exactly 1 change flag, catch trials must have 0 change flags. Sessions with fewer than 2 valid trials or missing eye tracking or no valid ROIs are excluded entirely.

ii.
```python
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()

if len(valid_trials) < 2:
    return None, {...}

# Per-trial filters:
if np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial)) or np.any(~np.isfinite(pupil_trial)):
    continue
if np.all(neural_trial == 0):
    continue
if bool(row["go"]) and int(change_flags.sum()) != 1:
    ...
    continue
```

iii. The AI applies more aggressive filtering than the reference, including checking for non-finite values and all-zero neural trials. Sessions missing eye tracking data are skipped entirely. The `exclude_invalid_rois=True` flag at load time also pre-filters bad ROIs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events["events"]` — the Allen SDK's inferred calcium events (deconvolved spike-like events), NOT the raw dF/F fluorescence traces.

ii.
```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "This follows the paper's use of inferred/discrete calcium events rather than raw fluorescence."

## 2-b. How is the `neural` data processed?

i. The events matrix is loaded for a single imaging plane and then temporally binned into 750ms image-presentation intervals using `reduce_to_bins`, which computes the mean of all ophys samples within each interval.

ii.
```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
...
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)

def reduce_to_bins(values, timestamps, bin_edges):
    ...
    for i in range(n_bins):
        lo = start_idx[i]
        hi = end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
        else:
            nearest = np.searchsorted(timestamps, centers[i], side="left")
            ...
            reduced[:, i] = values[:, nearest]
    return reduced
```

iii. The temporal rebinning averages the neural events within each 750ms stimulus interval, reducing the data from ~11Hz to ~1.33Hz (one value per image flash).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are excluded at load time via `exclude_invalid_rois=True`. Sessions with no valid ROIs (empty events) are skipped. Trials with non-finite binned neural values or all-zero neural activity are dropped.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)

if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}

if np.any(~np.isfinite(neural_trial)):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The `exclude_invalid_rois=True` parameter leverages the Allen SDK's built-in quality control. Additional per-trial checks ensure no degenerate data enters the dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the stimulus presentation intervals within each trial. The bin edges are defined by the `start_time` of each stimulus presentation in the change-detection block, plus a final edge at the last start + 750ms. The neural data (ophys events) is averaged within each interval.

ii.
```python
trial_stim = stimulus_presentations[
    (stimulus_presentations["start_time"] >= trial_start - 1e-6)
    & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
].copy()
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. From CONVERSION_NOTES.md: "For each kept trial, stimulus intervals were taken from `stimulus_presentations` rows in the change-detection block whose `start_time` falls within the AllenSDK trial `start_time` to `stop_time`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 750ms — the native image-presentation interval. The AI explicitly rebins the data from the native ophys frame rate (~11Hz, ~90ms) to this 750ms interval by averaging within each stimulus presentation window.

ii.
```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
...
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)
```

iii. From CONVERSION_NOTES.md: "The whitepaper and paper describe the task as 250 ms flashed images plus 500 ms gray. The paper's behavioral processing explicitly assigns events to each 750 ms image-presentation interval."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column and `omitted` flag. For omitted intervals, the label is "gray".

ii.
```python
stimulus_presentations = dataset.stimulus_presentations.copy()
stimulus_presentations = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"]
    .fillna("")
    .str.contains("change_detection")
].copy()
...
session["interval_image_names"].append(
    [
        NO_IMAGE_LABEL if omitted else str(image_name)
        for image_name, omitted in zip(
            trial_stim["image_name"].tolist(),
            omitted_flags.tolist(),
        )
    ]
)
```

iii. The AI uses the stimulus_presentations table which provides the actual image shown at each interval, including handling omitted flashes (labeled as "gray"). This gives one image label per 750ms interval.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping built from all unique image names (including "gray") across all sessions.

ii.
```python
image_names = {NO_IMAGE_LABEL}  # starts with "gray"
...
for trial_image_names in session["interval_image_names"]:
    image_names.update(trial_image_names)
...
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
...
image_identity = np.asarray(
    [image_name_to_idx[name] for name in session["interval_image_names"][trial_idx]],
    dtype=np.int64,
)
```

iii. A global mapping ensures consistent integer codes across sessions. The "gray" label for omitted intervals is included as a separate category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because it is computed per stimulus-presentation interval — the same intervals used as the time bins for the neural data.

ii.
```python
# Both use the same bin_edges from stimulus_presentations
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
...
session["interval_image_names"].append(
    [NO_IMAGE_LABEL if omitted else str(image_name)
     for image_name, omitted in zip(trial_stim["image_name"].tolist(), omitted_flags.tolist())]
)
```

iii. Since each time bin corresponds to one stimulus presentation interval, the image name directly maps to each bin.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
...
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
```

iii. The `is_change` flag from the stimulus table directly indicates which interval had the image change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is converted directly to a binary integer (0 or 1). Go trials have exactly one change interval (verified by sanity check); catch trials have zero change intervals.

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
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
```

iii. The AI validates the change flag consistency as a sanity check. Trials with unexpected change flag counts are discarded.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary — no thresholding is needed. The value is 1 only on the single change interval (for go trials), 0 everywhere else.

ii. See 4-b above.

iii. N/A — it is inherently binary from the `is_change` flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — one value per stimulus-presentation interval, inherently aligned with the neural bins.

ii. See 3-c.

iii. Both neural and change flag data share the same interval-based time bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed["speed"]` and its associated timestamps.

ii.
```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
...
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
```

iii. Non-finite values in speed and timestamps are filtered out before use.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is temporally binned into 750ms intervals using `reduce_to_bins` (mean within each interval), then discretized into 5 global quantile bins computed across all sessions.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
...
running_edges = compute_quantile_edges(all_running_values, nbins=5)
...
running_bins = digitize_with_edges(session["running_cont"][trial_idx], running_edges)
```

iii. The interval mean preserves the average behavior within each stimulus presentation window. Global quantile binning ensures balanced bin counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global quantile edges. The `compute_quantile_edges` function computes 4 internal quantile boundaries (at 20th, 40th, 60th, 80th percentiles), then `digitize_with_edges` uses `np.searchsorted` to assign values to bins 0-4.

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

iii. Quantile-based binning ensures roughly equal counts per bin. Tie-breaking with `np.nextafter` prevents collapsed bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned into the same 750ms stimulus-presentation intervals as the neural data using `reduce_to_bins`, so alignment is inherent.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. Both streams use the same bin edges derived from stimulus presentation times.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_width"]` and its timestamps.

ii.
```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
```

iii. `pupil_width` is used as the pupil diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in pupil_width are linearly interpolated in timestamp space using `fill_nan_by_time` (which uses `np.interp`). Then the interpolated signal is temporally binned into 750ms intervals via `reduce_to_bins` (mean). Finally, discretized into 5 global quantile bins.

ii.
```python
def fill_nan_by_time(values, timestamps):
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    ...
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)

pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. NaN interpolation fills missing values. Blink frames are NOT explicitly filtered using the `likely_blink` flag — the AI relies on NaN values being present during blinks, which may not always be the case.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — 5 global quantile bins using `compute_quantile_edges` and `digitize_with_edges`.

ii. See 5-c for the quantile edge computation. Applied identically to pupil values.

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — pupil is binned into the same 750ms stimulus-presentation intervals.

ii.
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. Same bin edges as neural and running data ensure alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}

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

iii. These four columns are mutually exclusive for valid (non-aborted, non-auto-rewarded) trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping. The integer code is replicated across all time bins (intervals) within the trial.

ii.
```python
trial_outcome = np.full(
    T, session["trial_outcomes"][trial_idx], dtype=np.int64
)
```

iii. The mapping order is `['hit', 'miss', 'false_alarm', 'correct_reject']`, matching the reference.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions with no eye tracking data are skipped entirely.
- **No valid ROIs**: Sessions with empty events are skipped.
- **All-NaN pupil**: Sessions where all pupil values are NaN are skipped.
- **Missing running speed**: Sessions with no valid running speed data are skipped.
- **Non-finite binned values**: Trials where any binned neural, running, or pupil value is non-finite are dropped.
- **All-zero neural**: Trials where all neural values are zero are dropped.
- **Change flag mismatches**: Go trials without exactly 1 change flag or catch trials with any change flags are dropped.
- **Too few trials**: Sessions with fewer than 2 valid trials are excluded.
- **NaN pupil interpolation**: NaN values in pupil_width are interpolated.

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
...
if np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial)) or np.any(~np.isfinite(pupil_trial)):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The AI applies more aggressive quality filtering than the reference, which may discard additional trials/sessions but ensures cleaner data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each NWB experiment file via `BehaviorOphysExperiment.from_nwb_path()` is the most time-consuming step, as it reads large neural, behavioral, and stimulus data from disk.

ii. N/A

iii. Each NWB file contains full-session neural events, running speed, eye tracking, trials, and stimulus presentations.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `reduce_to_bins` function iterates over bins in a Python for-loop to compute per-bin means. This could be vectorized. The per-trial loop in `load_session` also iterates sequentially over trials.

ii.
```python
def reduce_to_bins(values, timestamps, bin_edges):
    ...
    for i in range(n_bins):
        lo = start_idx[i]
        hi = end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The loop over bins is relatively small (10-17 bins per trial), so the overhead is modest compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. The `reduce_to_bins` function is called separately for neural, running, and pupil data with the same bin edges for each trial. The bin edge computation (searchsorted) is duplicated across these calls. Additionally, `convert_sessions_to_dataset` is called twice — once for the full dataset and once for the sample dataset.

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

iii. The repeated `searchsorted` calls in `reduce_to_bins` could be precomputed once and shared.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes detailed sanity check statistics (omitted interval counts, change flag mismatch counts, pre-change omission counts) that are stored in session metadata but are not used in decoder training. It also collects extensive per-session statistics and session summary DataFrames.

ii.
```python
session["sanity_total_omitted_intervals"] += int(omitted_flags.sum())
session["sanity_change_interval_omission_count"] += int(omitted_flags[change_idx].sum())
session["sanity_pre_change_omission_count"] += int(omitted_flags[change_idx[0] - 1])
...
session_summary = summarize_sessions(session_stats, sessions)
full_data["metadata"]["session_summary"] = session_summary.to_dict(orient="records")
```

iii. These statistics are useful for validation but don't affect downstream decoder training.
