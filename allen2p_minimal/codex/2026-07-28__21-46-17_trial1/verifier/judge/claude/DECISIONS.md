# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the local data directory for available NWB files, loads an experiment table CSV from the project metadata, filters to experiments whose NWB files exist locally and whose `behavior_type` is `active_behavior`, then loads each experiment individually using `BehaviorOphysExperiment.from_nwb_path()` with `exclude_invalid_rois=True`.

ii.
```python
available_ids = get_available_experiment_ids(data_root)
exp_table = load_experiment_table(data_root)
selected_df = select_experiments(exp_table, available_ids, args.max_sessions)
# ...
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
```

iii. The agent inspected the local data directory structure and found NWB files directly. It chose to load experiments via `BehaviorOphysExperiment.from_nwb_path()` rather than through the S3 cache, since data was already available locally. The agent filtered to `active_behavior` based on the paper's distinction between active and passive sessions, reasoning that passive sessions lack meaningful behavioral trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the selected experiments in the experiment table. Each experiment row has a `mouse_id` field.

ii.
```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
```

iii. The `mouse_id` field in the experiment table is the canonical unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a single `ophys_experiment_id` (one imaging plane), NOT grouped by `ophys_session_id`. The AI treats each experiment (imaging plane) as a separate session in the output.

ii.
```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    # ... load one experiment as one session
    session = {
        "experiment_id": int(experiment_id),
        "mouse_id": str(meta_row["mouse_id"]),
        "brain_region": str(meta_row["targeted_structure"]),
        # ...
    }
```

iii. The agent investigated whether to use `ophys_experiment_id` (one imaging plane) or `ophys_session_id` (multiple planes from same behavioral session) as the dataset "session." After testing and finding some planes had different timestamp arrays, the agent settled on one session per `ophys_experiment_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. Go and catch trials are included; aborted, auto-rewarded, and trials without a finite `change_time` are excluded. For each valid trial, stimulus presentations within the trial window are queried from `stimulus_presentations`, and each 750ms image-presentation interval becomes one time bin. Trials with no matching stimulus presentations are skipped.

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
    if len(trial_stim) == 0:
        continue
    stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
    bin_edges = np.concatenate(
        [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
    )
```

iii. The agent followed the task instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. The trial boundaries come from the SDK trials table, but the internal structure uses the stimulus_presentations table to define 750ms image-presentation intervals. This was motivated by the paper's statement: "We performed all of our behavioral analysis after assigning behavioral events to each image presentation interval."

## 1-e. How are trials filtered based on quality controls?

i. Multiple quality filters are applied:
- Aborted and auto-rewarded trials excluded
- Trials without finite `change_time` excluded
- Sessions with no eye tracking data are skipped entirely
- Sessions with no valid ROIs (empty events) are skipped
- Sessions with all-NaN pupil data are skipped
- Trials with any non-finite neural/running/pupil values after binning are skipped
- Trials where all neural values are zero are skipped
- Go trials without exactly 1 `is_change` flag are skipped; catch trials with any `is_change` flags are skipped
- Sessions with fewer than 2 valid trials after all filtering are skipped
- Only `active_behavior` experiments are included

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
# ...
if np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial)) or np.any(~np.isfinite(pupil_trial)):
    continue
if np.all(neural_trial == 0):
    continue
# ...
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
```

iii. The agent applied extensive quality controls based on the data exploration. Missing eye tracking and empty ROIs would produce unusable sessions. The all-zero neural trial filter was added after the validator exposed silent trials. The change flag consistency checks ensure data integrity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events["events"]` — the Allen SDK's **inferred calcium events** (deconvolved spike events), NOT dF/F traces.

ii.
```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
```

iii. The agent found guidance in the paper's methods.txt: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f." The agent also confirmed that the SDK's `events` property provides "spiking events in traces derived" with "event trace where events correspond to the rise time."

## 2-b. How is the `neural` data processed?

i. The inferred events matrix is loaded, then temporally rebinned by averaging within each 750ms stimulus-presentation interval using the `reduce_to_bins()` function. Each bin's value is the mean of the ophys frames falling within that interval.

ii.
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
# ...
def reduce_to_bins(values, timestamps, bin_edges):
    # ... for each bin, compute nanmean of values in [lo, hi)
    reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The 750ms binning matches the paper's analysis approach of assigning events to image presentation intervals. The `exclude_invalid_rois=True` parameter filters out low-quality ROIs at load time.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are excluded at load time via `exclude_invalid_rois=True`. Sessions with no valid ROIs (empty events) are skipped. Trials where all neural values are zero after binning are skipped. Trials with any non-finite neural values are skipped.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
# ...
if np.all(neural_trial == 0):
    continue
if np.any(~np.isfinite(neural_trial)):
    continue
```

iii. The `exclude_invalid_rois=True` leverages the SDK's built-in quality control which filters out ROIs that don't pass cell segmentation validation. The all-zero filter was added after discovering silent trials during validation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the stimulus presentation intervals within each trial. The ophys timestamps are used to bin neural events into 750ms windows defined by each stimulus presentation's start time. This is NOT aligned to a single event like trial start or change time, but rather to the sequence of image presentations.

ii.
```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The agent followed the paper's approach: "We performed all of our behavioral analysis after assigning behavioral events to each image presentation interval. By image presentation interval we refer to the 750 ms interval beginning with each image presentation."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one image-presentation interval (250ms stimulus + 500ms gray inter-stimulus interval). This is a rebinning from the native ophys frame rate (~11Hz, ~91ms per frame) to the stimulus presentation rate.

ii.
```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
# ...
"time_bin_size": float(time_bin_ms),  # 750.0
```

iii. The agent chose this based on the paper's methods: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation." Neural data within each 750ms bin is averaged (nanmean), and behavioral data is similarly averaged.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the `stimulus_presentations` table, with omitted stimuli labeled as `"gray"`.

ii.
```python
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

iii. Using the stimulus_presentations table provides the actual image shown at each interval, which is more direct than inferring it from the trials table's `initial_image_name`/`change_image_name` columns.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected across all sessions and sorted to create a global mapping from image name to integer index. Omitted stimuli are labeled `"gray"` and included as a category.

ii.
```python
image_names = {NO_IMAGE_LABEL}  # starts with "gray"
# ... collect all unique names ...
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
# ...
image_identity = np.asarray(
    [image_name_to_idx[name] for name in session["interval_image_names"][trial_idx]],
    dtype=np.int64,
)
```

iii. A global sorted mapping ensures consistent integer codes across all sessions. Including `"gray"` as a category handles omitted stimulus intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is defined per 750ms stimulus-presentation interval, one-to-one with the neural data time bins. Each time bin corresponds to one stimulus presentation, so the image name for that interval directly aligns.

ii.
```python
# Each interval maps to one image name from stimulus_presentations
session["interval_image_names"].append([...])
# Later, these are converted to integer codes at the same length as neural bins
```

iii. Since both neural data and image identity are binned per stimulus-presentation interval, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
```

iii. The `is_change` flag from the stimulus_presentations table directly indicates which presentation interval contains the change stimulus.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean flags from stimulus_presentations are converted to integer (0/1). Consistency checks verify go trials have exactly 1 change and catch trials have 0.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
# ...
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
```

iii. Using the SDK's `is_change` flag is more direct than computing from change_time. The sanity checks ensure data integrity.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1). Value of 1 only at the single stimulus interval where `is_change` is True (for go trials). Catch trials have 0 throughout.

ii.
```python
image_change_value_names = ["no_change", "change"]
```

iii. This is inherently binary from the `is_change` flag — no thresholding needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — one value per stimulus-presentation interval, directly aligned with neural time bins.

ii. See 4-a/4-b code.

iii. Per-interval alignment is inherent in the binning scheme.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed["speed"]` and `dataset.running_speed["timestamps"]`.

ii.
```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
```

iii. The `running_speed` attribute is the SDK's standard filtered running speed from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite values are removed, then running speed is averaged within each 750ms stimulus-presentation bin using `reduce_to_bins()`. Global percentile-based quantile edges (5 bins) are computed across all sessions, then applied using `digitize_with_edges()` (which uses `np.searchsorted`).

ii.
```python
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
# ...
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
# ...
running_edges = compute_quantile_edges(all_running_values, nbins=5)
running_bins = digitize_with_edges(session["running_cont"][trial_idx], running_edges)
```

iii. Percentile-based binning ensures roughly equal class counts. Global edges maintain consistency across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global quantile edges. The `compute_quantile_edges` function computes the 20th, 40th, 60th, and 80th percentile values as edges, then `digitize_with_edges` (using `np.searchsorted` with `side="right"`) assigns each value to a bin 0-4.

ii.
```python
def compute_quantile_edges(values, nbins):
    percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, percentiles)
    # ... ensure monotonicity ...
    return edges

def digitize_with_edges(values, edges):
    return np.searchsorted(edges, values, side="right").astype(np.int64)
```

iii. Using quantiles for bin edges ensures approximately equal numbers of samples in each bin, which is optimal for balanced decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same 750ms stimulus-presentation bins as the neural data, using `reduce_to_bins()` with the same bin edges.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```

iii. Using the same bin edges for both neural and running data ensures temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_width"]` and `dataset.eye_tracking["timestamps"]`.

ii.
```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
```

iii. The agent followed the Allen SDK tutorial (`visual_behavior_load_ophys_data.py`) which uses `pupil_width` for plotting "pupil diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in pupil_width are filled by linear interpolation using `fill_nan_by_time()` (which uses `np.interp`). Then pupil values are averaged within each 750ms bin via `reduce_to_bins()`. Global quantile edges (5 bins) are computed and applied.

ii.
```python
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
# ...
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
# ...
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. NaN interpolation fills gaps. The same binning and discretization approach as running speed ensures consistency.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — 5 bins using global quantile edges.

ii. Same approach as 5-c with `pupil_edges`.

iii. Same reasoning as running speed — balanced class counts for decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — averaged within the same 750ms stimulus-presentation bins.

ii.
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. Same bin edges ensure alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def get_trial_outcome(row):
    if bool(row["hit"]): return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]): return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]): return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]): return TRIAL_OUTCOME_TO_INT["correct_reject"]
    raise ValueError("Trial does not have a valid decoder outcome.")
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task. They are mutually exclusive for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and broadcast to all time bins within a trial.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}
# ...
trial_outcome = np.full(T, session["trial_outcomes"][trial_idx], dtype=np.int64)
```

iii. Trial outcome is static per-trial as specified in the instructions. Broadcasting to all time bins ensures consistent shape with other outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions with empty eye_tracking are skipped
- **All-NaN pupil**: Sessions where all pupil values are NaN are skipped
- **Non-finite running speed**: Non-finite values filtered before binning
- **NaN pupil values**: Filled by linear interpolation via `fill_nan_by_time()`
- **Non-finite binned values**: Trials with any non-finite neural/running/pupil after binning are skipped
- **All-zero neural**: Trials with all-zero neural data are skipped
- **Change flag mismatches**: Trials where go/catch status doesn't match `is_change` flags are skipped
- **Empty stimulus presentations**: Trials with no matching stimulus presentations are skipped
- **Too few trials**: Sessions with fewer than 2 valid trials after all filtering are skipped

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
pupil_width = fill_nan_by_time(...)  # interpolates NaN
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
if np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial)) or np.any(~np.isfinite(pupil_trial)):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The agent applied extensive quality controls. The NaN interpolation for pupil data fills in brief gaps. Sessions and trials that don't meet quality standards are skipped with descriptive reasons tracked in stats.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `BehaviorOphysExperiment.from_nwb_path()`, which reads large NWB files containing neural events, behavioral data, and stimulus presentations. The `reduce_to_bins()` function is also called many times (once per data stream per trial).

ii. N/A

iii. NWB file I/O dominates the runtime, as each file contains full-session arrays for all data streams.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `reduce_to_bins()` function uses a Python for-loop over bins to compute nanmean for each bin. This could potentially be vectorized. The per-trial loop in `load_session` iterates sequentially over all valid trials.

ii.
```python
def reduce_to_bins(values, timestamps, bin_edges):
    # ...
    for i in range(n_bins):
        lo = start_idx[i]
        hi = end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The binning loop is straightforward but could be replaced with vectorized operations for larger datasets.

## 9-c. What processing does the code repeat multiple times?

i. The `reduce_to_bins()` function is called separately for neural, running, and pupil data for each trial, each time computing `searchsorted` on the same timestamps and bin edges. The bin edge computation (`np.searchsorted`) is repeated for each data stream despite using the same temporal bins.

ii.
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. However, the timestamps differ for each data stream (ophys vs running vs eye tracking), so the repeated searchsorted calls use different inputs and are not truly redundant.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes extensive session-level statistics (`session_stats`) and sanity check counters (omitted interval counts, change flag mismatch counts, pre-change omission counts) that are stored in metadata but not used by the decoder. It also creates a separate sample dataset (`sample_data.pkl`) and a session summary DataFrame. The `summarize_sessions` function builds a detailed DataFrame that goes into metadata.

ii.
```python
stats = {
    "n_trials_before_filter": ...,
    "omitted_interval_count": ...,
    "change_interval_omission_count": ...,
    # ...
}
# Also:
sample_data = convert_sessions_to_dataset(sessions=sample_sessions, ...)
session_summary = summarize_sessions(session_stats, sessions)
```

iii. While these statistics are useful for debugging and validation, they represent extra computation not needed for the decoder itself. The sample dataset creation duplicates the conversion process for a subset.
