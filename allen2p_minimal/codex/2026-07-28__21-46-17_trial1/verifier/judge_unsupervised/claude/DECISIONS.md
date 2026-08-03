# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by: (1) scanning the NWB file directory for available `behavior_ophys_experiment_*.nwb` files, (2) reading the `ophys_experiment_table.csv` metadata table, (3) filtering to experiments with `behavior_type == "active_behavior"` that have corresponding NWB files, (4) iterating over each selected experiment and loading it via `BehaviorOphysExperiment.from_nwb_path()` with `exclude_invalid_rois=True`.

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
    exp_table = exp_table.set_index("ophys_experiment_id", drop=False)
    return exp_table

def select_experiments(exp_table, available_ids, max_sessions=None):
    available_ids = set(available_ids)
    selected = exp_table.loc[exp_table.index.intersection(available_ids)].copy()
    selected = selected[selected["behavior_type"] == "active_behavior"].copy()
    ...
    return selected

# In load_session:
dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
```

iii. The AI justified this approach based on the AllenSDK's standard loading mechanism. The `exclude_invalid_rois=True` parameter ensures only valid ROIs are included, following the paper's practice. The metadata table provides filtering criteria (behavior_type, project_code, etc.).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mouse_id` field from the experiment metadata table. Each unique `mouse_id` becomes a subject. The sorted unique mouse IDs form the `subjects` list, and each session is mapped to a subject via `subject_idx`.

ii.
```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[session["mouse_id"]])
```

iii. The `mouse_id` column in the experiment table directly identifies subjects. 38 unique mice were found in the local data subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` defines one session. This means each imaging plane is treated as a separate session, even when multiple planes are imaged simultaneously in Multiscope experiments (which share the same `ophys_session_id`).

ii.
```python
# In load_session:
session = {
    "experiment_id": int(experiment_id),
    ...
}
# Each experiment_id iterates separately:
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    ...
    session, stats = load_session(experiment_id=experiment_id, nwb_path=nwb_path, meta_row=meta_row)
```

iii. The AI reasoned: "An `ophys_experiment_id` is one imaging plane with one ophys timestamp stream. This keeps neural timestamps native and avoids merging planes with different neuron sets." This is documented in CONVERSION_NOTES.md.

## 1-d. How are the data split into trials?

i. Trials come from the AllenSDK `dataset.trials` DataFrame. Each row is a trial. The AI iterates over `valid_trials` rows and, for each trial, extracts stimulus presentations whose `start_time` falls within the trial's `[start_time, stop_time]` window.

ii.
```python
trials = dataset.trials.copy()
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
...
for trial_id, row in valid_trials.iterrows():
    trial_start = float(row["start_time"])
    trial_stop = float(row["stop_time"])
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
```

iii. Trial boundaries are defined by AllenSDK's `start_time` and `stop_time` fields, which correspond to the experimental trial structure.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in multiple stages: (1) Must be `go` or `catch`, not `aborted`, not `auto_rewarded`, and have finite `change_time`. (2) Must have at least one stimulus presentation within the trial window. (3) Neural, running, and pupil binned values must all be finite. (4) Neural activity must not be all zeros. (5) Go trials must have exactly 1 change flag; catch trials must have 0. (6) Sessions with fewer than 2 valid trials are dropped entirely.

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
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
```

iii. The filtering of aborted and auto-rewarded trials is explicitly required by the instructions. The additional quality checks (finite values, non-zero neural, change flag consistency) are sanity measures the AI added to ensure data integrity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `dataset.events["events"]`, which are the L0-regularized inferred calcium events from the AllenSDK, not raw fluorescence or dF/F traces.

ii.
```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
ophys_timestamps = dataset.ophys_timestamps.astype(np.float64)
```

iii. The AI justified: "This follows the paper's use of inferred/discrete calcium events rather than raw fluorescence." The AllenSDK `events` are deconvolved spike events derived from calcium traces using L0 event detection.

## 2-b. How is the `neural` data processed?

i. The events matrix (n_neurons x n_timepoints) is binned into 750ms image-presentation intervals by averaging within each bin. Bin edges are defined by stimulus presentation onset times plus one trailing edge. The `reduce_to_bins` function computes the mean of all ophys samples falling within each bin.

ii.
```python
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
# In reduce_to_bins:
reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The 750ms interval binning follows the paper's approach of analyzing data at the image-presentation interval level. The `reduce_to_bins` function uses `searchsorted` for efficient assignment of samples to bins, then averages.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are excluded at load time via `exclude_invalid_rois=True`. Trials with non-finite binned neural values or all-zero neural activity are dropped. No additional neuron-level filtering (e.g., by activity threshold or signal-to-noise) is applied.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
...
if np.any(~np.isfinite(neural_trial)):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The `exclude_invalid_rois=True` flag relies on AllenSDK's built-in ROI validation. The all-zero check removes trials where no neural events were detected.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onset times within each trial. The instructions say "temporally align based on ophys timestamp." The AI uses ophys timestamps to bin neural activity into intervals defined by stimulus presentation start times (from the stimulus_presentations table). Each trial is a variable-length sequence of 750ms image-presentation intervals.

ii.
```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The AI documented: "Successive image-presentation interval onsets within each AllenSDK trial" as the temporal alignment event. This uses ophys timestamps as the reference clock for binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, matching the native image-presentation interval (250ms image + 500ms gray). Rebinning is applied: the raw ophys data sampled at ~11 Hz (~91ms per frame) is averaged within each 750ms bin (about 8 ophys frames per bin).

ii.
```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
# Bin edges from stimulus presentation times:
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
```

iii. The AI justified this choice based on the paper's description of analyzing data at the 750ms image-presentation interval level. This is documented in CONVERSION_NOTES.md: "The paper's behavioral processing explicitly assigns events to each 750 ms image-presentation interval."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from the `image_name` column of the `stimulus_presentations` DataFrame, filtered to the `change_detection` stimulus block. For omitted flashes, the label is set to `"gray"`.

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
        trial_stim["image_name"].tolist(), omitted_flags.tolist(),
    )
])
```

iii. The image names correspond to the natural scene images used in the change-detection task (e.g., `im000`, `im031`, etc.). Omitted flashes are labeled as `"gray"` since no image was actually presented.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a sorted dictionary of all unique image names encountered across all sessions (including `"gray"`). The mapping is alphabetical.

ii.
```python
image_names = {NO_IMAGE_LABEL}  # starts with "gray"
...
image_names.update(trial_image_names)
...
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
...
image_identity = np.asarray([image_name_to_idx[name] for name in session["interval_image_names"][trial_idx]], dtype=np.int64)
```

iii. This produces a categorical integer encoding with 17 categories (gray + 16 images).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because each image presentation interval defines both a neural bin and an image label. The same stimulus presentation rows define bin edges for neural data and provide image names.

ii.
```python
# Same trial_stim used for both:
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
# Image names come from the same trial_stim rows:
session["interval_image_names"].append([...trial_stim["image_name"]...])
```

iii. Since both neural binning and image labels derive from the same stimulus presentation table rows, alignment is exact.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from the `is_change` column of the `stimulus_presentations` DataFrame.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
...
image_change = np.asarray(session["interval_change_flags"][trial_idx], dtype=np.int64)
```

iii. The `is_change` field is a boolean flag provided by the AllenSDK indicating whether a stimulus change occurred at that presentation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is converted to integer (0 or 1). NaN values are filled with False. Sanity checks ensure go trials have exactly 1 change flag and catch trials have 0.

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
image_change = np.asarray(session["interval_change_flags"][trial_idx], dtype=np.int64)
```

iii. The instructions specify "Have value of 1 right after a change in image identity, otherwise 0." The AI uses the AllenSDK's `is_change` flag directly, which marks the presentation interval where the change occurred.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1) derived directly from the boolean `is_change` field. No thresholding is needed since it is inherently binary.

ii.
```python
image_change = np.asarray(session["interval_change_flags"][trial_idx], dtype=np.int64)
# output_values for image_change:
["no_change", "change"]
```

iii. The instructions specify this as a binary variable, and `is_change` is already binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is aligned by the shared stimulus presentation intervals that define both neural bins and the change flag.

ii.
```python
# Same trial_stim for both:
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
```

iii. Alignment is inherent since both neural and change data share the same interval structure.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed["speed"]` with corresponding `dataset.running_speed["timestamps"]`.

ii.
```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
```

iii. The AllenSDK `running_speed` provides filtered running speed in cm/s, derived from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite values are removed. Running speed is then binned by averaging within each 750ms image-presentation interval using `reduce_to_bins`. Finally, it is discretized into 5 equal percentile bins using global quantile edges computed across all kept data.

ii.
```python
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
...
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
...
all_running_values = np.concatenate(all_running_values).astype(np.float32)
running_edges = compute_quantile_edges(all_running_values, nbins=5)
running_bins = digitize_with_edges(session["running_cont"][trial_idx], running_edges)
```

iii. The instructions specify "discretized into five equal percentile bins." The AI computes quantile edges globally across all kept data, then digitizes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using 4 quantile edges. The edges divide the global distribution of all running speed values into equal-frequency bins.

ii.
```python
def compute_quantile_edges(values, nbins):
    percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]  # [0.2, 0.4, 0.6, 0.8]
    edges = np.quantile(values, percentiles)
    ...
    return edges

def digitize_with_edges(values, edges):
    return np.searchsorted(edges, values, side="right").astype(np.int64)
```

iii. The output confirms exactly balanced bins: 113,579 intervals in each of 5 bins, confirming proper quantile discretization.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by binning running speed samples (at their own timestamps) into the same 750ms image-presentation interval bins used for neural data.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```

iii. The `reduce_to_bins` function handles the alignment by averaging running speed samples whose timestamps fall within each bin edge pair, using the same bin edges derived from stimulus presentation times.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter comes from `dataset.eye_tracking["pupil_width"]`, using the width of the fitted ellipse on the pupil as a proxy for diameter.

ii.
```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
```

iii. The AI chose `pupil_width` over `pupil_area` as a more direct proxy for "diameter." The eye tracking data provides ellipse fits with width, height, and area for the pupil.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in `pupil_width` are linearly interpolated in timestamp space via `fill_nan_by_time`. The interpolated signal is then binned by averaging within each 750ms interval, and finally discretized into 5 global quantile bins.

ii.
```python
def fill_nan_by_time(values, timestamps):
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    ...
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)
...
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. The interpolation handles missing pupil data (e.g., during blinks). The 5-bin quantile discretization follows the instructions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins using global quantile edges.

ii.
```python
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. Output confirms exactly balanced bins: 113,579 intervals in each of 5 bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil width is aligned by binning eye tracking samples (at their own timestamps) into the same 750ms interval bins used for neural data.

ii.
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. Same alignment mechanism as running speed, using `reduce_to_bins` with shared bin edges.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the `dataset.trials` DataFrame.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}

def get_trial_outcome(row):
    if bool(row["hit"]): return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]): return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]): return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]): return TRIAL_OUTCOME_TO_INT["correct_reject"]
    raise ValueError("Trial does not have a valid decoder outcome.")
```

iii. These four outcomes cover all non-aborted, non-auto-rewarded trial types in the change-detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first matching boolean flag (in order: hit, miss, false_alarm, correct_reject) determines the outcome integer. The outcome is then repeated (broadcast) across all time bins in the trial, making it a time-varying output with constant value per trial.

ii.
```python
trial_outcome = np.full(T, session["trial_outcomes"][trial_idx], dtype=np.int64)
...
output_trial = np.vstack([
    image_identity,
    image_change,
    running_bins,
    pupil_bins,
    trial_outcome,
]).astype(np.int64)
```

iii. The instructions specify trial outcome as "Static per-trial." The AI broadcasts the static value across all time bins in the output matrix, which is a valid way to represent it in the (n_output, n_timepoints) format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies handle missing/problematic data:
- Missing pupil data: NaN values are linearly interpolated via `fill_nan_by_time`
- Missing eye tracking entirely: sessions are skipped (3 sessions)
- All-NaN pupil: sessions skipped
- Non-finite running speed: samples removed before binning
- Non-finite binned values (neural/running/pupil): entire trial skipped
- All-zero neural: trial skipped
- Missing stimulus presentations: trial skipped
- Change flag mismatches: trial skipped
- Fewer than 2 valid trials: session skipped

ii.
```python
# Missing eye tracking
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
# Pupil NaN interpolation
pupil_width = fill_nan_by_time(dataset.eye_tracking["pupil_width"].to_numpy(), eye_timestamps)
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
# Running speed NaN removal
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
running_speed = running_speed[running_valid]
# Trial-level NaN check
if np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial)) or np.any(~np.isfinite(pupil_trial)):
    continue
```

iii. The AI documents these filtering outcomes in CONVERSION_NOTES.md: 3 sessions skipped for missing eye tracking, 0 change-flag mismatches in kept data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()` (disk I/O + parsing for 202 experiments), (2) The inner loop over trials calling `reduce_to_bins` for neural, running, and pupil data per trial, (3) Computing global quantile edges over all concatenated running/pupil values.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
...
for trial_id, row in valid_trials.iterrows():
    ...
    neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
    running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
    pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. Loading NWB files is inherently I/O-bound. The trial-level binning loop processes ~48,000 trials across 199 sessions, calling `reduce_to_bins` three times per trial.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `reduce_to_bins` function uses a Python for-loop over bins to compute means. The outer loop over trials within `load_session` could potentially be vectorized by computing all trial bin edges at once and using vectorized binning.

ii.
```python
# In reduce_to_bins - loop over bins:
for i in range(n_bins):
    lo = start_idx[i]
    hi = end_idx[i]
    if hi > lo:
        reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
    else:
        ...
```

iii. The bin-level loop in `reduce_to_bins` could be replaced with grouped operations (e.g., using `np.add.reduceat`). However, the number of bins per trial is small (10-17), so the performance impact is limited.

## 9-c. What processing does the code repeat multiple times?

i. (1) The full conversion pipeline is run twice: once for the full dataset and once for the sample dataset (`convert_sessions_to_dataset` called twice). (2) The `reduce_to_bins` function is called 3 times per trial (neural, running, pupil) with the same bin edges. (3) The quantile edge computation and digitization happen at the end on already-collected continuous values rather than being integrated into the loading loop.

ii.
```python
# Full dataset conversion
full_data = convert_sessions_to_dataset(sessions=sessions, ...)
# Sample dataset conversion (re-processes a subset)
sample_data = convert_sessions_to_dataset(sessions=sample_sessions, ...)
```

iii. The sample dataset re-runs `convert_sessions_to_dataset` on a subset of sessions that were already processed for the full dataset. This could be avoided by subsetting the full dataset's output.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Trial outcome is broadcast across all time bins (`np.full(T, ...)`) despite being a static per-trial variable; the decoder likely only needs one value per trial. (2) The sample dataset is redundant since the full dataset contains the same data. (3) Extensive summary statistics and sanity-check counters are computed during loading but only used for logging/diagnostics. (4) The `input` field is populated with empty arrays `np.zeros((0, T))` per trial since there are no decoder inputs, which is unnecessary allocation.

ii.
```python
# Trial outcome broadcast to all time bins:
trial_outcome = np.full(T, session["trial_outcomes"][trial_idx], dtype=np.int64)
# Empty input arrays:
input_trials.append(np.zeros((0, T), dtype=np.float32))
```

iii. Broadcasting trial outcome is not "wrong" but creates redundant data. The empty input arrays satisfy the format specification but waste memory.
