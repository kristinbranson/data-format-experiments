# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all locally available Allen Visual Behavior ophys NWB files by scanning `behavior_ophys_experiments`, extracting `ophys_experiment_id`s from filenames, reading `project_metadata/ophys_experiment_table.csv`, intersecting the table with the available NWB ids, filtering to `behavior_type == "active_behavior"`, sorting by acquisition date, mouse, and experiment id, and then loading each NWB through `BehaviorOphysExperiment.from_nwb_path(...)`.

ii. ```python
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

def select_experiments(exp_table: pd.DataFrame, available_ids, max_sessions=None):
    available_ids = set(available_ids)
    selected = exp_table.loc[exp_table.index.intersection(available_ids)].copy()
    selected = selected[selected["behavior_type"] == "active_behavior"].copy()
    selected = selected.reset_index(drop=True)
    selected = selected.sort_values(
        ["date_of_acquisition", "mouse_id", "ophys_experiment_id"]
    )
```

```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    nwb_path = (
        data_root
        / "behavior_ophys_experiments"
        / f"behavior_ophys_experiment_{experiment_id}.nwb"
    )
    session, stats = load_session(
        experiment_id=experiment_id,
        nwb_path=nwb_path,
        meta_row=meta_row,
    )
```

iii. `CONVERSION_NOTES.md` says the source is the local Allen Visual Behavior ophys release and that sessions included are the locally available `active_behavior` experiments. The trajectory also shows the agent first inspecting the dataset layout and then building the converter around `BehaviorOphysExperiment`.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from the experiment metadata table. The final `subjects` list is the sorted set of unique mouse ids across kept sessions, and `subject_idx` maps each kept session back to that list.

ii. ```python
session = {
    "experiment_id": int(experiment_id),
    "mouse_id": str(meta_row["mouse_id"]),
    ...
}
```

```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["mouse_id"]])
```

iii. The agent did not give a long separate defense here; the justification is implicit in the task requirement that `subjects` identify mice. `CONVERSION_NOTES.md` reports final subject counts in terms of mice.

## 1-c. How are the data split into sessions?

i. The agent treats one decoder session as one `ophys_experiment_id`, not one `ophys_session_id`. This means each imaging plane is its own session in the converted dataset.

ii. ```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    ...
    session, stats = load_session(
        experiment_id=experiment_id,
        nwb_path=nwb_path,
        meta_row=meta_row,
    )
```

```python
session = {
    "experiment_id": int(experiment_id),
    ...
}
```

iii. `CONVERSION_NOTES.md` explicitly justifies this: one decoder session equals one `ophys_experiment_id` because it is one imaging plane with one ophys timestamp stream, and this avoids merging planes with different neuron sets.

## 1-d. How are the data split into trials?

i. Trials come from the AllenSDK `dataset.trials` table. After filtering, the code iterates row-by-row over valid trial records. For each kept trial, it finds stimulus presentations whose `start_time` falls between that trial’s `start_time` and `stop_time`, and represents the trial as the sequence of image-presentation intervals within that AllenSDK trial.

ii. ```python
trials = dataset.trials.copy()
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
```

```python
for trial_id, row in valid_trials.iterrows():
    trial_start = float(row["start_time"])
    trial_stop = float(row["stop_time"])
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
    if len(trial_stim) == 0:
        continue
```

iii. `CONVERSION_NOTES.md` says the converter uses AllenSDK trial boundaries and then uses the stimulus-presentation onset times inside each trial to define the variable-length within-trial sequence.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering happens in two stages. First, only `go` or `catch` trials are kept, with `aborted`, `auto_rewarded`, and non-finite `change_time` trials removed. Second, after binning, a trial is dropped if it has no stimulus rows, any non-finite binned neural/running/pupil values, all-zero neural activity, or unexpected numbers of `is_change` flags for its go/catch label.

ii. ```python
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
if len(valid_trials) < 2:
    return None, {
        "skip_reason": "too_few_valid_trials",
        "n_valid_trials": int(len(valid_trials)),
    }
```

```python
if (
    np.any(~np.isfinite(neural_trial))
    or np.any(~np.isfinite(running_trial))
    or np.any(~np.isfinite(pupil_trial))
):
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

iii. The first-stage filter is justified in `CONVERSION_NOTES.md` as matching the prompt and Allen trial-table semantics. The extra all-zero and non-finite checks are not separately justified in the notes; they appear as pragmatic QC added by the agent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `dataset.events["events"]`, with timing from `dataset.ophys_timestamps`.

ii. ```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
ophys_timestamps = dataset.ophys_timestamps.astype(np.float64)
...
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. `CONVERSION_NOTES.md` explicitly says the neural signal used is AllenSDK inferred calcium `events`, not raw fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The code stacks each cell’s inferred-event trace into a neuron-by-time matrix, then reduces that continuous ophys time series to one value per 750 ms stimulus interval using `reduce_to_bins`. Inside a bin it takes the mean across samples; if a bin has no sample it falls back to the nearest timestamp.

ii. ```python
def reduce_to_bins(values, timestamps, bin_edges):
    ...
    if values.ndim == 1:
        ...
        if hi > lo:
            reduced[i] = np.nanmean(values[lo:hi], dtype=np.float64)
        else:
            ...
            reduced[i] = values[nearest]
        return reduced

    reduced = np.empty((values.shape[0], n_bins), dtype=np.float32)
    for i in range(n_bins):
        lo = start_idx[i]
        hi = end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
        else:
            ...
            reduced[:, i] = values[:, nearest]
```

iii. The notes justify interval-level aggregation by saying the task is naturally organized into native 750 ms image-presentation intervals and that all streams were reduced onto those interval bins. The notes do not separately justify using the mean rather than another summary statistic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is applied by loading with `exclude_invalid_rois=True`, skipping sessions with zero valid ROIs or zero event rows, and dropping trials whose binned neural matrix is non-finite or entirely zero.

ii. ```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)

if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
```

```python
if (
    np.any(~np.isfinite(neural_trial))
    or np.any(~np.isfinite(running_trial))
    or np.any(~np.isfinite(pupil_trial))
):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. `CONVERSION_NOTES.md` explicitly mentions excluding invalid ROIs and dropping trials with non-finite or all-zero neural activity. The ROI choice is tied to AllenSDK semantics; the all-zero trial removal is only justified as QC in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are not aligned to a single event such as change onset. Instead, each trial is represented as a sequence of stimulus-presentation intervals, and each neural column corresponds to one interval defined by the `stimulus_presentations.start_time` values within that trial.

ii. ```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)

neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

```python
"temporal_alignment_event": (
    "Successive image-presentation interval onsets within each AllenSDK trial"
),
"off_start": 0.0,
"off_end": None,
```

iii. The notes say the agent chose image-presentation intervals because the paper and whitepaper describe the task in 750 ms flashes-plus-gray units, and because AllenSDK exposes those onset times directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 750 ms bins, matching the native image-presentation cadence. The code enforces that bin size and rebins all neural and behavioral streams onto those 750 ms trial-local intervals.

ii. ```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
```

```python
if not math.isclose(args.time_bin_ms, TIME_BIN_MS_DEFAULT, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError(
        "This converter uses native 750 ms image-presentation intervals. "
        "Keep --time-bin-ms=750."
    )
```

iii. `CONVERSION_NOTES.md` explicitly says the converter uses the native 750 ms image-presentation interval because the task is organized as 250 ms image plus 500 ms gray and the paper assigns behavior to those intervals.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus table’s `image_name` and `omitted` columns for each trial-local stimulus interval.

ii. ```python
omitted_flags = trial_stim["omitted"].fillna(False).to_numpy(dtype=bool)
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

iii. The notes explicitly describe image identity as interval-wise, using the flashed image name on non-omitted intervals and a special label on omitted intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code converts each interval’s image name into a categorical label. For omitted presentations it replaces the raw `image_name` value with `gray`, collects the global label vocabulary across all kept trials, sorts it, and converts each trial’s labels into integer ids.

ii. ```python
NO_IMAGE_LABEL = "gray"
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

```python
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
...
image_identity = np.asarray(
    [
        image_name_to_idx[name]
        for name in session["interval_image_names"][trial_idx]
    ],
    dtype=np.int64,
)
```

iii. `CONVERSION_NOTES.md` says the agent wanted omitted intervals to be labeled as `gray` rather than leaving them as the AllenSDK `omitted` pseudo-image. The trajectory’s final patch repeats that choice.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned one-to-one with the neural bins: each neural time bin has one image-identity label from the corresponding stimulus interval in the same trial.

ii. ```python
T = neural_trial.shape[1]
image_identity = np.asarray(
    [
        image_name_to_idx[name]
        for name in session["interval_image_names"][trial_idx]
    ],
    dtype=np.int64,
)
...
output_trial = np.vstack(
    [
        image_identity,
        image_change,
        running_bins,
        pupil_bins,
        trial_outcome,
    ]
).astype(np.int64)
```

iii. The justification is implicit in the interval-based design described in the notes: all outputs are reduced onto the same 750 ms bins as neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `trial_stim["is_change"]` in the stimulus-presentation table.

ii. ```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
...
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
```

iii. `CONVERSION_NOTES.md` explicitly says image change comes from the Allen stimulus table `is_change` field.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code fills missing `is_change` values with `False`, turns the result into booleans, performs a sanity check that go trials have exactly one positive interval and catch trials have none, and finally stores the values as integers in the output tensor.

ii. ```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
```

```python
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
```

iii. The notes say the intended representation is binary and interval-wise, with `1` only on true change intervals and none on catch trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric thresholding is applied. The output is already binary and is encoded as two categories: `no_change` and `change`.

ii. ```python
"output_values": [
    [name for name, _ in sorted(image_name_to_idx.items(), key=lambda x: x[1])],
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    TRIAL_OUTCOME_VALUES,
],
```

iii. The notes explicitly describe image change as a binary interval-wise output from `is_change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned one-to-one to the same per-trial 750 ms stimulus bins used for neural activity.

ii. ```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)
...
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
```

iii. The notes justify this by saying all streams are reduced onto the native image-presentation intervals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed["speed"]` with timestamps from `dataset.running_speed["timestamps"]`.

ii. ```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
```

iii. `CONVERSION_NOTES.md` explicitly names AllenSDK `running_speed["speed"]` as the source signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code removes non-finite running-speed rows, bins the remaining continuous signal by interval mean using `reduce_to_bins`, then later discretizes the binned values.

ii. ```python
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
if running_valid.sum() == 0:
    return None, {"skip_reason": "missing_running_speed"}
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
...
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```

iii. The notes justify this at a high level by saying running speed is binned by interval mean before discretization.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five global quantile bins computed across the full kept dataset, not separately per session or per trial.

ii. ```python
all_running_values = np.concatenate(all_running_values).astype(np.float32)
running_edges = compute_quantile_edges(all_running_values, nbins=5)
...
running_bins = digitize_with_edges(
    session["running_cont"][trial_idx], running_edges
)
```

iii. `CONVERSION_NOTES.md` explicitly says running speed is discretized into 5 global quantile bins across the full kept dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by reducing the continuous speed trace to the same trial-local 750 ms stimulus bins used for neural data.

ii. ```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
...
running_bins = digitize_with_edges(
    session["running_cont"][trial_idx], running_edges
)
```

iii. The notes say all neural and behavioral streams were reduced onto the same interval bins.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_width"]`, with timing from `dataset.eye_tracking["timestamps"]`.

ii. ```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
```

iii. `CONVERSION_NOTES.md` explicitly states that the pupil-size measure used is AllenSDK `eye_tracking["pupil_width"]`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code linearly interpolates missing `pupil_width` values in timestamp space, skips sessions where all pupil values are missing, bins the interpolated trace by interval mean, then discretizes the result later.

ii. ```python
def fill_nan_by_time(values, timestamps):
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    if valid.sum() == 0:
        return None
    if valid.sum() == 1:
        filled = np.empty_like(values)
        filled[:] = values[valid][0]
        return filled
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)
```

```python
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
...
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. `CONVERSION_NOTES.md` explicitly says missing pupil samples were linearly interpolated before interval binning. That is the agent’s main stated justification for handling the blink-related NaNs.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global quantile bins computed across all kept binned pupil values.

ii. ```python
all_pupil_values = np.concatenate(all_pupil_values).astype(np.float32)
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
...
pupil_bins = digitize_with_edges(
    session["pupil_cont"][trial_idx], pupil_edges
)
```

iii. `CONVERSION_NOTES.md` explicitly says pupil diameter is discretized into 5 global quantile bins across the full kept dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by reducing the eye-tracking trace to the same 750 ms stimulus-interval bins as the neural matrix.

ii. ```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
output_trial = np.vstack(
    [
        image_identity,
        image_change,
        running_bins,
        pupil_bins,
        trial_outcome,
    ]
).astype(np.int64)
```

iii. The notes justify this with the same interval-based alignment rule used for all streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the AllenSDK trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
def get_trial_outcome(row):
    if bool(row["hit"]):
        return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]):
        return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]):
        return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]):
        return TRIAL_OUTCOME_TO_INT["correct_reject"]
```

iii. `CONVERSION_NOTES.md` explicitly names those four categories as the static per-trial outcome.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the mutually exclusive Allen trial outcome flags to one integer category, then repeats that category across all time bins in the trial so it can live beside the time-varying outputs.

ii. ```python
session["trial_outcomes"].append(get_trial_outcome(row))
...
trial_outcome = np.full(
    T, session["trial_outcomes"][trial_idx], dtype=np.int64
)
```

iii. The notes explicitly justify this as a static per-trial variable that is repeated over all intervals in the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled by a mix of interpolation, nearest-neighbor fallback, and outright skipping. Pupil-width NaNs are linearly interpolated; running-speed rows with non-finite values are dropped; empty eye-tracking, empty events, all-NaN pupil, or all-NaN running sessions are skipped; bins with no samples use the nearest timestamped value; and trials with non-finite binned streams or all-zero neural activity are discarded.

ii. ```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}

if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
...
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
...
if running_valid.sum() == 0:
    return None, {"skip_reason": "missing_running_speed"}
```

```python
if hi > lo:
    reduced[i] = np.nanmean(values[lo:hi], dtype=np.float64)
else:
    nearest = np.searchsorted(timestamps, centers[i], side="left")
    ...
    reduced[i] = values[nearest]
```

```python
if (
    np.any(~np.isfinite(neural_trial))
    or np.any(~np.isfinite(running_trial))
    or np.any(~np.isfinite(pupil_trial))
):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The notes explicitly justify pupil interpolation and some session-level skipping. The rest is mostly implicit pragmatic error handling from the implementation rather than something defended in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeatedly loading NWB files into `BehaviorOphysExperiment`, building the full events matrix, scanning the stimulus table for every trial, and repeatedly calling `reduce_to_bins` over many trials and bins for neural, running, and pupil streams.

ii. ```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
...
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
...
for trial_id, row in valid_trials.iterrows():
    ...
    neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
    running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
    pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. The agent did not explicitly discuss performance in the notes or trajectory. This assessment is inferred directly from the code structure.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin loops inside `reduce_to_bins` could be vectorized or replaced with grouped reductions. The per-trial boolean filtering of `stimulus_presentations` and the second-pass per-trial discretization in `convert_sessions_to_dataset` are also obvious vectorization targets.

ii. ```python
for i in range(n_bins):
    lo = start_idx[i]
    hi = end_idx[i]
    if hi > lo:
        reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
    else:
        ...
```

```python
for trial_id, row in valid_trials.iterrows():
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
```

iii. No explicit justification was given. This is inferred from the repeated Python-level loops in the final code.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats the same interval reduction three times per trial for neural, running, and pupil. It also makes a second full pass over all sessions and trials to digitize running/pupil and encode image labels after already storing the continuous intermediate data. There are also repeated `astype(...)` conversions and repeated scanning of the trial-local image labels to build global vocabularies and counts.

ii. ```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

```python
all_running_values.extend(session["running_cont"])
all_pupil_values.extend(session["pupil_cont"])
for trial_image_names in session["interval_image_names"]:
    image_names.update(trial_image_names)
...
for session in sessions:
    ...
    running_bins = digitize_with_edges(...)
    pupil_bins = digitize_with_edges(...)
```

iii. The notes do not justify the repeated passes; they are just how the implementation is structured.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several quantities are computed only for QC or metadata and are not used by the decoder itself: `trial_ids`, `go_flags`, omitted/change sanity counts, session summary statistics, `ophys_rate_hz`, and the continuous per-trial `running_cont` and `pupil_cont` arrays after they have been digitized. The converter also builds a sample dataset and rich metadata that are not part of the actual neural-decoder inputs.

ii. ```python
session = {
    ...
    "running_cont": [],
    "pupil_cont": [],
    ...
    "trial_ids": [],
    "go_flags": [],
    "sanity_total_omitted_intervals": 0,
    "sanity_change_interval_omission_count": 0,
    "sanity_pre_change_omission_count": 0,
    "sanity_change_flag_mismatch_count": 0,
}
```

```python
stats = {
    "n_trials_before_filter": int(len(trials)),
    ...
    "ophys_rate_hz": float(1.0 / np.mean(np.diff(ophys_timestamps))),
    ...
}
...
full_data["metadata"]["session_summary"] = session_summary.to_dict(orient="records")
```

iii. The notes justify these mostly as sanity checks and documentation, not as part of the decoder-facing representation. There is no claim that the decoder consumes them.
