# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK cache or load the full VisualBehavior project. It reads the local metadata CSV `ophys_experiment_table.csv`, discovers locally present `behavior_ophys_experiment_*.nwb` files, filters to files that exist locally and are not marked `passive`, and then opens each kept NWB file directly with `h5py`. Data are loaded in two passes: one pass to collect global running/pupil statistics and another pass to convert sessions.

ii.
```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)

    experiment_dir = data_root / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    available_files = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
    }

    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
```

```python
with h5py.File(session.filepath, "r") as f:
    ...
```

```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. In `CONVERSION_NOTES.md`, the AI says it used direct NWB reads because the local AllenSDK/NWB stack could not instantiate the files, and it explicitly chose to work from the locally available active experiment NWBs rather than the full SDK project listing.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values taken from the kept session metadata after local-file, active-session, and later session-level filtering.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes say the converter should preserve `mouse_id` from the metadata tables as the subject identifier, with one subject index per converted session.

## 1-c. How are the data split into sessions?

i. Each local `ophys_experiment_id` NWB file is treated as one converted session. The AI does not group multiple experiments that share an `ophys_session_id`; it keeps experiment files separate and sorts them by `ophys_experiment_id`.

ii.
```python
exp_table = exp_table.sort_values("ophys_experiment_id")
...
sessions.append(
    SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        behavior_session_id=int(row.behavior_session_id),
        ...
        filepath=available_files[int(row.ophys_experiment_id)],
    )
)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly states: “Treat each `ophys_experiment_id` file as one converted session,” justifying this by the AllenSDK `BehaviorOphysExperiment` granularity and the fact that each file has one neuron set and one targeted structure.

## 1-d. How are the data split into trials?

i. Trials come from the processed NWB `intervals/trials` table. After filtering, each row becomes one trial, and the trial is represented on a regular 30 Hz grid of bin centers spanning `start_time` to `stop_time`.

ii.
```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    ...
    trials = read_interval_table(f["intervals"]["trials"], columns)
    ...
    return trials
```

```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the NWB `trials` table was used as the authoritative Allen-processed trial definition, instead of reimplementing trial parsing, and that trials should be cut on a common 30 Hz grid within each `start_time`/`stop_time` window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are `go` or `catch`, and not `aborted` and not `auto_rewarded`. Trials with empty time bins are skipped. Sessions are also skipped if they have fewer than 2 kept trials, no task stimulus presentations, or missing eye tracking.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    ...
```

```python
presentations = get_task_presentations(f)
if presentations.empty:
    ...
```

```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

```python
if centers.size == 0:
    continue
```

iii. The notes justify this as keeping “contingent trials” only, excluding passive sessions entirely, and requiring pupil availability because pupil diameter is a required decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB event-detection output, not from dF/F. It reads `processing/ophys/event_detection/timestamps` and `processing/ophys/event_detection/data`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. In the notes, the AI argues that the paper’s neural analyses use calcium events, so event traces “best match” the paper and should be used instead of dF/F.

## 2-b. How is the `neural` data processed?

i. Neural processing consists of linearly resampling the event-detection matrix from native ophys timestamps onto the 30 Hz per-trial bin centers, then transposing it to neuron-by-time. There is no multi-plane stacking because each experiment file is treated as its own session.

ii.
```python
def linear_resample_matrix(
    src_time: np.ndarray,
    src_value: np.ndarray,
    dst_time: np.ndarray,
) -> np.ndarray:
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes say a common 30 Hz grid was chosen because native ophys rates vary and the decoder expects one consistent bin size across sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no explicit neuron-level quality filter. It uses every column in `event_detection/data` and sets brain-region indices by counting every row in `cell_specimen_table["cell_specimen_id"]`; it never checks `valid_roi`.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

```python
ophys_time, events = get_neural_data(f)
...
brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
```

iii. The notes justify this by claiming the processed NWB contents already reflect Allen curation and by stating that, for the included local files, listed cells were effectively treated as valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to trial start and stop times by creating absolute-time 30 Hz bin centers inside each trial and sampling neural data onto those centers.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes explicitly describe the alignment choice as “align by absolute ophys time, then cut into trials,” using ophys timestamps as the neural reference stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid (`33.333... ms` bins). Yes: all streams, including neural data, are temporally resampled onto that grid.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

```python
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "sampling_grid_hz": 30.0,
    ...
}
```

iii. The notes repeatedly justify this as a deliberate common-grid choice made because the raw ophys sampling rate differs across rigs and because behavior/eye tracking are already near 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `image_identity` is derived from task stimulus-presentation interval tables, specifically `image_name`, `omitted`, `start_time`, `stop_time`, `trials_id`, and `stimulus_block_name` from `intervals/*_presentations` groups whose block names contain `change_detection`.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    ...
    keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
    ...
    columns = [
        "start_time",
        "stop_time",
        "image_name",
        "omitted",
        "is_change",
        "trials_id",
        "stimulus_block_name",
        "active",
        "duration",
    ]
```

iii. The notes say stimulus presentations were used so the converter could represent flashed images, gray periods, and omissions on the resampled trial timeline, rather than relying only on trial-level initial/change image names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code creates a global image-code mapping, forces a `"gray"` category to exist and come first, initializes every time bin to gray, and then overwrites bins covered by non-omitted stimulus presentations with the corresponding image code.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes justify this by saying the decoder should explicitly encode gray inter-stimulus intervals and omitted flashes, so image identity should be reconstructed from presentation intervals rather than only from trial-level change metadata.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. `image_identity` is aligned on the same 30 Hz trial-center grid as neural data. For each stimulus-presentation row linked to the current trial, bins between `row.start_time` and `row.stop_time` get the corresponding image label.

ii.
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes say all outputs should be sampled onto the same 30 Hz trial grid as the neural data, using absolute timestamps for alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the task stimulus-presentation table’s `is_change` flag together with each presentation’s `start_time`, `stop_time`, and `trials_id`.

ii.
```python
columns = [
    "start_time",
    "stop_time",
    "image_name",
    "omitted",
    "is_change",
    "trials_id",
    ...
]
```

```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say the stimulus-presentation table is the source of truth for time-varying change flashes, with trial `change_time` used only as a sanity check during development.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code starts with zeros and marks bins as `1` only for bins covered by presentation rows whose `is_change` field is true. It does not extend the label through the following gray period.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as representing the change image flash itself on the resampled timeline, derived directly from processed stimulus-presentation annotations.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no extra thresholding step. The code uses a binary integer trace directly: `0` for non-change bins and `1` for bins belonging to a stimulus presentation with `is_change == True`.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes treat image change as an inherently categorical binary signal, so no discretization beyond the boolean-to-integer conversion was needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same 30 Hz trial-center grid as neural data, using presentation start/stop intervals to mark which bins belong to the change flash.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say all outputs were aligned to the same absolute-time 30 Hz trial grid used for neural resampling.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed`, using the NWB `timestamps` and `data` arrays.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes treat this as the processed running-speed stream already stored in the NWB, analogous to the AllenSDK running-speed object.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates running speed onto each trial’s 30 Hz bin centers, aggregates all finite trial samples across sessions in pass 1 to compute five quantile edges, and then bins each trial’s interpolated running trace in pass 2.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_values.append(running_trial)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes justify this as global equal-percentile discretization on a common 30 Hz time base so that class definitions are consistent across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five global quantile bins using edges computed from all finite running samples in pass 1. Values are clipped to the edge range before bin assignment.

ii.
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
```

```python
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```

iii. The notes explicitly say running and pupil should use five equal-percentile bins computed globally across the included dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 30 Hz trial-center timestamps used for neural resampling, so both share the same time axis within each trial.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes say all streams were aligned “on a common 30 Hz grid within each trial using absolute time and ophys timestamps as the neural reference stream.”

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width`, `.../height`, and the eye-tracking timestamps. The code computes a diameter-like quantity as `2 * max(width, height)`.

ii.
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    ...
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
```

iii. The notes say the requested output is specifically pupil diameter, so the AI chose to derive a diameter from the pupil geometry fields instead of using an existing single width column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code fills NaNs in the width/height-derived diameter trace by interpolation on the eye-tracking time axis, interpolates that continuous trace onto each trial’s 30 Hz bin centers, computes global quantile edges in pass 1, and bins the per-trial trace in pass 2.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
return timestamps, diameter
```

```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes justify this as using the required pupil-diameter measure, repairing blink-related or missing samples by interpolation, and then discretizing globally into five equal-percentile bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global quantile bins using `compute_quantile_edges` and `digitize_with_edges`, the same as running speed.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes explicitly pair running and pupil as the two outputs that should use global five-quantile discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same 30 Hz trial-center timestamps used for neural data.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes say running, pupil, and stimulus variables were all sampled onto the same trial grid as the neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
```

iii. The notes say outcome labels should come directly from the mutually exclusive processed trial-outcome flags in the NWB trial table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the four outcome booleans to fixed integer codes `0..3` and then repeats that code across every time bin in the trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
...
output_trial = np.vstack(
    [
        image_identity,
        image_change,
        running_bin,
        pupil_bin,
        outcome_trace,
    ]
)
```

iii. The notes justify repeating the label across time so every output uses the same `(n_output, T)` time-varying format, even though trial outcome is semantically static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles imperfect data by skipping problematic sessions and interpolating through missing continuous samples. Sessions are skipped if eye tracking is missing or if task presentations are absent; trials with zero valid bins are skipped; sessions with fewer than 2 usable trials are rejected; pupil NaNs are filled by time interpolation. There is no explicit clipping to recording end or NaN-to-bin-0 fallback for running/pupil.

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

```python
if presentations.empty:
    ...
```

```python
def fill_nan_by_time(time_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    ...
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values
```

```python
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. The notes justify excluding sessions without pupil because pupil is a required output, and they describe interpolation as the chosen repair strategy for blink-related or missing pupil samples.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive parts are repeated HDF5 I/O over large NWB files, especially reading the event matrices and doing a two-pass conversion. Per-trial resampling of neural, running, and pupil traces is the other substantial cost.

ii.
```python
with h5py.File(session.filepath, "r") as f:
    ...
```

```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The Step 6 notes explicitly say the conversion is “I/O-heavy,” that global binning requires a first pass that reads every file, and that the event matrices dominate load time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python-level loops that could be reduced: `decode_str_array`, the per-trial loop in `collect_global_statistics`, the per-trial loop in `convert_session`, and the inner loop over each trial’s stimulus-presentation rows. The neural interpolation itself was already vectorized.

ii.
```python
for value in values:
    ...
```

```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(...)
    ...
```

```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        ...
```

iii. The notes explicitly mention that the AI added vectorized matrix interpolation with `searchsorted` and broadcasting, which implies the remaining Python loops were left in place for convenience.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats session loading and several derived computations across the two passes. It opens every kept NWB file twice, re-reads the trial table twice, re-reads task presentations twice, and re-interpolates running and pupil once in pass 1 for global edge estimation and again in pass 2 for final output construction.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The Step 6 notes acknowledge this explicitly: the converter uses a “two-pass conversion,” and the AI notes that global binning means each file must be read twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded work is the pass-1 construction of per-trial continuous running and pupil traces solely to estimate global quantile edges; those per-trial continuous arrays are not retained. The code also reads stimulus-presentation columns such as `active`, `duration`, and `stimulus_block_name` that are not preserved in the final dataset, and it contains optional plotting code that is only for diagnostics.

ii.
```python
columns = [
    "start_time",
    "stop_time",
    "image_name",
    "omitted",
    "is_change",
    "trials_id",
    "stimulus_block_name",
    "active",
    "duration",
]
```

```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(...)
    ...
    running_values.append(running_trial)
    pupil_values.append(pupil_trial)
```

```python
def make_processing_plot(...):
    ...
```

iii. The notes justify the extra pass and diagnostics as necessary for global discretization and sanity checking, but those intermediate continuous traces and plots are not part of the final serialized decoder dataset.
