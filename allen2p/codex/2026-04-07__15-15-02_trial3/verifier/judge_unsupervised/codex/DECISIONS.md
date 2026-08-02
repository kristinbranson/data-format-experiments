# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads session metadata from `ophys_experiment_table.csv`, matches it to locally present NWB files, drops passive experiments, and then opens each NWB directly with `h5py`. Trials, stimulus presentations, neural events, running, and pupil data are then pulled from NWB groups on demand during a two-pass conversion.

ii. ```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)
    ...
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
```

```python
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. In `CONVERSION_NOTES.md`, the agent says it bypassed high-level AllenSDK loading because of an environment mismatch and instead mirrored the processed NWB contents directly with `h5py`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `mouse_id` from the experiment metadata table. The converted dataset stores unique sorted mouse IDs in `subjects`, and each converted session gets a `subject_idx`.

ii. ```python
SessionMeta(
    ...
    mouse_id=str(row.mouse_id),
    ...
)
```

```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes explicitly state that `mouse_id` from `ophys_experiment_table.csv` is the source for subject identity.

## 1-c. How are the data split into sessions?

i. The agent treats each local `behavior_ophys_experiment_<ophys_experiment_id>.nwb` file as one session. That means the converted session unit is an `ophys_experiment_id`, not a multi-plane `ophys_session_id`.

ii. ```python
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ophys_session_id=int(row.ophys_session_id),
            ...
            filepath=available_files[int(row.ophys_experiment_id)],
        )
    )
```

iii. In the notes, the agent justifies this as matching AllenSDK `BehaviorOphysExperiment` granularity: one imaging plane / one neuron set per file.

## 1-d. How are the data split into trials?

i. Trials are taken from the processed NWB `intervals/trials` table. Each retained trial is defined by the table’s `start_time` and `stop_time`, and a fixed 30 Hz grid is built inside those bounds.

ii. ```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id", "start_time", "stop_time", "go", "catch",
        "aborted", "auto_rewarded", "hit", "miss",
        "false_alarm", "correct_reject", "change_time",
        "initial_image_name", "change_image_name",
    ]
```

```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the NWB `trials` table is treated as the authoritative Allen-processed trial definition rather than reconstructing trials from lower-level logs.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only `go` or `catch` trials, excludes `aborted` and `auto_rewarded`, skips sessions with fewer than 2 kept trials, and skips sessions lacking stimulus presentations or eye tracking.

ii. ```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    continue
...
if presentations.empty:
    continue
...
except KeyError as exc:
    print(f"... skip ...: {exc}")
```

iii. `CONVERSION_NOTES.md` ties this to contingent-trial logic from AllenSDK and to the task requirement that pupil output is required, so sessions without eye tracking are dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/data` with timestamps from `processing/ophys/event_detection/timestamps`.

ii. ```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The notes say the agent chose event traces because the paper’s neural analyses use extracted calcium events rather than raw fluorescence.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from native ophys timestamps onto a common 30 Hz trial grid. The resulting trial matrix is transposed to neuron-by-time format and stored as `float32`.

ii. ```python
DT = 1.0 / 30.0
...
def linear_resample_matrix(src_time, src_value, dst_time):
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes describe this as a common 30 Hz grid chosen to harmonize mixed native ophys rates with the 30 Hz behavioral streams.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level filtering in `convert_data.py`. The script uses whatever rows are present in the NWB event matrix and only uses the cell table to count neurons for `brain_region_idx`.

ii. ```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

```python
ophys_time, events = get_neural_data(f)
...
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The agent’s notes argue that the included files were already effectively curated and that no additional filtering was needed beyond the processed NWB contents.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent samples neural activity on absolute ophys timestamps but segments it into per-trial bins defined from `trial.start_time` to `trial.stop_time`. In metadata, it labels the alignment event as `trial start`.

ii. ```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "trial start",
    "off_start": 0.0,
    "off_end": None,
}
```

iii. The notes say the streams are first aligned in absolute ophys time and then cut into trials, but the explicit chosen trial anchor is trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `33.333... ms` bins. Yes: the native data are resampled onto this common grid.

ii. ```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

```python
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. The notes justify this as a compromise between mixed native ophys rates and the 30 Hz behavioral streams.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from task stimulus-presentation tables in NWB intervals, mainly `image_name`, `omitted`, `start_time`, `stop_time`, and `trials_id`.

ii. ```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name",
    "active", "duration",
]
```

```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes say the agent used the stimulus-presentation tables rather than only trial-level image fields so the output could be truly time-varying.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent initializes each time bin to `gray`, then overwrites bins belonging to each stimulus flash with that flash’s image code. Omitted flashes are mapped back to `gray`.

ii. ```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes explicitly mention a `gray` category for inter-stimulus and omission periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same per-trial bin centers used for neural data; each bin is assigned according to whether its center falls inside a stimulus presentation interval.

ii. ```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes describe all outputs as sampled onto the same 30 Hz ophys-aligned trial grid as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the same stimulus-presentation tables, specifically the `is_change` flag and flash timing.

ii. ```python
columns = [
    ...,
    "is_change",
    ...
]
```

```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes cite processed stimulus presentations as the source for a time-varying change indicator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent creates a zero vector and sets bins to 1 for any presentation interval marked `is_change`.

ii. ```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say this is meant to mark the change-image flash itself, not an entire post-change trial epoch.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary categorical variable with no numeric thresholding: `0 = no_change`, `1 = change`.

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. The agent’s notes treat image change as an already discrete event from the processed stimulus table.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is assigned on the same trial bin centers as neural data using each presentation’s `start_time` and `stop_time`.

ii. ```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say all outputs share the common 30 Hz trial grid used for neural interpolation.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its `timestamps`.

ii. ```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes say the agent uses the filtered running-speed stream already stored in NWB.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto trial bins, then discretized using global five-quantile edges estimated in pass 1 across all kept sessions and trials.

ii. ```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The notes say global percentile bins were chosen so categories have consistent meaning across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five equal-percentile bins using global quantile edges. Categories are named `q1` to `q5`.

ii. ```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
running_edges = compute_quantile_edges(..., 5)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes explicitly call these global equal-frequency bins rather than per-session bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same trial-centered 30 Hz grid used for neural activity.

ii. ```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes say running, pupil, stimulus, and neural streams are all sampled on the same per-trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/eye_tracking/timestamps` and `acquisition/EyeTracking/pupil_tracking/width` and `height`.

ii. ```python
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The notes say the agent computed a diameter-like scalar from pupil-tracking geometry rather than using a precomputed pupil-area field.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent computes `diameter = 2 * max(width, height)` per frame, fills NaNs by time interpolation, interpolates onto the 30 Hz trial grid, and then digitizes with global five-quantile edges.

ii. ```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
```

```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes justify interpolation as a way to handle blink-related or missing eye-tracking samples and to keep a time-varying output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five equal-percentile bins using global quantile edges. Categories are named `q1` to `q5`.

ii. ```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes explicitly state that pupil bins are global, not session-specific.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated onto the exact same per-trial 30 Hz bin centers used for neural data.

ii. ```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes say eye tracking is aligned in absolute time and then sampled on the common trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the processed trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
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

iii. The notes say the outcome labels come directly from the Allen-processed `trials` table rather than being re-derived.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script maps the mutually exclusive booleans to integer classes and repeats that class across every time bin in the trial, making trial outcome a static-per-trial but time-broadcast output.

ii. ```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes say this was done to keep one consistent `(n_output, T)` format for all outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing string/integer fields in interval tables are tolerated by reading only present columns. Missing pupil samples are filled by interpolation if some finite values exist. Sessions missing the entire `EyeTracking` group are skipped. Trials with invalid or zero-length time windows are skipped. If a session ends up with fewer than two usable trials, it is discarded.

ii. ```python
for column in columns:
    if column not in group:
        continue
```

```python
if finite.sum() == 0:
    raise ValueError("All values are NaN")
...
values[~finite] = np.interp(...)
```

```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

iii. The notes describe this as pragmatic handling: interpolate partial eye-tracking dropouts, but drop sessions that cannot support the required pupil output at all.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is the two-pass scan over NWB files, especially repeated disk reads and per-trial interpolation of neural, running, and pupil time series.

ii. ```python
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly call out that full conversion is I/O-heavy and that global binning forces every file to be read twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorization opportunities are the Python loop over trials inside each session and the nested per-presentation loop used to paint `image_identity` and `image_change`.

ii. ```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(...)
    running_trial = linear_resample_vector(...)
    pupil_trial = linear_resample_vector(...)
```

```python
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
```

iii. The notes say neural interpolation was vectorized, but these trial- and presentation-level loops were left in straightforward Python.

## 9-c. What processing does the code repeat multiple times?

i. The script repeats file opening and data loading across two passes, rebuilds trial bins twice, and resamples running/pupil twice: once to compute global quantiles and again during final conversion.

ii. ```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly describe this as a two-pass design, chosen to get global running and pupil bin edges before final conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code generates optional diagnostic plots, stores empty `input` arrays for every trial, and expands static trial outcome into a full time series even though it is constant within a trial.

ii. ```python
input_trials = [np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in neural_trials]
```

```python
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

```python
if show_processing and not plotted:
    make_processing_plot(...)
```

iii. The notes frame these as format-compliance and debugging conveniences rather than information that changes the downstream decoder task.
