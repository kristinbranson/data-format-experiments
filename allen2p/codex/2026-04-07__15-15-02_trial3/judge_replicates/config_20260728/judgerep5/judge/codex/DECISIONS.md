# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache path from the reference. It reads the local `ophys_experiment_table.csv`, keeps rows whose NWB files are present locally, removes passive sessions, and then opens each NWB directly with `h5py`.

ii. 
```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)
    ...
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
```
```python
with h5py.File(session.filepath, "r") as f:
```

iii. In `CONVERSION_NOTES.md`, the AI says it switched to direct NWB reads because the local AllenSDK/NWB stack could not instantiate these files, and it limited scope to active local experiment files only.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment metadata.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes say `mouse_id` from `ophys_experiment_table.csv` is used for `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI treats each local `ophys_experiment_id` file as one session, rather than grouping multiple experiments by shared `ophys_session_id`.

ii.
```python
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

iii. The notes explicitly justify this as matching the AllenSDK `BehaviorOphysExperiment` granularity and yielding one imaging plane per converted session.

## 1-d. How are the data split into trials?

i. Trials come from the processed NWB `intervals/trials` table. Each kept trial uses the full `start_time` to `stop_time` window, but the data inside that window are represented on a new 30 Hz grid of bin centers.

ii.
```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    ...
    trials = read_interval_table(f["intervals"]["trials"], columns)
```
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the NWB `trials` table was treated as the authoritative Allen-processed trial definition, avoiding reimplementation of trial logic.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` or `catch` trials and excludes `aborted` and `auto_rewarded` trials. Trials with empty bin grids are skipped, sessions with fewer than two kept trials are dropped, and sessions missing required streams can be dropped earlier.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. The notes justify keeping only contingent `go`/`catch` trials and excluding aborted/auto-rewarded trials. They also justify dropping sessions with missing eye tracking because pupil output is required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from NWB `processing/ophys/event_detection/data` plus its `timestamps`, not from dF/F.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The notes justify this by saying the paper uses extracted calcium events and that `filtered_events` are visualization-only.

## 2-b. How is the `neural` data processed?

i. Neural event traces are linearly interpolated from native ophys timestamps onto a 30 Hz per-trial grid, then transposed to neuron-by-time for each trial.

ii.
```python
def linear_resample_matrix(
    src_time: np.ndarray,
    src_value: np.ndarray,
    dst_time: np.ndarray,
) -> np.ndarray:
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes justify this as creating a common 30 Hz grid across sessions while preserving ophys-based timing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The final script applies no explicit per-neuron quality filter. It uses every row in the event matrix and sets brain-region labels by the length of `cell_specimen_id`.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```
```python
ophys_time, events = get_neural_data(f)
```

iii. The notes argue that ROI/event curation is already embedded upstream in the processed NWB content, although the final script itself does not check `valid_roi`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to trial start: the script builds bin centers from `start_time` to `stop_time` and resamples neural data into those bins using absolute ophys timestamps.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    ...
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
```
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes say all streams are aligned by absolute ophys time and then cut into trials on a 30 Hz grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use `DT = 1/30` seconds, i.e. `33.333... ms` bins. Yes, the code rebins/resamples neural and behavioral streams to this 30 Hz grid.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```
```python
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. The notes justify this by saying native sampling rates differ across recordings and that the decoder needs a common bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The final script derives image identity from stimulus-presentation interval tables, especially `image_name`, `omitted`, `start_time`, `stop_time`, and `trials_id`.

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
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes justify using stimulus-presentation tables to build time-varying image identity on the resampled grid, with the trial table only as a sanity check.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code initializes each trial as `gray`, then overwrites bins covered by each stimulus presentation with the presented `image_name`. Omitted flashes stay `gray`. Finally, names are mapped to integer codes through a global vocabulary.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The notes justify explicitly encoding gray/omission periods because they occupy trial time on the common trial grid.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is defined on the same 30 Hz `centers` array used for neural interpolation, so both are aligned bin-by-bin within each trial.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = ...
```

iii. The notes say all outputs are sampled onto the same trial grid as neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The final script derives image change from stimulus-presentation rows, specifically the `is_change` flag and presentation timing fields.

ii.
```python
columns = [
    ...,
    "is_change",
    "trials_id",
    ...
]
```
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify using stimulus-presentation interval tables for both image identity and image-change traces.

## 4-b. What processing is involved in computing `output` *Image change*?

i. `image_change` is initialized to zero and set to one for any 30 Hz bins whose presentation row has `is_change == True`.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as marking the change-image flash from processed stimulus-presentation annotations.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for `no_change`, `1` for `change`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
],
```

iii. No further justification is given beyond the task requirement for a binary image-change output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is written onto the same 30 Hz per-trial grid used for the neural data.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say all streams are aligned to the common trial grid built from ophys time.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from NWB `processing/running/speed`, using its `data` and `timestamps`.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes justify this as using the processed running stream already stored in NWB.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz grid, then digitized into 5 global quantile bins computed across all retained sessions in pass 1.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes justify global percentile binning to keep category semantics consistent dataset-wide.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses 5 equal-percentile bins and names them `q1` to `q5`.

ii.
```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The notes explicitly say running speed should be discretized globally into five equal-frequency bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 30 Hz `centers` array used for neural interpolation, so each running bin corresponds to a neural bin.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes say all streams are aligned to one ophys-referenced trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The final script derives pupil diameter from `acquisition/EyeTracking/pupil_tracking/width`, `height`, and `eye_tracking/timestamps`.

ii.
```python
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The notes planned to use blink-filtered pupil-related eye-tracking fields, but the final code uses raw width/height arrays and derives a diameter surrogate from them.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes `2 * max(width, height)` as a diameter-like quantity, fills NaNs by interpolation in time, resamples that signal onto the 30 Hz trial grid, and then digitizes it into 5 global quantile bins. The final script does not use the SDK blink mask.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes justify interpolation to fill modest missingness and global percentile binning, but the final code is less conservative than the notes because it does not explicitly remove blinks.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI uses 5 equal-percentile bins named `q1` to `q5`.

ii.
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. The notes explicitly say pupil diameter should be globally discretized into five equal-frequency bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated to the same 30 Hz `centers` grid used for `neural`, so alignment is per bin within each trial.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes say running, pupil, stimulus, and neural signals all share the same trial grid.

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

iii. The notes justify using the processed NWB `trials` table as the authoritative outcome source.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The selected outcome class is converted to an integer index and repeated across every time bin of the trial as a constant trace.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes justify representing all outputs in a consistent time-varying `(n_output, T)` format, so trial outcome is repeated across bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by skipping sessions with missing required fields (`KeyError`, including missing `EyeTracking`), skipping trials with empty bin grids, requiring at least two usable trials per session, and filling pupil NaNs by time interpolation. Running/pupil quantile edges ignore non-finite values.

ii.
```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```
```python
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(...)
```
```python
diameter = fill_nan_by_time(timestamps, diameter)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. The notes justify excluding sessions with missing eye tracking because pupil output is required, and they describe interpolating modest missingness within retained sessions.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies disk I/O on the NWB event matrices as the main bottleneck, especially because pass 1 and pass 2 both reopen files and the largest arrays are neural event traces.

ii.
```python
print("Pass 1: collecting global running/pupil statistics")
...
with h5py.File(session.filepath, "r") as f:
```
```python
print("Pass 2: converting sessions")
...
with h5py.File(session.filepath, "r") as f:
```

iii. `CONVERSION_NOTES.md` explicitly says full conversion is I/O-heavy because each NWB event matrix must be read from disk.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The script already vectorizes neural interpolation, but it still loops over sessions, over trials within each session, and over stimulus-presentation rows within each trial.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    ...
    for trial in trials.itertuples(index=False):
```
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
```

iii. The notes say vectorized interpolation was added specifically to avoid per-neuron loops; they do not claim the remaining session/trial loops were further optimized.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats session reads and behavioral interpolation across two passes: pass 1 computes global running/pupil quantile edges and pass 2 reloads the same sessions to build trial outputs.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    ...
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly acknowledge this repeated processing and justify it as necessary for global discretization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads some trial-table columns that are never used downstream in the final conversion, such as `change_time`, `initial_image_name`, and `change_image_name`. It also supports optional diagnostic plotting that is not part of the saved dataset.

ii.
```python
columns = [
    ...,
    "change_time",
    "initial_image_name",
    "change_image_name",
]
```
```python
parser.add_argument(
    "--show-processing",
    action="store_true",
    help="Save diagnostic processing plots for up to 2 sessions",
)
```

iii. The notes justify the plotting path as a debugging/sanity-check aid. No explicit justification is given for reading unused trial columns beyond convenience and consistency with the full NWB trial schema.
