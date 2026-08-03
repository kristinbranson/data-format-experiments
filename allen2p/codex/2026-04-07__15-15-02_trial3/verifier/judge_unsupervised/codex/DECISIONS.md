# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads session metadata from `project_metadata/ophys_experiment_table.csv`, intersects that with locally present NWB files, excludes metadata rows marked `passive`, and then opens each kept `behavior_ophys_experiment_<id>.nwb` file with `h5py`. It does this in two passes: one pass to collect global running and pupil statistics and a second pass to build the converted trials.

ii. ```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)
    ...
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
```

```python
for idx, session in enumerate(sessions, start=1):
    with h5py.File(session.filepath, "r") as f:
        ...
```

iii. `CONVERSION_NOTES.md` says the agent chose “local active experiment NWBs only,” treated the processed NWB contents as authoritative, and used a two-pass design to compute global quantile bins before conversion.

## 1-b. How are the data split into subjects?

i. Subjects are defined from the per-session `mouse_id` stored in metadata. The agent makes a sorted unique `subjects` list and a per-session `subject_idx`.

ii. ```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes say subject identity comes from `mouse_id` in `ophys_experiment_table.csv`, matching the Allen project metadata rather than being inferred from filenames or NWB internals.

## 1-c. How are the data split into sessions?

i. Each local `ophys_experiment_id` NWB file is treated as one session in the output. Session order is the order of the filtered metadata rows, sorted by `ophys_experiment_id`.

ii. ```python
available_files = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
}
...
exp_table = exp_table.sort_values("ophys_experiment_id")
```

```python
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
    neural_all.append(neural_trials)
```

iii. The notes justify this by matching the AllenSDK `BehaviorOphysExperiment` granularity: one imaging plane / one neuron set / one targeted structure per `ophys_experiment_id`.

## 1-d. How are the data split into trials?

i. Trials come from the processed NWB `intervals/trials` table. After filtering, each remaining row becomes one trial, and the trial’s time axis is reconstructed from `start_time` to `stop_time`.

ii. ```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id", "start_time", "stop_time", "go", "catch", "aborted",
        "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "initial_image_name", "change_image_name",
    ]
    trials = read_interval_table(f["intervals"]["trials"], columns)
```

```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the agent intentionally used the already processed Allen-style trial table instead of re-deriving trial structure from lower-level logs.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Sessions are also skipped if they have fewer than two kept trials, lack task stimulus presentations, or lack eye tracking.

ii. ```python
trials = trials[(trials["go"] | trials["catch"]) &
                (~trials["aborted"]) &
                (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    continue
...
if presentations.empty:
    continue
...
except KeyError as exc:
    print(f"... skip ...: {exc}")
```

iii. The notes justify the trial mask from Allen contingent-trial logic and the extra session filters from decoder-format constraints: at least two trials are required, and pupil output requires eye-tracking data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection matrix and its timestamps, specifically `processing/ophys/event_detection/data` and `processing/ophys/event_detection/timestamps`.

ii. ```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The notes explicitly say the agent chose event traces instead of dF/F because the reference paper’s neural analyses use extracted calcium events.

## 2-b. How is the `neural` data processed?

i. The agent does not recompute dF/F or event detection. It linearly interpolates the event matrix from native ophys timestamps onto a common 30 Hz per-trial grid and transposes the result to neuron-by-time.

ii. ```python
def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
neural_trials.append(neural_trial.astype(np.float32, copy=False))
```

iii. The notes justify this as a compromise between the reference data and the decoder requirement that all sessions share one bin size. They also say `filtered_events` were avoided because those are for visualization in AllenSDK.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no explicit ROI-level filter. It uses all rows present in the NWB event matrix and assumes the stored NWB contents already reflect Allen ROI curation.

ii. ```python
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

iii. The notes say the agent checked local files and concluded all listed ROIs were already `valid_roi`, so it did not add an explicit `valid_roi` mask in conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned in absolute ophys time and then segmented by trial bounds. Within each trial, bins are centered from trial `start_time` to `stop_time`.

ii. ```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    return centers[valid]
```

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes describe this as “align by absolute ophys time, then cut into trials,” and the saved metadata labels the alignment event as `trial start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 30 Hz grid with `DT = 1/30 s`, i.e. `33.333... ms` bins. Native streams are resampled or interpolated onto that common grid.

ii. ```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

```python
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "sampling_grid_hz": 30.0,
}
```

iii. The notes justify this as a harmonization choice because native ophys rates differ across sessions, while the target decoder format requires one consistent bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from task stimulus-presentation rows, mainly `image_name`, `start_time`, `stop_time`, `omitted`, and `trials_id`.

ii. ```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name", "active", "duration",
]
df = read_interval_table(group, columns)
```

```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes say the agent preferred the processed stimulus-presentation tables for time-varying stimulus outputs and used trial fields only as sanity checks.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each trial is initialized as `gray` everywhere. For bins overlapping a presentation interval, the code writes that interval’s `image_name`; omitted stimuli are also mapped to `gray`.

ii. ```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes say the agent explicitly introduced a `gray` category so the full trial timeline could be labeled, including inter-stimulus and omitted periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is written on the same per-trial `centers` grid used for `neural`, by masking bins whose centers fall inside each presentation’s `[start_time, stop_time)`.

ii. ```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes say all streams are aligned on one trial grid so decoder inputs and outputs share the same time axis as the neural matrix.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the task-presentation `is_change` flag together with each presentation interval’s `start_time` and `stop_time`.

ii. ```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name", "active", "duration",
]
```

```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say the change label came from processed stimulus presentations rather than being re-derived from neighboring image names inside the converter.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector and sets it to `1` for bins inside any presentation interval marked `is_change`.

ii. ```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as aligning the “change” indicator to the actual change flash rather than to a trial-wide label.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no continuous thresholding step. The agent directly encodes the processed boolean change flag as binary categories `0` and `1`.

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
]
```

iii. The notes treat image change as already categorical in the source tables, so no discretization beyond integer coding was needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned exactly like image identity: bins on the shared 30 Hz trial grid are marked by overlap with each change presentation interval.

ii. ```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say image and change traces were both built directly on the neural trial grid for one-to-one temporal alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its `timestamps`.

ii. ```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes say the code uses the stored processed running-speed stream rather than recomputing speed from wheel deltas.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated onto the trial grid, then later converted to quantile bins. A first pass concatenates all kept-trial running samples to compute dataset-wide edges.

ii. ```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_values.append(running_trial)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The notes justify the two-pass design because global percentile bins require seeing all included sessions before final conversion.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The code computes five equal-quantile bin edges globally across all finite running samples from all kept sessions, then digitizes each per-trial value into bins `0..4`.

ii. ```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
```

```python
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes explicitly say quantile bins were computed globally, not per session, so the output categories have one dataset-wide meaning.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is linearly interpolated to the same per-trial `centers` grid used for neural data.

ii. ```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes say running, pupil, and stimulus outputs were all sampled onto the neural trial grid to avoid cross-stream timing mismatches.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `timestamps` plus raw `pupil_tracking/width` and `pupil_tracking/height`.

ii. ```python
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The notes say the agent chose the geometric pupil fields because the requested decoder output was specifically pupil diameter rather than pupil area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes `diameter = 2 * max(width, height)` at each eye-tracking frame, fills NaNs by time interpolation, resamples onto the trial grid, and later bins the values globally.

ii. ```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes justify the geometry-based conversion as matching the pupil ellipse fields, and the interpolation step as a way to handle blink-related missing samples while preserving a time-varying output for every bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, the code computes five global quantile bins over all finite pupil samples from kept sessions and digitizes each trial’s resampled values into bins `0..4`.

ii. ```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes say pupil bins were intentionally global so the categories are comparable across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The continuous pupil trace is linearly interpolated onto the same per-trial `centers` grid used for the neural matrix.

ii. ```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes treat pupil alignment the same way as running alignment: absolute timestamps first, then trial-grid resampling.

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

iii. The notes say the agent relied on the Allen-processed trial outcomes already stored in NWB rather than reconstructing outcome logic again.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the mutually exclusive trial outcome flags to indices `0..3` and repeats that label across all bins of the trial, producing a constant time series.

ii. ```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes justify this as keeping all outputs in one `(n_output, T)` format even though trial outcome is semantically static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye tracking for a whole session causes the session to be skipped via `KeyError`. Missing pupil samples within a session are linearly interpolated. Missing trial IDs are synthesized. Empty or invalid trial windows return zero-length bins and are skipped. All-zero neural trials are retained.

ii. ```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
if finite.sum() == 0:
    raise ValueError("All values are NaN")
...
if "id" not in trials:
    trials["id"] = np.arange(len(trials), dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly records these as edge-case rules: skip sessions without required pupil data, interpolate blink-related pupil gaps, and keep zero-event trials because they exist in the source event data.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is opening every NWB file twice, reading large event matrices from disk, and interpolating them onto trial grids during conversion.

ii. ```python
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly identify full-dataset I/O and the mandatory two-pass design as the main runtime costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over trials in both passes, loops over trial presentations within each trial, and loops over byte strings in `decode_str_array`. Those loops could be reduced further, especially the per-trial resampling in pass 1 and the per-presentation masking in pass 2.

ii. ```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(...)
    running_trial = linear_resample_vector(...)
    pupil_trial = linear_resample_vector(...)
```

```python
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. The notes mention that the agent vectorized neural interpolation, but the rest of the session/trial logic remains largely loop-based.

## 9-c. What processing does the code repeat multiple times?

i. The converter reads each kept NWB file twice. It also rebuilds trial bins twice conceptually: once in pass 1 for global running/pupil statistics and again in pass 2 for final data creation.

ii. ```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly call out the two-pass workflow as necessary for global bin edges but acknowledge that it repeats file I/O and trial-grid construction.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter stores several metadata fields in `SessionMeta` that are not used in the final dataset, optionally builds diagnostic plots that are not part of the decoder input, and performs plotting-only extraction of raw windows when `--show-processing` is enabled.

ii. ```python
@dataclass(frozen=True)
class SessionMeta:
    ...
    session_type: str
    experience_level: str
    project_code: str
```

```python
if show_processing and not plotted:
    ...
    raw_window = events[start:stop].T.astype(np.float32, copy=False)
    make_processing_plot(...)
```

iii. The notes say the plotting path exists only for manual validation, and those artifacts are not used by `train_decoder.py` or preserved inside `converted_data.pkl`.
