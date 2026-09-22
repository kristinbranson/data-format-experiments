# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK cache or discover the full project from `VisualBehaviorOphysProjectCache`. Instead, it enumerates locally present NWB files under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, joins them to `ophys_experiment_table.csv`, drops passive experiments, and then opens each NWB directly with `h5py`.

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

iii. In `CONVERSION_NOTES.md`, the AI says it used direct `h5py` reads because the local AllenSDK/NWB stack could not instantiate these NWB files. It also explicitly decided to work from “local active experiment files only,” not the full AllenSDK project listing.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by unique `mouse_id` values from the experiment metadata table, after the local-file and passive-session filters.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes describe `mouse_id` as the subject identifier and use it to build `subjects` and per-session `subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI treats each local `ophys_experiment_id` NWB file as one converted session. It does not group multiple imaging planes that share an `ophys_session_id`.

ii.
```python
for row in exp_table.itertuples(index=False):
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

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI states: “Treat each `ophys_experiment_id` file as one converted session,” justifying this as matching `BehaviorOphysExperiment` granularity and keeping a single imaging plane / neuron set / brain region per session.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI keeps only `go` or `catch` rows, excludes `aborted` and `auto_rewarded`, then defines each trial as the full interval from `start_time` to `stop_time`, discretized onto 30 Hz bin centers.

ii.
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
trials = trials.sort_values("id").reset_index(drop=True)

centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the AI used the processed NWB `trials` table as the “authoritative processed trial definition” and aligned all streams on a common 30 Hz grid within each trial.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is `(go | catch) & ~aborted & ~auto_rewarded`, plus skipping trials whose binned time grid is empty. Sessions are also skipped if they have fewer than 2 kept trials, no task presentations, or missing eye tracking.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    continue

presentations = get_task_presentations(f)
if presentations.empty:
    continue

if centers.size == 0:
    continue
```

iii. The AI justified this in the notes as “keep only contingent trials,” require pupil availability because pupil is a required decoder output, and require at least two usable trials per session for downstream decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection output, specifically `processing/ophys/event_detection/data`, together with its timestamps from `processing/ophys/event_detection/timestamps`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. In Step 4/5 of `CONVERSION_NOTES.md`, the AI explicitly chose event traces rather than dF/F because the paper’s neural analyses “explicitly use detected calcium events.”

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from native ophys timestamps onto per-trial 30 Hz bin centers. The output is transposed to neuron-by-time order. No additional normalization is applied.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time):
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)

neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI’s notes justify this as creating a common 30 Hz grid across sessions with different native ophys rates, while keeping alignment based on absolute ophys time.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter in the conversion code. The script counts all rows in `cell_specimen_table` and returns a constant brain-region index for all of them.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The notes argue that the local files already reflect upstream Allen processing and state that included local sessions had all listed cells valid, so no extra ROI filtering was added in the converter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial-local 30 Hz time bins spanning `start_time` to `stop_time`. Those bins are defined in absolute time, then event traces are sampled onto them from the ophys event timestamps.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    return centers[centers < (stop_time + 1e-9)]

neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes describe this as “align by absolute ophys time, then cut into trials” and say the 30 Hz bin centers are the common alignment grid for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `33.333...` ms bins. Yes: the code rebins/resamples neural, running, and pupil signals onto this grid by interpolation.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. The AI justified this in the notes as a way to unify single-plane and multi-plane sessions under one shared bin size while staying close to the behavior and eye-tracking rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from stimulus-presentation interval tables under `intervals/*`, using at least `start_time`, `stop_time`, `image_name`, `omitted`, `is_change`, and `trials_id`.

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
df = read_interval_table(group, columns)
```

iii. The notes say the AI chose stimulus-presentation tables to build time-varying image identity on the resampled trial grid, using the trial table only for cross-checks.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code initializes every bin as `"gray"`, then overwrites bins that fall inside each stimulus-presentation interval with the presented `image_name`, except omitted presentations, which remain `"gray"`. After collecting all image names globally, it maps them to integer codes.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
...
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The notes justify this by saying gray/omission periods should be encoded explicitly and that stimulus identity should be reconstructed from the actual presentation table rather than only from trial-level initial/change names.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same 30 Hz bin centers used for `neural`, using time-interval overlap between `centers` and each trial’s stimulus presentations.

ii.
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    image_identity[mask] = ...
```

iii. The AI’s notes say all streams were interpolated or sampled onto the same trial grid, so stimulus outputs and neural activity share one aligned time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag and the presentation intervals’ `start_time`/`stop_time`, restricted to the rows belonging to the current trial via `trials_id`.

ii.
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say the AI chose stimulus-presentation rows as the authoritative source for time-varying stimulus outputs and used the trial table mainly for trial segmentation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector and sets bins to `1` anywhere the 30 Hz bin center falls inside a presentation interval marked `is_change=True`.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as marking the change presentation itself on the common trial grid.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary in the conversion: `0` for `"no_change"` and `1` for `"change"`. No extra thresholding is applied.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
]
```

iii. The AI treats image change as a categorical binary output by construction.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change values are assigned on the same 30 Hz bin centers as the neural data, using the same per-trial `centers` vector and time-interval masks.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes repeatedly emphasize that all outputs are built on the same trial-local 30 Hz time base as `neural`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running group `processing/running/speed`, using both `timestamps` and `data`.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes describe this as using the processed NWB running-speed stream directly.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to each trial’s 30 Hz bin centers, pooled across sessions/trials in pass 1 to compute global quantile edges, and then digitized into five bins in pass 2.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes justify this as global equal-frequency discretization so the class meanings remain consistent across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is binned into five dataset-wide quantile bins using `compute_quantile_edges` and `digitize_with_edges`.

ii.
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    ...

def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
```

iii. The notes say these are “five equal-percentile bins” computed globally across the included data.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same `centers` vector used for neural resampling within each trial.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The AI justified this as a single common 30 Hz time grid for all streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking`, specifically `pupil_tracking/width`, `pupil_tracking/height`, and `eye_tracking/timestamps`.

ii.
```python
eye_group = f["acquisition"]["EyeTracking"]
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
diameter = 2.0 * np.maximum(width, height)
```

iii. The notes say the AI chose to compute pupil diameter from the raw width/height arrays in the NWB rather than using an AllenSDK convenience field.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes `2 * max(width, height)` per sample, fills NaNs by interpolation along the eye-tracking time axis, linearly interpolates to trial 30 Hz bin centers, and discretizes globally into five quantile bins.

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

iii. The notes justify this by requiring pupil availability as an output and handling blink-related missingness by interpolation within session.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five dataset-wide quantile bins using the same edge-computation and digitization logic used for running speed.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes explicitly describe “five equal-percentile bins” for pupil diameter across all included sessions/trials.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated to the same per-trial 30 Hz bin centers as the neural data.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The AI’s stated strategy is one shared alignment grid for neural and behavioral outputs.

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

iii. The notes treat these as the canonical mutually exclusive outcome labels for kept trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the four boolean outcome labels to integers `0..3`, then repeats the chosen label across all time bins of that trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes justify this as keeping every output in a consistent time-varying `(n_output, T)` representation, even for a static per-trial variable.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or unusable data by skipping sessions with missing eye tracking or no task presentations, skipping trials with empty time-bin grids, interpolating NaNs in pupil traces over time, clipping digitization inputs to the quantile-edge range, and requiring at least two usable trials per session.

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
if presentations.empty:
    continue
...
if centers.size == 0:
    continue
...
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
...
clipped = np.clip(values, edges[0], edges[-1])
```

iii. The notes justify these choices as necessary to keep pupil as a required output, avoid broken trial windows, and make global discretization robust.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by reading large NWB files and, in particular, reading/interpolating the event matrices. It also does two passes over the dataset, so file I/O happens twice.

ii.
```python
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. `CONVERSION_NOTES.md` explicitly lists the code as “I/O-heavy” and notes that the first pass for global binning forces each file to be read twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural interpolation itself was already vectorized, but the code still loops over sessions, trials, and per-trial stimulus-presentation rows. The per-trial `for row in trial_presentations.itertuples(...)` assignment for image outputs is a clear remaining non-vectorized loop.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    ...
    for trial in trials.itertuples(index=False):
        ...

for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. The notes say the AI intentionally vectorized event interpolation with `searchsorted` + broadcasting, but left the session/trial/presentation traversal in ordinary Python loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats session loading and some interpolation work across two passes. In pass 1 it opens each retained file, filters trials, and resamples running/pupil to trial bins to compute global quantiles; in pass 2 it opens the same files again and recomputes trial bins plus signal resampling for final conversion.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The AI states this directly in the notes: global binning requires a first pass, so “conversion reads each file twice.”

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and optionally plots diagnostic processing outputs that are not used downstream by the decoder. Even in normal conversion, pass 1 performs temporary per-trial running/pupil resampling solely to estimate global quantile edges and then discards those resampled traces.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
running_values.append(running_trial)
pupil_values.append(pupil_trial)
...
if show_processing and not plotted:
    make_processing_plot(...)
```

iii. The notes explicitly describe the two-pass design and the optional processing plots as validation and debugging aids rather than data kept for downstream analyses.
