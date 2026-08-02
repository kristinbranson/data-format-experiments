# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK cache path from the reference. It reads the local `ophys_experiment_table.csv`, intersects that metadata with NWB files present on disk, excludes passive experiments, and then opens each NWB directly with `h5py`. It performs two passes: a first pass to collect global running/pupil statistics and a second pass to build converted trial arrays.

ii.
```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)
    ...
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()

...
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as a response to the local environment: only a subset of NWB files is present locally, passive sessions should be excluded, and direct `h5py` reads were used because it believed the high-level AllenSDK/NWB loading path was broken in this environment.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values taken from the retained local experiment metadata after filtering.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes say `mouse_id` is the natural subject identifier from the Allen metadata tables, and the trajectory shows the AI intentionally limited the subject list to mice represented in the kept local sessions.

## 1-c. How are the data split into sessions?

i. Each local `ophys_experiment_id` NWB file is treated as one converted session. The AI does not group multiple experiments by `ophys_session_id`.

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

iii. `CONVERSION_NOTES.md` explicitly calls this a key decision: “Treat each `ophys_experiment_id` file as one converted session,” justified by matching the granularity of the local NWB files and `BehaviorOphysExperiment`.

## 1-d. How are the data split into trials?

i. Trials are read from the processed NWB `intervals/trials` table. For each kept trial, the AI builds a variable-length 30 Hz grid from `start_time` to `stop_time` and then resamples all streams onto that grid.

ii.
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the NWB `trials` table is the authoritative Allen-processed trial definition and should be used directly instead of re-deriving trial logic.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if `(go | catch) & ~aborted & ~auto_rewarded`. Trials producing no bin centers are skipped. Sessions are also dropped if they end up with fewer than 2 usable trials, if eye tracking is missing, or if there are no task presentations.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    ...
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
```

iii. The notes justify this as matching contingent-trial logic from Allen trial masks while also enforcing decoder requirements like at least two trials and required pupil/task data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/{timestamps,data}` in each NWB file, not from dF/F traces.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The notes explicitly say the AI chose “raw event magnitude traces” because it believed the paper’s neural analyses used calcium events rather than raw dF/F.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from native ophys timestamps onto each trial’s 30 Hz bin centers, then transposed to neuron-by-time format.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time):
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)

...
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes justify this as putting all sessions on a common 30 Hz trial grid while keeping alignment on absolute ophys time.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no explicit neuron-quality filter. It uses every row in `event_detection/data` and assigns brain-region labels by counting all rows in `cell_specimen_table`.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The notes discuss Allen ROI filtering in the SDK/whitepaper, but the implemented code does not reproduce that filtering and gives no separate code-level justification beyond using the processed NWB contents directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. For each trial, the AI builds bin centers from `start_time` to `stop_time` and samples event traces at those times.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes say the converter should “align by absolute ophys time, then cut into trials,” with “trial start” stored as the metadata alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `33.333...` ms bins. Yes, temporal resampling/rebinning is applied to all streams.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. The notes say this was chosen to unify mixed native sampling rates and keep behavior/eye-tracking and neural data on the same common grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from task stimulus-presentation interval tables, especially `image_name`, `omitted`, `start_time`, `stop_time`, and `trials_id`.

ii.
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name", ...
]
...
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes justify using stimulus-presentation tables because they expose flashed image identity, omissions, and exact presentation intervals directly on the task timeline.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI creates a global categorical mapping over all image names plus an extra `gray` category. Within each trial, bins default to `gray`, then bins overlapping non-omitted stimulus presentations are set to that presentation’s image code.

ii.
```python
image_names = set(["gray"])
...
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes explicitly say the converter should “encode gray/omission periods explicitly” and use time-varying image identity from stimulus presentations rather than only trial-level image names.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is written on the same 30 Hz trial grid used for neural data, using time masks from stimulus presentation intervals.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes justify this as using one shared ophys-aligned trial grid for all outputs and neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation field `is_change`, not from trial `go` plus `change_time`.

ii.
```python
columns = [..., "is_change", ...]
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say the converter should use presentation rows to build the change signal because they explicitly encode which flash is a change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI initializes a zero vector for the trial and sets bins to `1` for any 30 Hz bin falling inside a stimulus-presentation interval marked `is_change`.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as a direct reconstruction from raw presentation intervals rather than using a hand-coded fixed-duration window after change time.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary with categories `0 = no_change` and `1 = change`.

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

iii. This follows the decoder requirement that image change be a binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same 30 Hz trial grid as neural data, using the same `centers` array and presentation-interval masks.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes repeatedly justify the use of one common trial grid for neural, running, pupil, and stimulus outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/{timestamps,data}` in the NWB.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes say this corresponds to the Allen processed running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz grid, then discretized into five global quantile bins computed in pass 1 across all retained sessions/trials.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes justify global percentile binning so category meanings are consistent across the full converted dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is digitized into 5 quantile bins using dataset-wide edges.

ii.
```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
running_edges = compute_quantile_edges(..., 5)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes explicitly say running speed should be discretized globally into five equal-percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled onto the same trial `centers` array used for neural interpolation.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes justify this as shared alignment on a single common 30 Hz trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/{width,height}` plus eye-tracking timestamps.

ii.
```python
eye_group = f["acquisition"]["EyeTracking"]
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The notes say the AI wanted an explicit diameter-like quantity from pupil geometry rather than using a precomputed width field from the SDK table.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes `diameter = 2 * max(width, height)`, linearly fills NaNs across time, resamples onto the 30 Hz trial grid, and then discretizes globally into five quantile bins.

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

iii. The notes justify this with a claim that sessions missing `EyeTracking` should be excluded and that remaining blink-related missingness can be handled by interpolation before discretization.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is digitized into 5 global quantile bins.

ii.
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_quantile_edges(..., 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes say pupil diameter should be discretized the same way as running speed so class semantics are stable across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is resampled onto the same trial `centers` array used for neural interpolation.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes justify this as part of the single common 30 Hz trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trial table.

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

iii. The notes treat these as the canonical Allen trial-outcome labels for contingent trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps those booleans to indices `0..3` in fixed order and repeats the resulting category across all time bins in the trial.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes explicitly say all outputs should share one `(n_output, T)` style representation, so trial outcome is made time-constant within each trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing `EyeTracking` causes the entire session to be skipped. Empty or invalid trial windows are skipped. Pupil NaNs are linearly interpolated within a session. Duplicate quantile edges are perturbed upward by machine epsilon. If no sessions survive pass 1, the script raises an error.

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
if centers.size == 0:
    continue
...
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
...
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. The notes justify skipping sessions without required pupil output and interpolating remaining missing pupil values as a pragmatic way to keep most active sessions.

## 9-a. What are the most time-consuming steps of the code?

i. The code is dominated by repeated NWB I/O and interpolation across two full passes. Pass 1 reopens every kept session to compute global running/pupil statistics; pass 2 reopens them again to convert trials and resample neural, running, pupil, and stimulus outputs.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The trajectory explicitly describes pass 1 and pass 2 as the main runtime and repeatedly notes that the work is I/O-bound on opening many NWB files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidates are the per-trial loops in `collect_global_statistics()` and `convert_session()`, plus the nested per-presentation loop used to paint `image_identity` and `image_change` masks into each trial.

ii.
```python
for trial in trials.itertuples(index=False):
    ...

for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. The notes do not emphasize vectorization, but the implementation clearly relies on repeated Python-level loops for trial-by-trial and flash-by-flash work.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats reading the same sessions twice, recomputes trial bin centers in both passes, and resamples running/pupil once during pass 1 for global bin-edge estimation and again during pass 2 for the final stored outputs.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    with h5py.File(session.filepath, "r") as f:
        ...
        for trial in trials.itertuples(index=False):
            centers = build_trial_bins(...)
            running_trial = linear_resample_vector(...)
            pupil_trial = linear_resample_vector(...)

...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly describe the converter as a two-pass workflow and justify the duplication as necessary to get dataset-wide quantile edges before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads several metadata fields into `SessionMeta` that are not used downstream, reads `initial_image_name` and `change_image_name` from the trial table but never uses them in output construction, and optionally generates diagnostic plots that are not part of the final pickle. More broadly, pass-1 per-trial resampling work is discarded after only contributing to bin-edge estimation.

ii.
```python
class SessionMeta:
    ophys_experiment_id: int
    ophys_session_id: int
    behavior_session_id: int
    mouse_id: str
    targeted_structure: str
    session_type: str
    experience_level: str
    project_code: str
    filepath: Path

...
columns = [..., "change_time", "initial_image_name", "change_image_name"]
...
if show_processing and not plotted:
    make_processing_plot(...)
```

iii. The notes frame the plotting and extra metadata as validation/documentation support, but those pieces do not affect the converted dataset itself.
