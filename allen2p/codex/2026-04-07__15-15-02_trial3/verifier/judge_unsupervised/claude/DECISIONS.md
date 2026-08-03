# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV metadata table (`ophys_experiment_table.csv`) to discover all available experiments, then filters to only locally available NWB files. It filters out passive sessions (`exp_table[~exp_table["passive"]]`). Each NWB file is opened individually with `h5py` (not the AllenSDK high-level loader) and the relevant data groups are read directly. A two-pass approach is used: Pass 1 collects global running/pupil statistics and filters sessions; Pass 2 converts each session's trials.

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
    exp_table = exp_table.sort_values("ophys_experiment_id")
```

iii. The AI justified using `h5py` directly because the local AllenSDK/NWB stack could not instantiate the NWB files due to version/environment incompatibilities. It documented this in CONVERSION_NOTES.md Step 4 as an "Environment mismatch" discrepancy.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the `mouse_id` column in the experiment metadata table. A sorted unique list of mouse IDs across all kept sessions forms the `subjects` list. Each session is mapped to its subject via `subject_to_idx`.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
# ...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The AI noted that each experiment file has a `mouse_id` in the metadata table, which is used as the subject identifier. This matches the AllenSDK convention.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one NWB file = one imaging plane) is treated as a separate session. Only active (non-passive) experiments with locally available NWB files and valid eye-tracking data are included. This yields 199 sessions from the 202 active NWB files (3 excluded due to missing EyeTracking).

ii.
```python
# In get_local_session_metadata:
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ...
            filepath=available_files[int(row.ophys_experiment_id)],
        )
    )
```

iii. The AI documented this in CONVERSION_NOTES.md Step 5 Key Decision #2: "Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session."

## 1-d. How are the data split into trials?

i. Trials are defined by the NWB `intervals/trials` table within each session. The trial `start_time` and `stop_time` columns define the temporal extent of each trial. Trials are filtered (see 1-e) and then sorted by `id`.

ii.
```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id", "start_time", "stop_time", "go", "catch", "aborted",
        "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "initial_image_name", "change_image_name",
    ]
    trials = read_interval_table(f["intervals"]["trials"], columns)
```

iii. The AI noted that the NWB file already stores the Allen-processed trial table, so trial segmentation follows the reference processing by using these pre-computed boundaries rather than re-deriving them.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, trials with zero-length time windows are skipped (via `build_trial_bins` returning empty). Sessions with fewer than 2 kept trials are excluded entirely.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
# ...
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
```

iii. The AI justified this filtering based on the instructions ("Include both the Go and Catch trials, but exclude the Aborted and Auto-rewarded trials") and reference code (`trial_masks.contingent_trials` defines contingent trials as GO and CATCH only).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `processing/ophys/event_detection/data` in the NWB file, which contains detected calcium event magnitudes. The corresponding timestamps come from `processing/ophys/event_detection/timestamps`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI documented this choice in CONVERSION_NOTES.md: "Paper analyses frequently use discrete calcium events rather than raw dF/F" and "Use raw event magnitudes, not `filtered_events` (visualization-only) and do not recompute dF/F."

## 2-b. How is the `neural` data processed?

i. The raw event detection data (shape: time x neurons) is linearly interpolated from native ophys timestamps onto a common 30 Hz trial grid using vectorized `searchsorted` + broadcasting interpolation. The result is transposed to (neurons x time) format.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time):
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1
    t0 = src_time[idx_lo]
    t1 = src_time[idx_hi]
    denom = np.where(t1 > t0, t1 - t0, 1.0)
    w = ((dst_time - t0) / denom).astype(np.float32)
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

iii. The AI chose to resample to 30 Hz to create a common bin size across all sessions, since native rates vary (31 Hz single-plane, 11 Hz multiplane). The interpolation approach is vectorized for efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level filtering is applied in the conversion code. The AI relies on the NWB files already containing only valid ROIs (verified that all ROIs in the local NWB files have `valid_roi == True`). Sessions missing EyeTracking are excluded entirely (3 sessions).

ii.
```python
# No explicit valid_roi filtering code exists.
# The get_cell_count_and_region_idx function simply counts all cells:
def get_cell_count_and_region_idx(f, region_index):
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The AI noted in CONVERSION_NOTES.md Step 10: "Included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed), so event matrices matched converted neuron counts exactly."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to "trial start" - the `start_time` of each trial from the NWB trials table. A 30 Hz grid of bin centers is created from `start_time` to `stop_time`, and neural event traces are interpolated onto this grid.

ii.
```python
def build_trial_bins(start_time, stop_time):
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

iii. The AI documented: `temporal_alignment_event: "trial start"` and `off_start: 0.0`, meaning data starts at the trial start time. The instructions say "Temporally align based on ophys timestamp," which the AI interpreted as using ophys timestamps as the neural reference stream for interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a 30 Hz grid (DT = 1/30 s = ~33.33 ms bins). This involves rebinning from the native ophys rate (~31 Hz for single-plane, ~11 Hz for multiplane) via linear interpolation.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

iii. The AI justified this choice: "Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB file (e.g., `intervals/Natural_Images_*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `trials_id` columns.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        # ...
        columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", ...]
```

iii. The AI documented: "Use stimulus presentation interval tables to build time-varying image identity and image-change signals on the resampled trial grid."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, a trace is initialized to the "gray" category index. Then, for each stimulus presentation within the trial (matched by `trials_id`), timepoints within the presentation's `[start_time, stop_time)` window are assigned the corresponding image index. Omitted flashes are mapped to "gray".

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The AI encoded gray/omission periods explicitly with a "gray" category, noting the instructions ask for "image identity of the image presented during the non-grey screen."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed directly on the same 30 Hz trial grid (`centers`) as the neural data, so no separate alignment step is needed. The presentation times are in absolute time (same clock as ophys timestamps), and the `centers` array indexes into these presentation intervals.

ii.
```python
# Same centers used for neural and image identity:
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
# Neural:
neural_trial = linear_resample_matrix(ophys_time, events, centers)
# Image identity: computed on same centers array
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. No explicit justification needed - the alignment is inherent in using the same time grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentation interval tables, along with `start_time` and `stop_time` of each presentation.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI used the pre-computed `is_change` flag from the NWB stimulus presentations, which corresponds to the AllenSDK's `is_change_event` logic.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is initialized to 0 for the full trial. For each stimulus presentation within the trial that has `is_change == True`, the corresponding time bins are set to 1.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. The AI interpreted "Have value of 1 right after a change in image identity" as setting 1 during the entire change-image flash interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. It is stored as a two-category variable with values `["no_change", "change"]`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. No additional thresholding logic; the binary nature follows directly from the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed directly on the same 30 Hz trial grid (`centers`) as the neural data.

ii. See 3-c code snippet - same `centers` array is used.

iii. Alignment is inherent in using the shared time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which contains the filtered running speed data and timestamps.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The AI noted this corresponds to `RunningSpeed.from_stimulus_file` in the reference code and uses the filtered running speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps onto the 30 Hz trial grid, then digitized into 5 equal-percentile bins using globally computed bin edges.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Global binning ensures consistent bin semantics across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile (quintile) bins computed globally across all included sessions. Bin edges are computed from all finite running speed values across all kept trials in Pass 1.

ii.
```python
def compute_quantile_edges(values, nbins):
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    if np.unique(edges).size < edges.size:
        eps = np.finfo(np.float64).eps
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
    return edges

running_all = np.concatenate(running_values).astype(np.float64, copy=False)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The AI noted: "Five equal-percentile bins will be computed from all finite samples across all included sessions/trials, not per session, so class semantics are consistent dataset-wide."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid as neural data, ensuring temporal alignment.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same alignment approach as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height`, with timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
    diameter = fill_nan_by_time(timestamps, diameter)
    return timestamps, diameter
```

iii. The AI checked the eye-tracking processing code and decided to compute diameter as `2 * max(width, height)`, treating width/height as semi-axes (radii) of the pupil ellipse fit.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `2 * max(width, height)`. NaN values (from blinks/missing data) are filled via linear time interpolation. The diameter is then linearly interpolated onto the 30 Hz trial grid and digitized into 5 equal-percentile bins using globally computed edges.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
# ...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The AI noted: "Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs before discretization."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: discretized into 5 equal-percentile bins using globally computed bin edges from all finite pupil diameter values across all kept trials.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. Same global binning rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as neural data.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same alignment approach as all other streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the mutually exclusive boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]): return 0
    if bool(trial_row["miss"]): return 1
    if bool(trial_row["false_alarm"]): return 2
    if bool(trial_row["correct_reject"]): return 3
    raise ValueError("Trial has no valid outcome label")
```

iii. The AI noted these correspond to the mutually exclusive outcome flags defined in `Trial._get_trial_data` in the reference code.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is determined per trial as a single categorical value (0-3), then broadcast as a constant trace across all time bins in the trial. This makes it "static per-trial" as required but stored in the same (n_output, n_timepoints) format as other outputs.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
output_trial = np.vstack([
    image_identity, image_change, running_bin, pupil_bin, outcome_trace,
])
```

iii. The AI documented: "trial_outcome will be repeated across bins within a trial as a constant categorical trace to keep one consistent (n_output, T) format."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing data are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` acquisition are excluded entirely (3 sessions skipped).
- **NaN pupil values**: Interpolated via `fill_nan_by_time` using linear interpolation across time.
- **Zero-length trials**: Skipped via `build_trial_bins` returning empty array.
- **All-zero neural trials**: Retained (documented as arising from sparse event detections in source data, not conversion errors).
- **Missing stimulus presentations**: Gracefully handled; if no presentations found, session is skipped.

ii.
```python
def fill_nan_by_time(time_axis, values):
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("All values are NaN")
    if finite.all():
        return values
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values

# Missing eye tracking:
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

iii. The AI investigated all-zero neural trials by loading raw NWB data and confirmed the zeros exist in the source data.

## 9-a. What are the most time-consuming steps of the code?

i. The two-pass structure means each NWB file is read twice (once for global statistics, once for conversion). The full conversion took ~395 seconds for 199 sessions. Pass 1 (global statistics) and Pass 2 (conversion) each involve loading large event matrices and running/pupil data from disk. The main I/O bottleneck is reading the event_detection matrix from each NWB file.

ii.
```python
# Pass 1: ~0.5s per session
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
# Pass 2: ~1.9s per session
neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The AI documented timing: "Pass 1 bin-stat collection ~0.36 s/session, Pass 2 conversion ~1.86 s/session."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several inner loops iterate over individual trials and stimulus presentations:
- The per-trial loop in `convert_session` iterates over each trial row and processes it individually.
- The inner loop over stimulus presentations per trial applies image identity masks one row at a time.
- The `collect_global_statistics` function loops over trials per session to collect running/pupil values.

ii.
```python
# Per-trial loop in convert_session:
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(...)
    neural_trial = linear_resample_matrix(ophys_time, events, centers)
    # ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. The AI acknowledged I/O as the main bottleneck rather than computation, and vectorized the neural interpolation step but not the trial-level or presentation-level loops.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in `collect_global_statistics` (Pass 1) for running/pupil bin edges, and once in `convert_session` (Pass 2) for the actual conversion. In both passes, it loads the trial table, stimulus presentations, running speed, and pupil data.

ii.
```python
# Pass 1 (collect_global_statistics):
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)

# Pass 2 (convert_session):
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. The AI documented this: "Global binning requires a first pass over sessions, so conversion reads each file twice."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes omitted stimulus presentations (setting them to "gray"), which results in the same value as the default initialization. The code also collects image names from all sessions in Pass 1 even though image names could potentially be collected in a simpler way. The `decode_str_array` function handles multiple string encoding cases that may not all be encountered. The processing plots (when enabled) add overhead that is only used for visual inspection.

ii.
```python
# Omitted flashes are redundant since default is already "gray":
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
```

iii. The AI documented that the `--show-processing` mode is optional and only runs for up to 2 sessions.
