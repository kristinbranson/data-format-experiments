# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly using `h5py` (rather than the AllenSDK high-level API, which was broken in its environment). It discovers available experiments by reading `ophys_experiment_table.csv`, then filters to only those with local NWB files on disk. It does NOT filter by `project_code == 'VisualBehavior'`; instead it filters out passive sessions (`~passive`). Data loading proceeds in two passes: Pass 1 reads each NWB to compute global running/pupil bin edges; Pass 2 re-reads each NWB to build the final converted trial data.

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

iii. The AI chose to read NWB files directly with `h5py` because the local AllenSDK/NWB stack could not instantiate these NWB files due to an `external_resources` version mismatch (documented in CONVERSION_NOTES Step 4). The AI filtered out passive sessions because they lack meaningful trial outcomes, but did not filter by `project_code`, potentially including experiments from VisualBehaviorMultiscope and other project codes.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV table, sorted alphabetically.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard unique identifier for each animal in the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (i.e., each NWB file / imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session. Each experiment file maps to one session in the converted output.

ii.
```python
# In get_local_session_metadata:
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ophys_session_id=int(row.ophys_session_id),
            ...
        )
    )

# In convert_session, each session corresponds to one experiment file:
with h5py.File(session.filepath, "r") as f:
    ...
```

iii. The AI documented this as a deliberate decision in CONVERSION_NOTES Step 5: "Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session."

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. Each trial spans from `start_time` to `stop_time`. The AI creates a 30 Hz grid of bin centers within each trial window rather than using native ophys frame indices. Trials with empty bins (stop_time <= start_time or no valid bins) are skipped.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
        return np.asarray([], dtype=np.float64)
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

iii. The AI chose a 30 Hz common grid to handle mixed native sampling rates (31 Hz single-plane, 11 Hz multiplane) while matching the behavior/eye-tracking rate (30 Hz).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only go and catch trials, excluding aborted and auto-rewarded trials: `(go | catch) & ~aborted & ~auto_rewarded`. Additionally, passive sessions are excluded entirely, and sessions missing eye tracking are excluded. Sessions with fewer than 2 usable trials are excluded. The AI does NOT filter on `change_time.notna()`.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
```

iii. The AI's CONVERSION_NOTES Step 4 states: "Use the NWB trials table directly as the authoritative processed trial definition, then filter to GO/CATCH and exclude aborted/auto-rewarded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection` in the NWB files — the extracted calcium event magnitudes — NOT from dF/F traces.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI explicitly chose events over dF/F based on the paper methods. CONVERSION_NOTES Step 4: "Neural signal choice: SDK exposes both `dff_traces` and `events`; paper subset analyses use calcium events... Resolution: Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`."

## 2-b. How is the `neural` data processed?

i. The event detection matrix (time × neurons) is linearly interpolated from native ophys timestamps onto a 30 Hz trial grid using vectorized searchsorted-based interpolation. The matrix is transposed to (neurons × time) format.

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

iii. The AI documented this in CONVERSION_NOTES Step 5: "Resample all streams to a common 30 Hz grid. This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. The AI relies on the NWB files already containing only valid ROIs (the Allen SDK pipeline filters invalid ROIs during NWB generation). Sessions without eye tracking are excluded at the session level.

ii. No explicit neuron filtering code. The cell count comes directly from the NWB:
```python
def get_cell_count_and_region_idx(f, region_index):
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 10: "Included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed), so event matrices matched converted neuron counts exactly."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, bin centers are generated at 30 Hz from `start_time` to `stop_time`. The event detection traces are then linearly interpolated onto these bin centers. This gives variable-length trials.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. CONVERSION_NOTES Step 5: "Align by absolute ophys time, then cut into trials: For each trial, create bin centers from trial start_time to stop_time at 30 Hz and sample/interpolate all streams onto that grid."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a 30 Hz grid (time bin size = 33.33 ms). This is explicitly set as a constant `DT = 1.0 / 30.0`. Native ophys rates (31 Hz single-plane, 11 Hz multiplane) are resampled via linear interpolation.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
'time_bin_size': TIME_BIN_MS,
```

iii. CONVERSION_NOTES Step 4: "Native sampling rates vary. Resample all streams to a common 30 Hz grid (33.333... ms bins). This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB file (`intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `trials_id` columns. It is NOT derived from the trial-level `initial_image_name`/`change_image_name` columns.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
        columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", ...]
        df = read_interval_table(group, columns)
```

iii. CONVERSION_NOTES Step 5: "Use stimulus presentation interval tables to build time-varying image identity and image-change signals on the resampled trial grid."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, the AI iterates over stimulus presentations that match the trial's `trials_id`. For each non-omitted presentation, the image name is mapped to an integer code at the matching time bins. During gray periods (ISI) and omitted stimuli, the code defaults to a `gray` category (index 0). A global sorted image name → integer mapping is built across all sessions.

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

iii. CONVERSION_NOTES Step 5: "Build per-bin categorical state on 30 Hz grid: actual image_name during image flashes; gray during gray/omitted periods."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid (`centers`) as the neural data. Presentation start/stop times are compared against the bin centers to determine which image is on screen at each time point.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. Using the same `centers` array ensures alignment with the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentation tables. When a presentation is marked `is_change == True`, the corresponding time bins are set to 1.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. CONVERSION_NOTES Step 5: "Binary per-bin trace: 1 during change-image flash interval, else 0."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is initialized to 0 for the full trial. For each stimulus presentation within the trial that has `is_change == True`, the corresponding time bins (where `centers` falls within the presentation's `[start_time, stop_time)`) are set to 1. This means the "change" indicator spans only the duration of the change image flash (~250 ms), not a longer window.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. The AI uses the stimulus presentation `is_change` flag which marks when the image identity actually changed from the previous non-omitted image. This captures only the flash duration, unlike the reference which uses a 750 ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
["no_change", "change"],
```

iii. N/A — binary by construction.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed on the same 30 Hz trial grid (`centers`) as the neural data.

ii. See 4-b code snippet using `centers` and `mask`.

iii. Same alignment mechanism as all other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which contains timestamps and speed values.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. This corresponds to the Allen SDK's `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`. Then it is discretized into 5 equal-percentile bins computed globally across all sessions. Bin edges are computed in Pass 1 from all finite values across all retained sessions.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
def compute_quantile_edges(values, nbins):
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    ...
    return edges

def digitize_with_edges(values, edges):
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```

iii. CONVERSION_NOTES Step 5: "Interpolate filtered running speed onto 30 Hz trial grid; discretize globally across included data into 5 equal-frequency bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0–4) using quantile-based edges computed globally. Values are clipped to the range of the edges before digitization. The bins represent equal-frequency percentile ranges across all included sessions.

ii.
```python
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Equal-percentile bins ensure roughly balanced class counts for decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid (`centers`) as the neural data before discretization.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Using the same `centers` guarantees frame-by-frame alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `height` arrays. The AI computes pupil diameter as `2 * max(width, height)`.

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

iii. CONVERSION_NOTES Step 5: "Compute pupil diameter as `2 * max(width, height)` after blink filtering."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After computing `2 * max(width, height)`, NaN values (from blinks/bad frames) are filled by linear interpolation over time using `fill_nan_by_time`. The result is then linearly interpolated to the 30 Hz trial grid and discretized into 5 equal-percentile bins globally.

ii.
```python
def fill_nan_by_time(time_axis, values):
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.all():
        return values
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values
```

iii. CONVERSION_NOTES Step 5: "Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — discretized into 5 equal-percentile bins (0–4) using globally computed quantile edges.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Same rationale as running speed: equal-frequency bins for balanced decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as the neural data before discretization.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same alignment mechanism as running speed and neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

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
    raise ValueError("Trial has no valid outcome label")
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast as a constant trace across all time bins in the trial.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The outcome is static per-trial as specified in the instructions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` in the NWB are excluded entirely (KeyError caught in Pass 1).
- **NaN pupil values**: Blink-related NaN values in pupil diameter are filled by linear interpolation over time before resampling.
- **Invalid trial times**: `build_trial_bins` returns empty for non-finite or non-positive-duration trials, which causes those trials to be skipped.
- **Failed sessions**: KeyError exceptions during session loading cause the session to be skipped with a warning.
- **Sessions with few trials**: Sessions with fewer than 2 usable trials are excluded (RuntimeError in convert_session or check in Pass 1).
- **Duplicate quantile edges**: If percentile computation yields duplicate edges, they are made strictly increasing with epsilon offsets.

ii.
```python
# Missing eye tracking
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")

# NaN pupil interpolation
def fill_nan_by_time(time_axis, values):
    ...
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])

# Invalid trials
if centers.size == 0:
    continue

# Few trials
if len(neural_trials) < 2:
    raise RuntimeError(...)
```

iii. CONVERSION_NOTES Step 5: "Require pupil availability at session level: Three active local files lack eye-tracking acquisition entirely; these sessions will be excluded."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading NWB files from disk with h5py, particularly the neural event detection matrices and behavioral data streams. The code reads each NWB file twice (Pass 1 for global statistics, Pass 2 for conversion), doubling the I/O cost.

ii. Pass 1 timing per session was ~0.36s; Pass 2 was ~1.86s; total conversion for 199 sessions was ~395s.

iii. CONVERSION_NOTES Step 7: "Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially, performing interpolation and stimulus trace construction for each trial individually. The inner loop over stimulus presentations for each trial (`for row in trial_presentations.itertuples`) could potentially be vectorized using interval-based broadcasting.

ii.
```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(...)
    neural_trial = linear_resample_matrix(ophys_time, events, centers)
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The AI noted the per-trial loop is not the main bottleneck compared to I/O, but the nested per-presentation loop adds overhead.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice — once in Pass 1 (to collect running/pupil values for global bin edge computation) and once in Pass 2 (to convert trials). This means running speed interpolation and pupil diameter computation are performed twice for every trial. The neural event matrix is only read in Pass 2.

ii.
```python
# Pass 1:
def collect_global_statistics(sessions):
    for idx, session in enumerate(sessions, start=1):
        with h5py.File(session.filepath, "r") as f:
            running_time, running_speed = get_running_data(f)
            pupil_time, pupil_diameter = get_pupil_data(f)
            for trial in trials.itertuples(index=False):
                running_trial = linear_resample_vector(running_time, running_speed, centers)
                pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)

# Pass 2:
def convert_session(session, ...):
    with h5py.File(session.filepath, "r") as f:
        running_time, running_speed = get_running_data(f)
        pupil_time, pupil_diameter = get_pupil_data(f)
        for trial_idx, trial in trials.iterrows():
            running_trial = linear_resample_vector(running_time, running_speed, centers)
            pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The two-pass design was chosen to compute global bin edges before the final conversion pass. The reference code avoids this by storing intermediate results in memory during a single pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The two-pass architecture means the running and pupil resampling in Pass 1 is purely for computing bin edges and is discarded — the same computation is repeated in Pass 2. Additionally, the `stimulus_block_name` filtering and full stimulus presentation table parsing is performed in both passes (for image name collection in Pass 1, and for output construction in Pass 2).

ii. See 9-c code snippets showing duplicate running/pupil processing.

iii. This is an efficiency trade-off rather than a correctness issue. The AI chose to avoid storing all per-trial running/pupil values in memory across passes.
