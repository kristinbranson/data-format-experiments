# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than the Allen SDK's Python API. It discovers available sessions from `ophys_experiment_table.csv` in the project metadata directory, cross-references against NWB files actually present on disk, and reads each NWB file's HDF5 groups directly. It performs two passes: pass 1 collects global statistics (running/pupil bin edges, image vocabulary) without loading neural data; pass 2 loads neural data and constructs per-trial arrays.

ii.
```python
def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    ...
```
```python
def read_session_raw(session: SessionInfo, load_events: bool = True) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        ...
```

iii. The AI justified using `h5py` directly because the local `pynwb/hdmf` environment could not instantiate the NWB 2.6.0 files via `BehaviorOphysExperiment.from_nwb_path` due to an `external_resources` abstract-method mismatch. The AI documented this in CONVERSION_NOTES Step 5 Key Decision 9.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata table CSV. Each unique `mouse_id` string becomes one subject.

ii.
```python
mouse_id=str(int(row.mouse_id)),
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen SDK metadata. This matches the reference approach.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one NWB file = one imaging plane) is treated as a separate session. The AI does NOT group multiple experiments from the same `ophys_session_id` into a single session.

ii.
```python
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
```

iii. The AI documented this in CONVERSION_NOTES Step 4: "Treat each NWB experiment file as one decoder session because neural traces are experiment-specific." For the `VisualBehavior` project (single-plane), each `ophys_session_id` maps to one experiment, so this is functionally equivalent to the reference approach of grouping by `ophys_session_id`. However, the AI does not filter by `project_code == 'VisualBehavior'`; it instead filters out passive sessions and sessions missing eye tracking.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table in each NWB file. The AI keeps trials where `(go | catch) & not aborted & not auto_rewarded`. Each trial spans from `start_time` to `stop_time`, giving variable-length trials.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The AI justified this filter as matching the standard go/catch trial structure for the change-detection task, consistent with the SDK trial definitions and the instructions to "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only go or catch trials, (2) excluding aborted trials, (3) excluding auto-rewarded trials. Sessions with fewer than 2 valid trials are skipped. Additionally, 3 active sessions missing eye-tracking data entirely are excluded before processing. Passive sessions are excluded entirely.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ...")
    continue
...
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The AI justified passive-session exclusion because passive sessions produce degenerate trial outcomes (all misses/correct rejects) and the decoder task requires meaningful trial outcomes. Sessions missing eye tracking were excluded to avoid errors in pupil diameter computation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` (precomputed calcium events), NOT from `dff_traces` (dF/F).

ii.
```python
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI justified this choice in CONVERSION_NOTES Step 4: "Use precomputed `events` as the neural signal for conversion because that best matches the analysis paper and avoids diverging from reference processing." The strategy paper explicitly states its neural analyses used "detected calcium events."

## 2-b. How is the `neural` data processed?

i. Calcium event traces are linearly interpolated from native ophys timestamps to a common 30 Hz time grid (1/30 s bins). The interpolation is done per-trial using custom matrix interpolation. Additionally, if `valid_roi` is present in the cell_specimen_table and differs from the event-trace width, invalid ROIs are filtered out.

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
```

iii. The AI justified the 30 Hz grid as matching the strategy paper's approach of interpolating onto common 30 Hz timestamps. The valid_roi filtering mirrors the SDK's `exclude_invalid_rois=True` behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are filtered using the `valid_roi` flag from the cell_specimen_table in the NWB file. Only neurons where `valid_roi == True` are kept.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. This matches the SDK's default behavior of `exclude_invalid_rois=True` in `CellSpecimens`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. For each trial, a 30 Hz time grid is constructed from `start_time` to `stop_time`, and the neural event traces are linearly interpolated onto this grid.

ii.
```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt

grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The AI uses trial start as the alignment event, consistent with the instructions to "Temporally align based on ophys timestamp." The temporal_alignment_event metadata is set to "trial start".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a fixed 30 Hz time grid (33.33 ms bins), regardless of the native ophys frame rate. All data streams (neural, running, pupil, stimulus) are interpolated onto this common grid. This constitutes temporal rebinning from the native ophys rate (~11 Hz for multi-plane, ~31 Hz for single-plane).

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The AI justified this in CONVERSION_NOTES Step 5: "The strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps relative to behavioral events, and behavior/eye tracking are naturally 30 Hz, so 30 Hz is the most defensible common grid."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `is_change` columns from the active task presentation group.

ii.
```python
stim_group = choose_task_presentation_group(h5f)
stim = read_interval_table(
    stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted", "trials_id", "active", "flashes_since_change"],
)
```

iii. The AI chose to use the stimulus presentations table rather than the trials table's `initial_image_name`/`change_image_name` to capture the actual stimulus displayed at each timepoint, including gray-screen periods between flashes and omitted stimuli.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is computed as a piecewise-constant categorical signal on the 30 Hz grid. For each time bin, the code finds which stimulus presentation interval contains it. If the bin falls within a non-omitted image presentation, it gets the image's integer code; otherwise it gets the `gray` code. A global vocabulary includes `gray` plus all unique image names.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    ...
```

iii. The AI justified including `gray` as an explicit category because the task includes 500 ms gray periods between flashes and omissions extend gray instead of showing an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz time grid as the neural data, using stimulus presentation start/stop times to determine which image is displayed at each bin.

ii.
```python
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Same 30 Hz grid ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column and the `start_time`/`stop_time` of stimulus presentations.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
    ...
```

iii. The AI used the stimulus presentations table's `is_change` flag rather than the trials table's `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary time-varying signal. It is 1 during the time bins that fall within the changed-image presentation window (where `is_change=True` and not omitted), and 0 elsewhere. The window duration is one stimulus flash (250 ms image on screen).

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    ...
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    sub_idx = idx_valid[in_interval]
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    assign = np.flatnonzero(valid)[in_interval]
    codes[assign] = changed.astype(np.int64)
    return codes
```

iii. The AI initially used a single-bin impulse at `change_time` but found it too sparse for decoding. It then switched to using the full stimulus presentation window where `is_change=True`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1), so no thresholding is applied. It is 1 during the change-image presentation interval and 0 otherwise.

ii. See 4-b.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz grid as neural data and other outputs.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same alignment approach as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This corresponds to the SDK's `running_speed` attribute, read directly from HDF5.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`, then discretized into 5 quintile bins using global bin edges computed from all valid trials across all sessions.

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Linear interpolation preserves the signal. Global quintile binning ensures consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.nanpercentile` at [20, 40, 60, 80] to compute global bin edges, then `np.digitize` to assign bins. The edges are computed with a robustness fix: if an edge equals or is less than the previous edge, it is bumped by 1e-6.

ii.
```python
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles

def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The robust edge computation handles cases where percentiles collapse (e.g., many identical values).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz grid as the neural data.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Same grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height`, combined as `max(width, height)`. Timestamps come from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The AI used `max(width, height)` as a proxy for pupil diameter from the ellipse fit.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The pupil diameter is computed as `max(width, height)`, then interpolated to the 30 Hz grid using only valid (finite) samples (implicitly handling blinks/NaNs). Then discretized into 5 quintile bins using global edges.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. By filtering to finite values before interpolation, blink frames (which are typically NaN) are excluded, and the interpolation bridges across blink gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 quintile bins using global edges from `np.nanpercentile` at [20, 40, 60, 80], applied with `np.digitize`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same justification as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 30 Hz grid as neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the NWB `intervals/trials` table.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping, then broadcast as a constant value across all time bins in the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The outcome is static per trial but represented as time-varying (repeated) to match the output format requirement.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions lacking the `EyeTracking` group are excluded before processing.
- **Invalid ROIs**: Neurons with `valid_roi == False` are filtered out.
- **Blink/NaN in pupil**: `interpolate_pupil` uses only finite values, interpolating across blink gaps.
- **Sessions with few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Edge cases in discretization**: `robust_quintile_edges` bumps degenerate percentile edges by 1e-6.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
...
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. The AI documented these in CONVERSION_NOTES Step 10. The missing eye tracking issue was discovered during the first full conversion attempt and fixed by pre-filtering.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the NWB file I/O, particularly reading neural event traces and performing the two-pass conversion (each session's NWB file is read twice: once in pass 1 for global statistics, once in pass 2 for full conversion).

ii. N/A (timing is printed but not in a single code snippet)

iii. The AI's CONVERSION_NOTES Step 7 estimated pass 2 as the dominant cost at ~1.35s/session, with total conversion at ~356s for 199 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within each session iterates sequentially over trials. The `stimulus_identity_codes` and `stimulus_change_codes` functions already use vectorized `np.searchsorted` operations. The `interpolate_matrix` function vectorizes across neurons.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    neural_trial = interpolate_matrix(...)
    running_cont = interpolate_vector(...)
    ...
```

iii. The per-trial loop is necessary because each trial has a different time grid. Within each trial, operations are vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is read twice: once in pass 1 (`collect_global_statistics`) to compute global bin edges and image vocabulary without loading neural data, and once in pass 2 (`convert_sessions`) to load neural data and construct per-trial arrays. Running speed and pupil are interpolated in both passes.

ii.
```python
# Pass 1
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(sessions)
# Pass 2
data = convert_sessions(sessions=sessions, running_edges=running_edges, ...)
```

iii. The two-pass design avoids storing all neural arrays in memory during global statistics computation, trading I/O time for memory efficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads several stimulus table columns (`flashes_since_change`, `trials_id`, `active`) that are used for filtering/selection but not directly included in the output. The `session_type` is stored in `SessionInfo` but only used for logging. Processing plots are only generated when `--show-processing` is set, so they are not wasteful by default.

ii. N/A

iii. The extra columns read from the stimulus table are cheap to load and useful for validation/debugging.
