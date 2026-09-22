# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than the Allen SDK's `VisualBehaviorOphysProjectCache`. It discovers sessions from `ophys_experiment_table.csv` in the project metadata directory, filters to only those experiments whose NWB files exist on disk, and further filters out passive sessions. Each NWB file is read with `h5py.File()` to extract trials, neural data, running speed, pupil tracking, and stimulus presentations.

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

def read_session_raw(session: SessionInfo, load_events: bool = True) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        ...
```

iii. The AI documented in CONVERSION_NOTES.md (Step 5, Key Decision 9) that the local `pynwb/hdmf` stack could not instantiate the NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` due to an `external_resources` abstract-method mismatch. Direct HDF5 reads mirror the SDK field definitions explicitly.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment table CSV. Each unique mouse_id is assigned an index as sessions are processed.

ii.
```python
subject_to_idx: Dict[str, int] = {}
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The `mouse_id` field from the metadata CSV is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one NWB file, corresponding to one imaging plane) is treated as a separate session. This differs from grouping by `ophys_session_id`, where multiple imaging planes from a single recording session would be combined.

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

iii. The AI documented in CONVERSION_NOTES.md (Step 4, Discrepancies) the decision to treat each NWB experiment file as one decoder session because neural traces are experiment-specific. The AI also noted that passive sessions are excluded (Key Decision 1) because the decoder task requires trial outcome and passive sessions produce degenerate outcomes.

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB `intervals/trials` table. The AI filters to keep only go and catch trials that are not aborted and not auto-rewarded. Each trial's time window spans from `start_time` to `stop_time`, resampled onto a common 30 Hz grid.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The AI followed the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded trials. The trial boundaries come from the NWB trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only go or catch trials, (2) excluding aborted trials, (3) excluding auto-rewarded trials. Sessions with fewer than 2 valid trials are skipped. Sessions missing eye-tracking pupil data are excluded entirely.

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

iii. The AI documented the exclusion of 3 active sessions missing eye tracking entirely (`795953296`, `806456687`, `833631914`). The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` (precomputed calcium events), not from dF/F traces.

ii.
```python
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI documented in CONVERSION_NOTES.md (Step 4, Step 5 Key Decision 2) that the strategy paper explicitly states its neural analyses used detected calcium events, not dF/F. The AI chose events to best match the reference analyses.

## 2-b. How is the `neural` data processed?

i. Neural event traces are linearly interpolated from the native ophys timestamps onto a common 30 Hz grid defined by `session_grid(start, stop)` with `TIME_BIN_SIZE_S = 1/30`. The interpolation is done per-trial using a custom `interpolate_matrix` function. If `valid_roi` is present in the cell specimen table and differs from the event trace width, invalid ROIs are filtered out.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0

def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt

def interpolate_matrix(source_t, source_values, query_t):
    ...

neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md (Step 5 Key Decision 3) that the strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps, and behavior/eye tracking are naturally 30 Hz, making 30 Hz the most defensible common grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` mask from the cell specimen table in the NWB file. If the valid_roi count differs from the event trace width, only valid ROIs are retained.

ii.
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
    n_neurons = events.shape[1]
```

iii. The AI documented in CONVERSION_NOTES.md (Step 1 notes) that the SDK defaults to `exclude_invalid_rois=True` and the `CellSpecimens` class filters to `valid_roi == True`. The AI replicates this filtering via direct h5py reads.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. A 30 Hz grid is created from `start_time` to `stop_time` of each trial, and the neural event traces are interpolated onto this grid.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The instructions say to "temporally align based on ophys timestamp." The AI uses `ophys_timestamps` as the source time base and interpolates onto per-trial grids anchored to trial start/stop times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a common 30 Hz grid (33.33 ms bins). This involves temporal rebinning via linear interpolation from the native ophys frame rate (~11 Hz for multi-plane or ~31 Hz for single-plane) to 30 Hz.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0

def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The AI justified 30 Hz as the common grid because the strategy paper interpolates to 30 Hz for its analyses, and behavior/eye tracking are naturally 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (the active task presentation group in the NWB file), using `image_name`, `start_time`, `stop_time`, and `omitted` columns. A `gray` category is used for inter-stimulus intervals and omitted stimuli.

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

iii. The AI documented in CONVERSION_NOTES.md (Step 5 Key Decision 6) that a `gray` category is needed because the task includes 500 ms gray periods between stimuli and omissions extend the gray period.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each time bin on the 30 Hz grid is assigned an image code: if the bin falls within a non-omitted stimulus presentation window, it gets the image name's code; otherwise it gets the `gray` code. The global image vocabulary is `["gray"] + sorted(unique_image_names)`.

ii.
```python
image_values = ["gray"] + sorted(image_names)
image_to_code = {name: idx for idx, name in enumerate(image_values)}
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. Using the stimulus presentations table allows precise timing of image on/off transitions including the gray inter-stimulus intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid as the neural data, using `np.searchsorted` on the stimulus presentation start times.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Both neural and image identity use the same `grid` array, ensuring temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, using `is_change`, `start_time`, `stop_time`, and `omitted` columns.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The AI uses the pre-computed `is_change` flag from the stimulus presentations table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time-varying signal: 1 during the stimulus presentation window where `is_change` is True and the stimulus is not omitted; 0 otherwise. The positive window corresponds to the duration of the change-image presentation (250ms image display period).

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    ...
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    codes[assign] = changed.astype(np.int64)
    return codes
```

iii. The AI documented in CONVERSION_NOTES.md (Step 10) that the initial single-bin impulse at `change_time` was too sparse and was replaced with the presentation-window approach using `is_change`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), no thresholding needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz trial grid as neural data.

ii.
```python
grid = session_grid(start, stop)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This is the standard running speed signal from the Allen SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`, then discretized into 5 equal percentile bins (quintiles) using globally computed bin edges.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)

def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    ...
    return percentiles

def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Bin edges are computed globally across all sessions in a first pass to maintain consistent categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global quintile edges at the 20th, 40th, 60th, and 80th percentiles are computed from all valid running speed values across all included sessions and trials. `np.digitize` maps values into 5 bins (0-4). A robustness correction ensures monotonically increasing edges.

ii.
```python
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles
```

iii. Percentile-based binning ensures roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid as the neural data.

ii.
```python
grid = session_grid(start, stop)
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

iii. The AI uses `max(width, height)` as an approximation of pupil diameter from the ellipse fit parameters.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `max(width, height)`, then interpolated onto the 30 Hz trial grid. Non-finite values (NaN/inf from blinks) are excluded before interpolation. Then discretized into 5 quintile bins using globally computed edges.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)

pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Excluding non-finite values before interpolation prevents blink artifacts from propagating.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: global quintile edges at 20th, 40th, 60th, 80th percentiles, with `np.digitize` producing bins 0-4.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as the neural data.

ii.
```python
grid = session_grid(start, stop)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. These are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast across all time bins in the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting the static outcome across time bins satisfies the format requirement that all outputs be time-varying arrays.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions lacking the `EyeTracking` group are excluded entirely before processing.
- **Non-finite pupil values**: The `interpolate_pupil` function filters out non-finite values before interpolation and raises an error if no valid samples exist.
- **Sessions with few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Invalid ROIs**: Neurons failing the `valid_roi` check are excluded.
- **Robust quintile edges**: If percentile edges are not monotonically increasing, a small epsilon is added to enforce monotonicity.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if int(raw["keep_mask"].sum()) < 2:
    continue
...
for i in range(1, len(percentiles)):
    if percentiles[i] <= percentiles[i - 1]:
        percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. The AI documented in CONVERSION_NOTES.md (Step 10) that the missing eye tracking issue caused a failure on the first full conversion attempt and was resolved by pre-filtering.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the two-pass design where every NWB file is read twice: once in `collect_global_statistics` (pass 1, without neural data) and once in `convert_sessions` (pass 2, with full neural data and interpolation).

ii.
```python
# Pass 1
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(sessions)
# Pass 2
data = convert_sessions(sessions=sessions, ...)
```

iii. The AI noted in CONVERSION_NOTES.md that neural interpolation is the dominant cost in pass 2 because every trial needs event traces resampled onto the 30 Hz grid.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_sessions` iterates over each trial sequentially within each session, performing interpolation and output construction. The stimulus identity/change code computation uses a loop over individual stimulus presentations.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
    ...
```

iii. The trial loop involves different grid sizes per trial, making full vectorization difficult. The neural interpolation within each trial is already vectorized across neurons.

## 9-c. What processing does the code repeat multiple times?

i. The code reads every NWB file twice: once in pass 1 (for global statistics) and once in pass 2 (for actual conversion). Running speed and pupil diameter are interpolated in both passes. This doubles the I/O cost.

ii.
```python
# In collect_global_statistics (pass 1):
raw = read_session_raw(session, load_events=False)
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# In convert_sessions (pass 2):
raw = read_session_raw(session)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The two-pass design avoids storing all neural arrays in memory while computing global bin edges, but at the cost of reading each file twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code interpolates neural event traces from the native ophys rate to 30 Hz. For sessions that are natively ~31 Hz (single-plane), this is essentially 1:1 and adds no information. For sessions that are natively ~11 Hz (multi-plane), the interpolation upsamples by ~3x, creating synthetic intermediate values that do not represent real measured activity. This upsampling may actually degrade decoder performance by smoothing the sparse event signal.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The AI justified 30 Hz as matching the paper's analysis approach, but the reference solution keeps the native ophys rate without resampling.
