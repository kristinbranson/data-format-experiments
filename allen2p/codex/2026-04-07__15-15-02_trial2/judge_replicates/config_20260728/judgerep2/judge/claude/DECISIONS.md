# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads an `ophys_experiment_table.csv` metadata file to discover all experiments, then loads each experiment's NWB file directly using `h5py` (not the AllenSDK session objects). Passive sessions are filtered out. Sessions missing eye-tracking data are also excluded. A `DATALIMIT_SUBSET.csv` file is checked to restrict to available local experiments.

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

iii. The AI chose direct `h5py` reads because the local `pynwb/hdmf` stack could not instantiate the NWB 2.6.0 files through the AllenSDK's `BehaviorOphysExperiment.from_nwb_path`. The metadata CSV and NWB file listing provide the experiment discovery. Passive sessions are excluded because they lack meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata table. A `subject_to_idx` dictionary maps each unique mouse ID to a sequential index.

ii.
```python
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. `mouse_id` is the standard unique identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. The AI does NOT group multiple experiments (imaging planes) from the same `ophys_session_id` into a single session. Each experiment corresponds to one imaging plane, so multi-plane sessions are split into separate decoder sessions.

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

iii. The AI treats each NWB experiment file as one decoder session because neural traces are experiment-specific (one imaging plane per file). This differs from the reference, which groups experiments by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table from the NWB file. The AI keeps trials where `(go or catch) and not aborted and not auto_rewarded`. For each valid trial, the time window is from `start_time` to `stop_time`, resampled onto a common 30 Hz grid.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The trial filter matches the instruction to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. The `start_time` to `stop_time` window provides the full trial period.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch, (2) not aborted, (3) not auto-rewarded. Sessions with fewer than 2 valid trials are skipped. Sessions missing eye-tracking data are excluded entirely.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ...")
    continue
```

iii. The filtering criteria follow the instructions. The reference also filters on `change_time.notna()`, which the AI does not explicitly check but is implicitly handled because go/catch trials should have valid change times.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` (precomputed calcium events), NOT from `dff_traces` (dF/F).

ii.
```python
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI chose calcium events because the strategy paper explicitly states its neural analyses used detected calcium events rather than raw dF/F. The events are already present in the NWB files.

## 2-b. How is the `neural` data processed?

i. The calcium event data is read from the NWB file, filtered to valid ROIs (if `valid_roi` mask differs from the event trace width), and then linearly interpolated from native ophys timestamps onto a common 30 Hz grid per trial.

ii.
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. Valid ROI filtering matches the SDK's default `exclude_invalid_rois=True`. Interpolation to 30 Hz is justified by the strategy paper's approach of linearly interpolating neural responses onto common 30 Hz timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` mask from the cell specimen table in the NWB file. No additional quality filtering is applied beyond what is in the released data.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The AllenSDK defaults to `exclude_invalid_rois=True`, so filtering by `valid_roi` matches the SDK behavior. The reference solution does not apply additional neuron quality filtering either.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (`start_time`). A 30 Hz time grid is created from `start_time` to `stop_time`, and the event traces are linearly interpolated onto this grid.

ii.
```python
grid = session_grid(start, stop)  # 30 Hz grid from start_time to stop_time

def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt

neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The instructions say to "temporally align based on ophys timestamp." The AI creates a uniform 30 Hz grid per trial and interpolates all data streams onto it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to a common 30 Hz time base (33.33 ms bins). This is different from the native ophys frame rate (~11 Hz for multi-plane or ~31 Hz for single-plane).

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
```

iii. The AI justified 30 Hz because the strategy paper linearly interpolates onto common 30 Hz timestamps, behavior and eye tracking are natively at 30 Hz, and it provides a consistent bin size across sessions with different native ophys rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (active task block), using `image_name`, `start_time`, `stop_time`, and `omitted` fields. A `gray` category is used for inter-stimulus intervals and omitted stimuli.

ii.
```python
stim_group = choose_task_presentation_group(h5f)
stim = read_interval_table(stim_group, [
    "start_time", "stop_time", "image_name", "is_change", "omitted",
    "trials_id", "active", "flashes_since_change",
])
```

iii. The AI uses the stimulus presentation table rather than the trials table's `initial_image_name`/`change_image_name`, which allows distinguishing between image display and gray-screen periods within each trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin on the 30 Hz grid, the AI determines which stimulus presentation (if any) is active at that time. If a non-omitted stimulus is being shown, its image name is mapped to an integer code. Otherwise, the "gray" code is assigned. The image vocabulary includes "gray" plus all unique image names.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    ...
    for i, (name, is_omitted) in enumerate(zip(names, omit)):
        if (not is_omitted) and str(name) in image_to_code:
            target[i] = image_to_code[str(name)]
    ...
image_values = ["gray"] + sorted(image_names)
```

iii. The inclusion of "gray" as an explicit category captures the 500ms inter-stimulus gray periods in the task. This differs from the reference, which only distinguishes between `initial_image_name` and `change_image_name`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz grid as the neural data, using `stimulus_identity_codes` which maps each time bin to the active stimulus presentation.

ii.
```python
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Using the same `grid` for all data streams ensures temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, specifically the `is_change` field combined with `start_time`, `stop_time`, and `omitted`.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
    ...
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    ...
```

iii. The AI uses the stimulus table's `is_change` flag rather than the trial-level `change_time` and `go` fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each time bin, the AI checks if the bin falls within a stimulus presentation that is marked as `is_change` and not omitted. The value is 1 during the change stimulus presentation window and 0 otherwise.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    assign = np.flatnonzero(valid)[in_interval]
    codes[assign] = changed.astype(np.int64)
    return codes
```

iii. The AI marks image_change as 1 during the full change stimulus presentation (250ms image), not for a fixed 750ms window as in the reference.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. It is 1 during a change stimulus presentation and 0 otherwise.

ii. See 4-b code snippet.

iii. The binary encoding directly captures whether a change stimulus is being presented.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the same 30 Hz grid as neural data.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Alignment is ensured by using the same time grid for all data streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This matches the SDK's `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps onto the 30 Hz trial grid using `np.interp`, then discretized into 5 bins using global quintile edges computed across all sessions.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
...
running_edges = robust_quintile_edges(running_all)
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global quintile edges ensure consistent bin definitions across sessions. The `robust_quintile_edges` function uses the 20th, 40th, 60th, 80th percentiles as edges.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.digitize` with edges at the 20th, 40th, 60th, and 80th percentiles of all running speed values across all sessions.

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

iii. The quintile approach ensures roughly equal class frequencies. A robustness step prevents duplicate edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz grid as the neural data.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Alignment is ensured by using the same time grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `height`, plus timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

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

iii. The AI computes pupil diameter as the maximum of width and height, providing a more robust measure than width alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (`max(width, height)`) is interpolated onto the 30 Hz grid using `np.interp`, skipping NaN values (from blinks). Then discretized into 5 bins using global quintile edges.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. NaN filtering before interpolation avoids propagating blink artifacts. The approach is similar to the reference's blink removal before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed - discretized into 5 bins using global quintile edges (20th, 40th, 60th, 80th percentiles).

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz grid as neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the NWB trials table.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. These are the SDK's canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first matching outcome label is mapped to an integer code (0-3). The code is broadcast across all time bins in the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting the static per-trial outcome across all time bins satisfies the decoder format requirement.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions lacking eye-tracking data are excluded entirely before processing.
- **NaN pupil values**: The `interpolate_pupil` function filters out non-finite values before interpolation.
- **Few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Valid ROI filtering**: If the cell table has a `valid_roi` mask that differs from the event trace dimensions, invalid ROIs are filtered out.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. These edge cases were discovered during development and documented in CONVERSION_NOTES.md.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading NWB files and interpolating neural event traces onto the 30 Hz grid. The two-pass design means each NWB file is read twice (once for global statistics, once for full conversion).

ii. N/A

iii. From the conversion output, full conversion took ~356 seconds for 199 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `stimulus_identity_codes` function has an inner Python loop over stimulus presentations that could potentially be vectorized. The per-trial loop iterating over `np.flatnonzero(raw["keep_mask"])` is sequential.

ii.
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The stimulus identity loop is small relative to the neural interpolation cost.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in `collect_global_statistics` (pass 1, without neural data) and once in `convert_sessions` (pass 2, with neural data). Running speed and pupil diameter are interpolated in both passes.

ii.
```python
# Pass 1
raw = read_session_raw(session, load_events=False)
...
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# Pass 2
raw = read_session_raw(session)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The two-pass design trades computation time for memory efficiency, avoiding storing all neural arrays while computing global bin edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `flashes_since_change` from the stimulus table but does not use it in the final output. The `active` flag from stimulus presentations is read but only used for choosing the task presentation group, not directly in the output.

ii.
```python
stim = read_interval_table(stim_group, [
    "start_time", "stop_time", "image_name", "is_change", "omitted",
    "trials_id", "active", "flashes_since_change",
])
```

iii. These extra fields were loaded for potential use during development but are not needed in the final conversion.
