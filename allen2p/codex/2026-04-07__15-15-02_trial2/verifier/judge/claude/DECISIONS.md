# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the `ophys_experiment_table.csv` metadata file to discover all experiments, then filters to only those with matching NWB files on disk. It excludes passive sessions and sessions missing eye-tracking data. Each NWB file is read directly using `h5py` (not the Allen SDK) because the local `pynwb/hdmf` stack was incompatible with the NWB 2.6.0 files. The conversion uses a two-pass design: pass 1 collects global statistics (bin edges, image vocabulary) without retaining neural arrays; pass 2 builds per-trial arrays.

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
    sessions = [SessionInfo(...) for row in exp_table.itertuples(index=False)]
    filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
    return sessions
```

iii. The AI justified using direct h5py reads because the Allen SDK's `BehaviorOphysExperiment.from_nwb_path` failed due to a `pynwb/hdmf` compatibility mismatch with NWB 2.6.0 files. The two-pass design was chosen to keep memory bounded. Passive sessions were excluded because the decoder task requires trial outcomes and the strategy paper focuses on active sessions.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment table metadata. A `subject_to_idx` dictionary maps each mouse ID to a sequential index as sessions are processed.

ii.
```python
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. The AI registered subjects in encounter order during the conversion pass.

## 1-c. How are the data split into sessions?

i. Each NWB file (one per `ophys_experiment_id`) is treated as a separate session. This means that multi-plane ophys sessions with multiple experiments are split into separate decoder sessions rather than being grouped.

ii.
```python
for sess_num, session in enumerate(sessions, start=1):
    t0 = time.perf_counter()
    raw = read_session_raw(session)
    ...
```

iii. The AI noted in CONVERSION_NOTES.md Step 4 that it chose to "Treat each NWB experiment file as one decoder session because neural traces are experiment-specific." This contrasts with the reference, which groups experiments by `ophys_session_id` to reconstruct multi-plane sessions.

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB `intervals/trials` table. The AI keeps only trials where `(go | catch) & ~aborted & ~auto_rewarded`. For each valid trial, a time grid is constructed from `start_time` to `stop_time` at a fixed 30 Hz rate (1/30 s bin size), giving variable-length trials.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The AI followed the instruction to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. The trial window uses start_time to stop_time from the trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch, (2) not aborted, (3) not auto-rewarded. Additionally, sessions with fewer than 2 valid trials are excluded. Sessions missing eye-tracking data are excluded entirely. The AI does NOT require a valid `change_time` (unlike the reference).

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ...")
    continue
```

iii. The AI justified excluding passive sessions, aborted trials, and auto-rewarded trials based on both the task instructions and the reference paper's focus on active behavioral sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` (calcium events), NOT from `dff` traces.

ii.
```python
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI justified this choice by noting that the strategy paper "explicitly states its neural analyses used detected calcium events" and that events are already present in the NWB files.

## 2-b. How is the `neural` data processed?

i. The neural event data is linearly interpolated from the native ophys timestamps onto a common 30 Hz time grid using `interpolate_matrix`. If `valid_roi` is present in the cell specimen table and differs from the event trace width, the traces are filtered to keep only valid ROIs.

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left = np.clip(right - 1, 0, n_src - 1)
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The AI chose to resample to 30 Hz because the strategy paper "linearly interpolates onto common 30 Hz timestamps" and because local data mix ~31 Hz single-plane and ~11 Hz multi-plane ophys sampling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. If `valid_roi` is present in the cell specimen table and its count differs from the number of event columns, the events are filtered to keep only valid ROIs.

ii.
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
```

iii. The AI noted that the SDK defaults to `exclude_invalid_rois=True` and that the released NWB content already reflects this filtering, so additional filtering should only apply when the event array includes all ROIs rather than just valid ones.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. A 30 Hz grid is constructed from trial `start_time` to `stop_time`, and event traces are linearly interpolated onto this grid.

ii.
```python
grid = session_grid(start, stop)  # 30 Hz grid from start_time to stop_time
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI uses ophys timestamps as the source for interpolation onto the trial grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed 30 Hz time bin (1/30 s = ~33.33 ms). This is a resampling from the native ophys rate (~11 Hz for multi-plane, ~31 Hz for single-plane) onto a uniform 30 Hz grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The AI justified 30 Hz by noting the strategy paper interpolates to 30 Hz timestamps and that behavior/eye-tracking are naturally 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`image_name`, `start_time`, `stop_time`, `omitted` columns) from the active task presentation group in the NWB file.

ii.
```python
stim_group = choose_task_presentation_group(h5f)
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name", "is_change", "omitted", ...])
...
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]
    ...
```

iii. The AI used the stimulus presentations table because it provides per-flash timing with explicit start/stop times and image names, enabling precise time-varying image identity including gray-screen periods and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 30 Hz time bin, the most recent stimulus presentation is found via `searchsorted`. If the bin falls within the presentation window (before `stop_time`) and the stimulus is not omitted, the image name is mapped to a global integer code. Otherwise, the bin is labeled as "gray" (code 0). The global image vocabulary includes "gray" plus all unique non-omitted image names sorted alphabetically.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(starts)
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    ...
    target[i] = image_to_code[str(name)]
```

iii. The AI included an explicit "gray" category for inter-stimulus intervals and omissions, which ensures the image identity variable is well-defined at all time points.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity codes are computed on the same 30 Hz grid as the neural data, using absolute stimulus presentation times.

ii.
```python
grid = session_grid(start, stop)
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Both neural and image identity use the same `grid` time points, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` and `omitted` columns in the stimulus presentations table, along with `start_time` and `stop_time` of each stimulus presentation.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The AI used the stimulus presentations table rather than the trials table's `change_time` and `go` flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 30 Hz time bin, if it falls within a stimulus presentation window that is marked as `is_change` and not `omitted`, the image change code is 1; otherwise 0. This means image change is 1 for the duration of the change-image flash (250 ms), not for 750 ms.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    sub_idx = idx_valid[in_interval]
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    assign = np.flatnonzero(valid)[in_interval]
    codes[assign] = changed.astype(np.int64)
    return codes
```

iii. The AI chose to mark the change during the stimulus presentation window from the stimulus table. This differs from the reference, which marks change for 750 ms (one flash + one gray period) starting at `change_time`, and only for go trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1). No thresholding is applied -- it is directly derived from the `is_change` flag.

ii. See 4-b above.

iii. Binary by definition from the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz grid as neural data.

ii. Same mechanism as image identity -- computed on the `grid` time points.

iii. Aligned by construction on the shared time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This is the SDK's standard running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps onto the 30 Hz trial grid using `np.interp`. Then it is discretized into 5 bins using global percentile-based edges (quintiles at 20th, 40th, 60th, 80th percentiles). Bin edges are computed across all valid trials in the first pass.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
...
running_bins = digitize_with_edges(running_cont, running_edges)
...
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
```

iii. Percentile-based binning ensures roughly equal class counts. Global edges ensure consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using `np.digitize` with global quintile edges.

ii.
```python
def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. `np.digitize` with 4 edges produces 5 bins (0 through 4).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz grid as neural data, ensuring frame-level alignment.

ii.
```python
grid = session_grid(start, stop)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Both use the same `grid` array.

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
...
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The AI chose `max(width, height)` as a proxy for pupil diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `max(width, height)`. NaN/invalid values are filtered out before interpolation. The valid values are linearly interpolated onto the 30 Hz trial grid using `np.interp`. Then discretized into 5 quintile bins using global edges (same approach as running speed).

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. Filtering NaN values before interpolation prevents blink artifacts from corrupting neighboring timepoints.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 bins (0-4) using `np.digitize` with global quintile edges.

ii.
```python
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Global quintile edges ensure consistent bin definitions across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 30 Hz grid as neural data.

ii.
```python
grid = session_grid(start, stop)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Aligned by construction on the shared time grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB `intervals/trials` table.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. These four columns are the SDK's canonical trial outcome labels. They are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3: hit, miss, false_alarm, correct_reject). The integer code is broadcast across all time bins within the trial (time-varying representation of a per-trial static variable).

ii.
```python
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting the static outcome across time satisfies the format requirement for time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions without eye-tracking data are excluded entirely before conversion.
- **Invalid ROIs**: If the `valid_roi` mask differs from event trace width, traces are filtered.
- **NaN pupil values**: Invalid/NaN pupil values are filtered before interpolation.
- **Sessions with few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Trial outcome edge cases**: If no outcome label matches, a `ValueError` is raised (fail-fast).

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
...
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10, noting that 3 sessions were excluded for missing eye tracking and that sparse-event warnings (2,467 all-zero neural trials) reflected genuine data properties rather than conversion errors.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the two-pass NWB file reading and neural interpolation. Each session requires reading large event arrays from HDF5 and interpolating them onto the 30 Hz grid. The full conversion took ~356 seconds for 199 sessions.

ii. N/A (timing is printed during execution)

iii. The AI noted that neural interpolation is the dominant cost and that the two-pass design avoids retaining all neural arrays simultaneously.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `convert_sessions` processes each trial sequentially, calling `interpolate_matrix`, `interpolate_vector`, `interpolate_pupil`, `stimulus_identity_codes`, and `stimulus_change_codes` per trial. The neural interpolation (`interpolate_matrix`) is already vectorized across neurons within each trial, but the trial loop itself is sequential.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    ...
```

iii. The AI noted that the trial loop is not a bottleneck compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice (once in pass 1 for global statistics, once in pass 2 for full conversion). In pass 1, running and pupil data are interpolated per trial to collect global bin edge statistics; in pass 2, the same interpolation is repeated. Neural interpolation is only done in pass 2.

ii.
```python
# Pass 1: collect_global_statistics
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
    ...
    running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
    pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# Pass 2: convert_sessions
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The two-pass design is a deliberate tradeoff: it avoids storing all neural arrays in memory while computing global bin edges, at the cost of reading each NWB file twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes pupil diameter as `max(width, height)` when the reference uses only `pupil_width`. The 30 Hz resampling is unnecessary additional processing since the reference keeps data at native ophys rate. The stimulus presentation table processing adds complexity for image identity and image change that the reference handles more simply from the trials table.

ii. N/A

iii. The more complex stimulus-table-based approach for image identity and image change adds processing steps compared to the reference's simpler trials-table approach, though both aim to produce the same output.
