# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly via `h5py` from the local `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` directory. It reads `ophys_experiment_table.csv` from the metadata directory to discover all experiments, maps experiment IDs to NWB file paths, and then reads each NWB file directly for trial, neural, running, and eye-tracking data. The AI uses a two-pass approach: pass 1 collects global statistics (running/pupil bin edges, image names) without loading neural data; pass 2 loads neural data and builds per-trial arrays.

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

iii. The AI could not use the Allen SDK's `VisualBehaviorOphysProjectCache` or `BehaviorOphysExperiment.from_nwb_path` due to `pynwb/hdmf` compatibility issues with the NWB 2.6.0 files. Instead, it used direct `h5py` reads mirroring the SDK field definitions. This is documented in CONVERSION_NOTES.md Step 5, Key Decision 9.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata CSV table. Each unique mouse_id becomes a subject string.

ii.
```python
mouse_id=str(int(row.mouse_id)),
...
subject_to_idx: Dict[str, int] = {}
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The `mouse_id` field from the experiment table is the standard identifier for each animal. The AI uses it directly as the subject identifier string.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. Unlike the reference solution which groups multiple experiments (imaging planes) from the same `ophys_session_id` into one session, the AI treats each experiment (single imaging plane) as its own session.

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

iii. The AI justified this by noting that "neural traces are experiment-specific" (CONVERSION_NOTES.md Step 4). Each NWB file corresponds to one `ophys_experiment` (one imaging plane), so treating each as a session avoids needing to merge neurons across planes.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table from the NWB file. For each kept trial (go or catch, not aborted, not auto-rewarded), the trial window spans from `start_time` to `stop_time`. The AI creates a 30 Hz time grid for each trial window and interpolates all signals onto it.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The trial boundaries come from the SDK's built-in trials table. The AI follows the instruction to include go and catch trials and exclude aborted and auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only `(go | catch) & ~aborted & ~auto_rewarded`. Sessions with fewer than 2 valid trials after filtering are excluded. Additionally, 3 sessions missing eye-tracking data are excluded entirely before processing.

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

iii. The filtering criteria match the instructions. Passive sessions are also excluded because they produce degenerate trial outcomes (all misses/correct rejects). The eye-tracking check prevents crashes on sessions without pupil data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` in the NWB files — these are precomputed calcium event traces, NOT dF/F traces.

ii.
```python
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI chose calcium events over dF/F because "the strategy paper explicitly states its neural analyses used detected calcium events" (CONVERSION_NOTES.md Step 4). The events are already present in the NWB files as precomputed arrays.

## 2-b. How is the `neural` data processed?

i. The neural event traces are linearly interpolated from the native ophys timestamps (~11 Hz or ~31 Hz depending on imaging configuration) onto a common 30 Hz time grid. Invalid ROIs are filtered out if a `valid_roi` mask is present in the cell specimen table. The result is transposed to (n_neurons, n_timepoints) format.

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    ...
    right = np.searchsorted(source_t, query_t, side="left")
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)

neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The 30 Hz resampling was chosen because the strategy paper "linearly interpolates onto common 30 Hz timestamps relative to behavioral events" and behavior/eye tracking are naturally 30 Hz (CONVERSION_NOTES.md Step 5, Key Decision 3).

## 2-c. How is the `neural` data filtered based on quality controls?

i. If `valid_roi` is present in the cell specimen table and the number of valid ROIs differs from the event trace width, the event traces are filtered to keep only valid ROIs.

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

iii. This mirrors the SDK's `exclude_invalid_rois=True` behavior documented in the reference code exploration (CONVERSION_NOTES.md Step 1).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps via linear interpolation onto a common 30 Hz grid spanning each trial's `start_time` to `stop_time`. The grid is generated as: `start + np.arange(n_bins) * (1/30)`.

ii.
```python
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt

grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. All signals (neural, running, pupil, stimulus) are interpolated onto the same 30 Hz grid, ensuring temporal alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 1/30 s (~33.33 ms). The AI rebins all data to a common 30 Hz grid, regardless of the native ophys frame rate.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. The AI justified the 30 Hz grid based on the strategy paper's use of 30 Hz interpolation and the fact that behavior/eye tracking are naturally at 30 Hz (CONVERSION_NOTES.md Step 5, Key Decision 3).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, and `omitted` columns from the active task presentation group.

ii.
```python
stim_group = choose_task_presentation_group(h5f)
stim = read_interval_table(stim_group, [
    "start_time", "stop_time", "image_name", "is_change", "omitted",
    "trials_id", "active", "flashes_since_change",
])
```

iii. The AI used the stimulus presentations table rather than the trials table's `initial_image_name`/`change_image_name`, which provides more precise timing of when each image is actually on screen versus gray-screen periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global image vocabulary is built from all non-omitted, non-empty image names across all sessions, with "gray" added as an explicit category. For each trial's 30 Hz time grid, the code determines which stimulus presentation interval each timepoint falls within, and assigns the corresponding image code. Timepoints during gray periods or omitted stimuli get the "gray" code.

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
    for i, (name, is_omitted) in enumerate(zip(names, omit)):
        if (not is_omitted) and str(name) in image_to_code:
            target[i] = image_to_code[str(name)]
    ...
```

iii. The AI includes "gray" as a distinct image identity category to represent the 500ms inter-stimulus intervals and omitted stimuli, giving a more complete time-varying representation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity codes are computed on the same 30 Hz trial grid as the neural data, using the stimulus presentation start/stop times in absolute experiment time.

ii.
```python
grid = session_grid(start, stop)
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. All outputs share the same 30 Hz grid, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table, along with `start_time`, `stop_time`, and `omitted`.

ii.
```python
stim = read_interval_table(stim_group, [..., "is_change", ...])
stim["is_change"] = stim["is_change"].astype(bool)
```

iii. The AI uses the stimulus presentation-level `is_change` flag rather than the trial-level `change_time` and `go` flag used by the reference.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 30 Hz timepoint, the code checks if it falls within a stimulus presentation that has `is_change=True` and is not omitted. If so, the code is 1; otherwise 0.

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

iii. The AI marks the entire changed-image presentation window as 1, rather than a 750ms window. This covers the 250ms image presentation duration.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. N/A — the values are directly 0 or 1 based on whether the timepoint falls within a change presentation.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz grid as all other signals.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Alignment is guaranteed by using the same grid for all signals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This corresponds to the SDK's `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`. Global quintile bin edges are computed from all valid running speed values across all sessions, then applied via `np.digitize`.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
...
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    ...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global quintile binning ensures consistent bin definitions across sessions. The `robust_quintile_edges` function handles edge cases where percentile values may be tied.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using global quintile edges at the 20th, 40th, 60th, and 80th percentiles. `np.digitize` assigns bin indices.

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

iii. The robust edge computation adds a small epsilon when percentile values are tied, preventing degenerate bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid used for neural data.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Same grid alignment as all other signals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height`, with timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`. The diameter is computed as `max(width, height)`.

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

iii. The AI uses max(width, height) as a proxy for pupil diameter, rather than just `pupil_width` as the reference does.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Invalid (non-finite) pupil samples are excluded before interpolation. The valid samples are linearly interpolated onto the 30 Hz trial grid using `np.interp`. Then global quintile binning is applied.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. The AI filters by `np.isfinite` rather than using the `likely_blink` flag from the reference. The NWB data may have NaN values during blinks, which would be filtered by `isfinite`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: global quintile edges at 20th, 40th, 60th, 80th percentiles, applied via `np.digitize`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same robust quintile approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same grid alignment as all other signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
trials = read_interval_table(trial_group, [
    "go", "catch", "aborted", "auto_rewarded",
    "hit", "miss", "false_alarm", "correct_reject",
    "change_time", "start_time", "stop_time",
])
...
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. These four columns are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first matching outcome from the ordered list `["hit", "miss", "false_alarm", "correct_reject"]` is mapped to an integer code (0-3). This code is broadcast as a constant across all time bins in the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The mapping is deterministic and consistent with the SDK's trial outcome definitions. Broadcasting across time makes it a time-varying output suitable for the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions without the `EyeTracking` HDF5 group are excluded entirely before processing.
- **Invalid pupil samples**: Non-finite pupil values are excluded from interpolation; `np.interp` interpolates over gaps.
- **Sessions with few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Invalid ROIs**: If `valid_roi` mask is present and differs from event trace width, invalid ROIs are filtered out.
- **Tied percentile edges**: The `robust_quintile_edges` function adds epsilon to prevent degenerate bins when percentile values are identical.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
...
if int(raw["keep_mask"].sum()) < 2: continue
...
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. These are documented in CONVERSION_NOTES.md Steps 10 and 12 as edge cases discovered during development.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the two-pass reading of all NWB files via `h5py`. Pass 2 is more expensive because it loads the neural event traces and performs per-trial interpolation. The full conversion took ~356 seconds.

ii. From CONVERSION_NOTES.md: "Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid."

iii. The two-pass design was chosen to avoid keeping all neural data in memory while computing global bin edges.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_sessions` iterates over each trial sequentially, performing interpolation and code assignment individually. The `stimulus_identity_codes` function contains a Python for-loop over stimulus presentations to assign image codes. The neural interpolation `interpolate_matrix` is vectorized across neurons within each trial.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    neural_trial = interpolate_matrix(...)
    running_cont = interpolate_vector(...)
    ...

# In stimulus_identity_codes:
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The neural interpolation within each trial is already vectorized. The per-trial loop is hard to vectorize since trials have variable lengths.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is read twice: once in pass 1 (for global statistics, without neural data) and once in pass 2 (with neural data for final conversion). This doubles the I/O for non-neural data (trials, running, pupil, stimulus tables).

ii.
```python
# Pass 1:
raw = read_session_raw(session, load_events=False)

# Pass 2:
raw = read_session_raw(session)
```

iii. The two-pass design trades I/O efficiency for memory efficiency — it avoids keeping all neural arrays in memory while computing global bin edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads several stimulus table columns that are not directly used in the final output, such as `flashes_since_change`, `trials_id`, and `active`. The `choose_task_presentation_group` function scans multiple presentation groups to find the best one. These are used for validation/selection but add overhead.

ii.
```python
stim = read_interval_table(stim_group, [
    "start_time", "stop_time", "image_name", "is_change", "omitted",
    "trials_id", "active", "flashes_since_change",
])
```

iii. The extra columns provide robustness in identifying the correct stimulus block but aren't used in the final per-trial output construction.
