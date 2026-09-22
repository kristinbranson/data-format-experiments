# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than through the Allen SDK. It discovers sessions from a metadata CSV (`ophys_experiment_table.csv`), maps each experiment ID to its NWB file on disk, and filters to active (non-passive) sessions. It additionally filters out sessions that lack eye-tracking pupil data. For each session, it opens the NWB file and reads trial intervals, ophys timestamps, event detection traces, running speed, and pupil tracking data.

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
        trial_group = h5f["intervals"]["trials"]
        trials = read_interval_table(trial_group, [...])
        ...
        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
        )
        events = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
        )
```

iii. The AI documented in CONVERSION_NOTES.md that the Allen SDK's `pynwb/hdmf` stack could not instantiate the NWB 2.6.0 files due to an `external_resources` abstract-method mismatch. The AI therefore used direct HDF5 reads mirroring the SDK field definitions. This is documented under Step 5 Key Decision 9.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment metadata CSV. The AI converts mouse IDs to strings and builds a subject-to-index mapping dynamically as sessions are processed.

ii.
```python
sessions = [
    SessionInfo(
        ...
        mouse_id=str(int(row.mouse_id)),
        ...
    )
    for row in exp_table.itertuples(index=False)
]
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The `mouse_id` field is the standard identifier for each animal. The AI noted 38 mice in the local data subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one NWB file, one imaging plane) is treated as a separate session. The AI does NOT group multiple experiments from the same `ophys_session_id` into a single session. Passive sessions are excluded. Sessions missing eye-tracking data are also excluded.

ii.
```python
exp_table = exp_table[~exp_table["passive"]].copy()
exp_table = exp_table.sort_values("ophys_experiment_id")
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The AI documented that each NWB file corresponds to one experiment/imaging plane. For the VisualBehavior project (single-plane), each ophys session has one experiment, so this is effectively equivalent to grouping by `ophys_session_id`. The AI also noted that passive sessions were excluded because they lack meaningful trial outcomes (Key Decision 1 in CONVERSION_NOTES.md Step 5).

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. Only trials that are `go` or `catch` and not `aborted` and not `auto_rewarded` are kept. Each trial spans from `start_time` to `stop_time` (variable length). Trial data is resampled onto a common 30 Hz time grid within each trial window.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The AI followed the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. Variable-length trials from `start_time` to `stop_time` are used rather than a fixed window.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering: `(go | catch) & ~aborted & ~auto_rewarded`. Sessions with fewer than 2 valid trials are excluded. Sessions missing eye-tracking data are excluded entirely (3 sessions).

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

iii. The AI documented the trial filtering logic and noted that 3 active sessions were excluded due to missing eye-tracking data. The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection` (precomputed calcium events), NOT from `dff_traces` (dF/F). The events array is read from `processing/ophys/event_detection/data` in the NWB file.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI documented in CONVERSION_NOTES.md (Step 4, Step 5) that the strategy paper explicitly states its neural analyses used "detected calcium events." The AI chose events over dF/F to match the paper's analysis approach.

## 2-b. How is the `neural` data processed?

i. Calcium event traces are filtered by `valid_roi` if present, then linearly interpolated from the native ophys timestamps onto a common 30 Hz time grid for each trial. The interpolation is done using a custom `interpolate_matrix` function.

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

iii. The AI documented that valid_roi filtering matches the SDK's `exclude_invalid_rois=True` behavior. The 30 Hz interpolation follows the paper's approach of "linearly interpolates onto common 30 Hz timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `valid_roi` from the cell specimen table, which removes non-cell-body ROIs, duplicates, and other problematic segmentations. No additional quality filtering is applied beyond what's in the NWB files.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The AI documented that the Allen SDK pipeline already applies quality control and that `valid_roi` filtering matches the SDK's default behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start (`start_time`). A 30 Hz grid is created from `start_time` to `stop_time`, and the event traces are linearly interpolated onto this grid.

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

iii. The AI documented that all trial streams (neural, running, pupil, stimulus) are aligned in absolute experiment time and resampled to the same 30 Hz grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a fixed 30 Hz time bin (33.33 ms). All data streams are resampled via linear interpolation onto this common grid. This differs from the native ophys frame rate (~31 Hz for single-plane VisualBehavior sessions).

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"metadata": {
    ...
    "time_bin_size": float(TIME_BIN_SIZE_MS),
    ...
}
```

iii. The AI justified the 30 Hz grid by citing the strategy paper, which "linearly interpolates calcium event responses onto common 30 Hz timestamps." The behavior and eye-tracking streams are also naturally at 30 Hz, making 30 Hz the "most defensible common grid."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, using the `image_name`, `start_time`, `stop_time`, and `omitted` columns from the active task presentation group.

ii.
```python
stim_group = choose_task_presentation_group(h5f)
stim = read_interval_table(
    stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted", "trials_id", "active", "flashes_since_change"],
)
```

iii. The AI used the stimulus presentations table rather than the trials table's `initial_image_name`/`change_image_name` to construct the time-varying image identity, allowing it to track image-on-screen vs gray-period at each time bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A piecewise-constant signal is constructed on the 30 Hz grid: during image presentation windows, the image code is assigned; during gray inter-stimulus intervals and omitted stimuli, the code for "gray" is assigned. A global image vocabulary (["gray"] + sorted unique image names) is built across all sessions.

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

iii. The AI justified including "gray" as an explicit category because the task includes 500ms gray periods and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity codes are computed directly on the same 30 Hz grid as the neural data, using `np.searchsorted` on the stimulus presentation times. This guarantees alignment.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Both neural and image identity use the same `grid` array, ensuring temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table, combined with `start_time` and `stop_time` of each stimulus presentation.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The AI uses the stimulus table's `is_change` flag rather than reconstructing the change event from the trials table's `change_time` and `go` columns.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time-varying signal is constructed: 1 during the presentation window of a stimulus marked as `is_change` (and not omitted), 0 otherwise. The change is marked for the duration of the changed-image presentation (250ms), not a fixed 750ms window.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    ...
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    assign = np.flatnonzero(valid)[in_interval]
    codes[assign] = changed.astype(np.int64)
    return codes
```

iii. The AI initially used a single-bin impulse at change_time but revised to use the `is_change` presentation window after finding decoder performance was below chance with the sparse impulse.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 = no change, 1 = change). No additional thresholding is needed.

ii.
```python
["no_change", "change"]
```

iii. N/A - already categorical.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the same 30 Hz grid as neural data using `np.searchsorted` on stimulus presentation times.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same grid alignment as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed values and timestamps.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This is the same running speed data exposed by the SDK's `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`, then discretized into 5 percentile-based bins. Bin edges are computed globally across all sessions using the 20th, 40th, 60th, and 80th percentiles.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
...
running_edges = robust_quintile_edges(running_all)
...
running_bins = digitize_with_edges(running_cont, running_edges)

def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles
```

iii. The AI computes global bin edges in a first pass across all sessions, then applies them in a second pass. The robust edge adjustment prevents degenerate bins when percentiles collide.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins using global quintile edges (20th, 40th, 60th, 80th percentiles). Values below the 20th percentile are bin 0, above the 80th are bin 4.

ii.
```python
def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Equal percentile binning ensures roughly balanced class counts, which is important for decoder training.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid as neural data, ensuring alignment.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Same grid as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking`, using `max(pupil_width, pupil_height)` as the diameter measure. Sessions without eye-tracking data are excluded.

ii.
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
...
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64
)
```

iii. The AI chose `max(width, height)` as the pupil diameter measure. The reference uses only `pupil_width`. Sessions without the `EyeTracking` group are excluded before processing.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is interpolated from its native timestamps to the 30 Hz trial grid, filtering out non-finite values (NaN from blinks) before interpolation. Then discretized into 5 percentile-based bins globally.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Filtering non-finite values before interpolation effectively removes blink frames, which would be NaN. The AI noted this is equivalent to the SDK's blink handling.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: discretized into 5 bins using global quintile edges.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Equal percentile binning for balanced classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same grid alignment.

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

iii. These four columns are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping, then broadcast across all time bins within the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The mapping order matches the reference. Broadcasting the static outcome across all time bins satisfies the "static per-trial" requirement while maintaining the time-varying array format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions excluded entirely (`has_required_eye_tracking` check).
- **Invalid ROIs**: Filtered out using `valid_roi` mask from cell specimen table.
- **Missing pupil data**: `interpolate_pupil` filters non-finite values before interpolation.
- **Few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Invalid trial outcomes**: `trial_outcome_code` raises ValueError for trials without a valid outcome.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if int(raw["keep_mask"].sum()) < 2:
    continue
...
valid = np.isfinite(pupil_diameter)
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading NWB files. The AI uses a two-pass design where each NWB file is read twice: once for global statistics (pass 1) and once for full conversion (pass 2). Neural interpolation is also noted as a significant cost.

ii. N/A (architectural observation)

iii. The AI documented estimated runtimes in CONVERSION_NOTES.md Step 7 and noted pass 1 at ~0.5s/session and pass 2 at ~1.35s/session.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_sessions` iterates over each valid trial sequentially, performing interpolation and stimulus code construction. The `stimulus_identity_codes` function has an inner Python loop over stimulus presentations that could be vectorized.

ii.
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The per-trial loop is noted as not a bottleneck relative to I/O.

## 9-c. What processing does the code repeat multiple times?

i. The AI uses a two-pass design that reads each NWB file twice: once in `collect_global_statistics` (pass 1, without neural data) and once in `convert_sessions` (pass 2, with neural data). This doubles the I/O for behavioral data (running speed, pupil, trials, stimulus presentations).

ii.
```python
# Pass 1
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(sessions)
# Pass 2
data = convert_sessions(sessions=sessions, ...)
```

iii. The AI justified this as a memory optimization: pass 1 avoids storing all neural arrays while computing global bin edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The 30 Hz resampling introduces unnecessary interpolation for VisualBehavior single-plane data (native ~31 Hz). The data could have been kept at native resolution without loss. Additionally, the two-pass design re-reads all behavioral data unnecessarily.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0  # Fixed 30 Hz grid
...
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The AI justified the 30 Hz grid by citing the paper's approach, but for VisualBehavior single-plane sessions at ~31 Hz, this adds interpolation overhead and potential artifacts without clear benefit.
