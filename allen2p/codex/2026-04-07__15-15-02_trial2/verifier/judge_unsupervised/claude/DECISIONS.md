# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` (not the AllenSDK). It reads the `ophys_experiment_table.csv` metadata to identify all experiment files, then iterates over the corresponding NWB files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`. Each NWB file corresponds to one ophys experiment. The code uses a two-pass approach: pass 1 collects global statistics (running speed/pupil quintile edges, image names) without loading neural data, and pass 2 loads everything including neural events for conversion.

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

iii. The AI chose direct `h5py` reads because the local `pynwb/hdmf` stack could not instantiate the NWB 2.6.0 files through the SDK's `BehaviorOphysExperiment.from_nwb_path`. The AI documented this incompatibility in CONVERSION_NOTES.md Step 5 Key Decision #9.

## 1-b. How are the data split into subjects (mice)?

i. Each session has a `mouse_id` from the `ophys_experiment_table.csv` metadata. Subjects are collected into a list as new mouse IDs are encountered. A `subject_to_idx` dictionary maps mouse ID strings to indices.

ii.
```python
subject_to_idx: Dict[str, int] = {}
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. Uses the `mouse_id` field from the experiment metadata CSV, consistent with the SDK's metadata approach.

## 1-c. How are the data split into sessions?

i. Each ophys experiment NWB file is treated as one session. Only active (non-passive) sessions are included. Sessions missing eye-tracking pupil data are excluded (3 sessions). Each experiment file yields one entry in the `neural`, `input`, and `output` lists.

ii.
```python
exp_table = exp_table[~exp_table["passive"]].copy()
...
sessions = [SessionInfo(...) for row in exp_table.itertuples(index=False)]
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The AI excluded passive sessions because the decoder task requires trial outcome labels (hit/miss/etc.), and passive sessions produce degenerate outcomes (all go trials are misses, all catch are correct rejects). 3 active sessions lacking eye-tracking data were also excluded, resulting in 199 sessions from 38 mice.

## 1-d. How are the data split into trials?

i. Within each session, trials are defined by the `intervals/trials` table in the NWB file. Each trial spans from `start_time` to `stop_time`. A time grid is constructed for each trial at 30 Hz spacing.

ii.
```python
trial_group = h5f["intervals"]["trials"]
trials = read_interval_table(trial_group, ["go", "catch", "aborted", "auto_rewarded", ...])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. Trial boundaries come from the NWB `intervals/trials` table, which matches the SDK's trial timing logic.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ...")
    continue
```

iii. This matches the instructions ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials") and the reference SDK's trial taxonomy.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `processing/ophys/event_detection/data` in the NWB files, which contains precomputed calcium events. Timestamps come from `processing/ophys/event_detection/timestamps`.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI chose calcium events over dF/F because the reference paper explicitly states its neural analyses used "detected calcium events." CONVERSION_NOTES.md Step 5 Key Decision #2 documents this.

## 2-b. How is the `neural` data processed?

i. The raw event traces (shape: n_timepoints x n_neurons at ~31 Hz) are linearly interpolated onto a common 30 Hz trial grid, then transposed to (n_neurons, n_timepoints).

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```
The `interpolate_matrix` function does linear interpolation:
```python
def interpolate_matrix(source_t, source_values, query_t):
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left = np.clip(right - 1, 0, n_src - 1)
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
```

iii. The paper interpolates calcium event responses onto common 30 Hz timestamps, which this code replicates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` flag from the cell_specimen_table, consistent with the SDK's `exclude_invalid_rois=True` default. In practice, all ROIs in the local dataset are valid, so no filtering occurs.

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

iii. The AI correctly implements the SDK's valid_roi filtering logic, but notes that all ROIs in the local NWB files are already valid. No additional ad-hoc neuron filtering is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, a 30 Hz time grid is constructed from `start_time` to `stop_time`, and the neural events are interpolated onto that grid. The metadata records `temporal_alignment_event` as "trial start" with `off_start = 0.0`.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI constructs a grid using absolute experiment time (the same time base as ophys timestamps), so neural, behavioral, and stimulus data are all aligned in the same reference frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 1/30 seconds (~33.33 ms). This is a fixed constant. Neural data (originally ~31 Hz) and behavioral data are linearly interpolated onto this 30 Hz grid. No temporal averaging or summing is applied -- it is pure interpolation.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The reference paper linearly interpolates onto common 30 Hz timestamps, and behavior/eye tracking are naturally 30 Hz, making 30 Hz the common grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from the stimulus presentations table in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, and `omitted` columns from the active task block.

ii.
```python
stim = read_interval_table(stim_group, [
    "start_time", "stop_time", "image_name", "is_change", "omitted",
    "trials_id", "active", "flashes_since_change",
])
```

iii. The AI selects the active change-detection stimulus block via `choose_task_presentation_group`, which looks for presentations with `active` and `image_name` columns, prioritizing blocks with "change_detection" in `stimulus_block_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin in the trial grid, the code determines which stimulus presentation interval contains that time point. If the time bin falls within a non-omitted stimulus presentation, the image name code is assigned. Otherwise, "gray" is assigned. Image names are mapped to integer codes: gray=0, then alphabetically sorted unique image names.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(starts)
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        names = np.asarray(image_names[sub_idx], dtype=object)
        omit = omitted[sub_idx]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        assign = np.flatnonzero(valid)[in_interval]
        codes[assign] = target
    return codes
```

iii. The AI uses a piecewise-constant representation: each time bin gets the identity of the currently displayed image, or "gray" during inter-stimulus intervals and omitted stimuli.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity codes are computed on the same 30 Hz trial time grid as the neural data. Both use the same `grid` array for each trial.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. All signals (neural, stimulus, behavior) share the same time grid, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` and `omitted` columns of the stimulus presentations table, plus `start_time` and `stop_time`.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The AI uses the stimulus-level `is_change` flag rather than the trial-level `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each time bin, the code checks if it falls within a stimulus presentation interval where `is_change` is True (and not omitted). If so, the bin is labeled 1; otherwise 0. This means the entire duration of the change stimulus flash (~250ms = ~8 bins at 30 Hz) is labeled as "change."

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        changed = is_change[sub_idx] & (~omitted[sub_idx])
        assign = np.flatnonzero(valid)[in_interval]
        codes[assign] = changed.astype(np.int64)
    return codes
```

iii. The AI initially used a single-bin impulse at `change_time` but found decoder accuracy was below chance. It then changed to labeling the entire change-stimulus presentation window, which improved accuracy. This is documented in CONVERSION_NOTES.md Step 10.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable: 0 (no change) or 1 (change). No further thresholding is applied.

ii.
```python
codes = np.zeros(query_t.shape, dtype=np.int64)
...
codes[assign] = changed.astype(np.int64)
```
Output values: `["no_change", "change"]`

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Computed on the same 30 Hz trial grid as neural data.

ii.
```python
grid = session_grid(start, stop)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same alignment approach as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` with timestamps from `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This matches the SDK's `BehaviorSession.running_speed` field.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from the original timestamps (~60 Hz) onto the 30 Hz trial grid. Then it is digitized into 5 quintile bins using globally computed bin edges.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```
Global edges:
```python
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles
```

iii. The two-pass approach computes quintile edges from all valid time bins across all sessions, then applies them in the second pass.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using `np.digitize` with 4 global quintile edges at the 20th, 40th, 60th, and 80th percentiles. A robustness check ensures edges are strictly increasing.

ii.
```python
running_edges = robust_quintile_edges(running_all)
...
def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The instructions say "discretized into five equal percentile bins", which quintile edges implement.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Computed on the same 30 Hz trial time grid as neural data.

ii.
```python
grid = session_grid(start, stop)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Same alignment approach as all other signals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height`. Timestamps come from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64
)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The AI computes `max(width, height)` as a proxy for pupil diameter. The reference SDK code computes `pupil_area = pi * max(width, height)^2`. Since quintile binning is a monotonic transformation, using max(w,h) vs pi*max(w,h)^2 produces identical bin assignments.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (`max(width, height)`) is interpolated onto the 30 Hz trial grid, skipping NaN values (which correspond to blink-masked frames in the NWB). Then digitized into 5 quintile bins using global edges.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. The NWB files already have blink frames set to NaN in the width/height fields (verified: NaN count matches `likely_blink` True count). The AI's `interpolate_pupil` filters to finite values before interpolation, effectively interpolating across blinks.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 quintile bins (0-4) using global edges at 20th/40th/60th/80th percentiles.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Instructions say "discretized into five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Computed on the same 30 Hz trial time grid as neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same alignment approach as all other signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the NWB `intervals/trials` table.

ii.
```python
trials = read_interval_table(trial_group, [..., "hit", "miss", "false_alarm", "correct_reject", ...])
...
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. The four outcome types match the SDK's trial taxonomy for Go (hit/miss) and Catch (false_alarm/correct_reject) trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A single integer code is determined per trial by checking which of the four outcome flags is True. This code is broadcast (repeated) across all time bins in the trial.

ii.
```python
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```
Outcome mapping: `{"hit": 0, "miss": 1, "false_alarm": 2, "correct_reject": 3}`

iii. The instructions say "Trial outcome. Static per-trial." The AI broadcasts it as time-varying to fit the format, which is documented as Key Decision #5.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data cases are handled:
- **Missing eye tracking**: 3 active sessions lacking `EyeTracking/pupil_tracking` are excluded entirely.
- **Missing pupil values (blinks)**: NaN values in pupil width/height (blink-masked by the NWB) are skipped during interpolation; `interpolate_pupil` filters to finite values.
- **All-zero neural events**: Trials with all-zero neural activity (2,467 trials, ~4.83%) are kept as-is, since they represent genuine calcium event sparsity.
- **No valid trial outcome**: A `ValueError` is raised if a trial has no outcome flag set (though this doesn't occur in practice).
- **Robust quintile edges**: If percentile edges are non-strictly-increasing (e.g., due to many identical values), they are perturbed by 1e-6.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
def robust_quintile_edges(values):
    ...
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Steps 9-10.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identified neural interpolation as the dominant cost. The two-pass design helps: pass 1 (global statistics, no neural loading) takes ~0.5s/session, while pass 2 (full conversion with neural interpolation) takes ~1.35s/session. Total runtime for 199 sessions was ~356 seconds.

ii.
```python
# Pass 1: ~0.5s/session (no neural)
raw = read_session_raw(session, load_events=False)
# Pass 2: ~1.35s/session (with neural)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The AI notes neural interpolation is the bottleneck because every trial requires resampling the full neuron x time event matrix.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop in pass 2 processes each trial sequentially. While neural interpolation within each trial is vectorized across neurons, the loop over trials within a session could potentially be vectorized by pre-computing all trial grids and doing batch interpolation. Also, the `stimulus_identity_codes` function has a Python loop over stimulus presentations:

ii.
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The AI notes that trial interpolation is vectorized across neurons but is still done per-trial. The inner loop over stimulus names could be vectorized with array operations.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in pass 1 (global statistics) and once in pass 2 (full conversion). In pass 1, running speed and pupil data are interpolated per trial to compute global quintile edges, and then the same interpolation is repeated in pass 2.

ii.
```python
# Pass 1
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
    pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# Pass 2
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The two-pass approach is a design choice to avoid storing all neural arrays while computing global bin edges. It trades compute time for memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `input` arrays are explicitly empty (`np.zeros((0, len(grid)))`) for every trial, yet the code still creates and stores them. The `flashes_since_change` column is loaded from stimulus presentations but never used. The `trials_id` column from stimulus presentations is also loaded but unused.

ii.
```python
input_trial = np.zeros((0, len(grid)), dtype=np.float32)
session_input.append(input_trial)
...
stim = read_interval_table(stim_group, [..., "trials_id", ..., "flashes_since_change"])
```

iii. The empty input arrays are required by the data format specification. The extra stimulus columns are loaded but unused, representing minor overhead.
