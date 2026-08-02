# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads metadata from `ophys_experiment_table.csv` to identify available experiment NWB files, then directly opens each NWB/HDF5 file using `h5py` (not the AllenSDK). For each session, it reads the trials table, stimulus presentations, ophys event traces with timestamps, running speed, and eye-tracking data from their respective HDF5 groups. A two-pass approach is used: pass 1 collects global statistics (running/pupil quintile edges, image vocabulary) without loading neural data; pass 2 loads the full data including neural traces and constructs per-trial arrays.

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
        ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"], ...)
        events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], ...)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], ...)
        ...
```

iii. The AI justified using `h5py` instead of the AllenSDK because the local `pynwb/hdmf` stack could not instantiate the NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` due to an `external_resources` abstract-method mismatch. The field semantics were verified to match the SDK objects.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses the `mouse_id` column from `ophys_experiment_table.csv` to identify unique subjects. A dictionary maps mouse_id strings to sequential indices, and each session is assigned the appropriate subject index.

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

iii. The AI documented that `mouse_id` from session metadata is the appropriate identifier, consistent with the SDK's `BehaviorSession.metadata` providing mouse identifiers.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (one per `ophys_experiment_id`) is treated as one session. The AI iterates over all active experiment files that have the required eye-tracking data.

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

iii. The AI justified treating each NWB experiment file as one decoder session because neural traces are experiment-specific (different imaging planes within the same behavior session have different neurons), and the whitepaper notes that multi-plane sessions can have up to 8 experiments per session.

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` table in each NWB file. Each trial has a `start_time` and `stop_time`. The time grid for each trial spans from `start_time` to `stop_time` at 30 Hz resolution.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The AI documented that trial boundaries come directly from the NWB `intervals/trials` table, matching the SDK's `Trials` object which computes `start_time` and `stop_time` from synchronized timestamps.

## 1-e. How are trials filtered based on quality controls?

i. The AI includes only "go" and "catch" trials and excludes "aborted" and "auto_rewarded" trials. Additionally, sessions with fewer than 2 valid trials are excluded.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session {session.ophys_experiment_id} ...")
    continue
```

iii. The AI justified this filtering rule based on the instructions ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials") and the SDK's trial logic which explicitly defines these categories.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the precomputed calcium event detection traces stored at `processing/ophys/event_detection/data` in the NWB files, along with associated timestamps at `processing/ophys/event_detection/timestamps`.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI justified using `events` (detected calcium events) rather than `dff_traces` because the strategy paper explicitly states its neural analyses used detected calcium events, making this the best match to the reference processing.

## 2-b. How is the `neural` data processed?

i. The raw event traces (shape: n_timepoints x n_neurons on ophys timestamps) are linearly interpolated onto a common 30 Hz time grid defined by the trial's `start_time` and `stop_time`. The interpolated matrix is then transposed to (n_neurons, n_timepoints).

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

iii. The AI justified the 30 Hz interpolation based on the strategy paper which "linearly interpolates onto common 30 Hz timestamps relative to behavioral events."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` flag from the NWB's `cell_specimen_table`. If the number of valid ROIs differs from the total number of event trace columns, only valid ROI columns are kept.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
n_neurons = events.shape[1]
```

iii. The AI justified this by referencing the SDK's `CellSpecimens.__init__(..., exclude_invalid_rois=True)` behavior which filters the ROI table to `valid_roi == True`. The CONVERSION_NOTES state: "Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering beyond reference QC."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the trial's `start_time`. Each trial's time grid starts at `start_time` and extends to `stop_time` in steps of 1/30 s. The metadata records `temporal_alignment_event: "trial start"` and `off_start: 0.0`.

ii.
```python
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
```

iii. The instructions state to "Temporally align based on ophys timestamp" which is ambiguous. The AI chose to align to trial start using ophys-synchronized timestamps as the time base, which is a reasonable interpretation since all data streams are already synchronized via the ophys timestamp framework.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 1/30 s (approximately 33.33 ms). All data streams are interpolated onto this common 30 Hz grid. This constitutes rebinning from the native ophys rate (~31 Hz single-plane or ~11 Hz multi-plane) and from the native behavior/eye-tracking rates.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
```

iii. The AI justified the 30 Hz time base because the strategy paper "linearly interpolates calcium event responses onto common 30 Hz timestamps" and behavior/eye tracking are natively 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, and `omitted` columns from the active task presentation group.

ii.
```python
stim = read_interval_table(stim_group, [
    "start_time", "stop_time", "image_name", "is_change", "omitted",
    "trials_id", "active", "flashes_since_change",
])
```

iii. The AI documented this in the variable mapping as using "stimulus-presentation `image_name`, `start_time`, `stop_time`, `omitted`, active task block only."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A categorical time-varying signal is constructed on the 30 Hz grid. For each time point, the code finds which stimulus presentation interval contains it (using `searchsorted`). If the time point falls within a non-omitted image presentation (between `start_time` and `stop_time`), the corresponding `image_name` is encoded as an integer. Otherwise, the time point is labeled as "gray" (code 0). The global category list is `["gray"] + sorted(unique_image_names)`.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
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

iii. The AI justified the explicit "gray" category because "the task includes 500 ms gray periods and omissions extend gray instead of showing an image, a gray category is needed for a complete time-varying identity signal."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz time grid as the neural data, using the same `grid` array derived from trial `start_time` and `stop_time`. Both neural and stimulus data are sampled at the same absolute times.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Alignment is ensured by computing all outputs on the same `grid` as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` and `omitted` columns of the stimulus presentations table, along with `start_time` and `stop_time`.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The AI documented this as using "stimulus-presentation `is_change`, `start_time`, `stop_time`."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time-varying signal is constructed on the 30 Hz grid. For each time point, the code checks whether it falls within a stimulus presentation interval that is marked as `is_change=True` (and not omitted). If so, the value is 1; otherwise 0. This means the "change" label spans the entire duration of the changed-image presentation window (~250 ms = ~7-8 bins).

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

iii. The AI initially used a single-bin impulse at `change_time` but found this was "too sparse" and caused below-chance decoder accuracy. They switched to marking the entire changed-image presentation window, documented as a design change in CONVERSION_NOTES Step 10.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already a binary variable (0 = no change, 1 = change) and does not require additional thresholding.

ii.
```python
["no_change", "change"],
```

iii. The instructions specify "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same 30 Hz `grid` as the neural data, using the same trial time grid.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same alignment approach as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The AI documented that this uses the SDK-equivalent filtered running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The raw running speed is linearly interpolated onto the trial's 30 Hz time grid using `np.interp`. Then it is discretized into 5 equal-percentile bins using global quintile edges computed across all valid trial timepoints in the dataset.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The AI justified using global quintile edges to ensure "class definitions are shared across sessions."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using the 20th, 40th, 60th, and 80th percentile values computed globally across all valid timepoints. `np.digitize` maps continuous values to bin indices.

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

iii. The instructions specify "discretized into five equal percentile bins." The AI adds a small epsilon to prevent degenerate edges where percentiles coincide.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz `grid` as the neural data.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Same alignment approach as all other outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height`, with timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

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

iii. The AI chose to compute pupil diameter as `max(width, height)` of the pupil tracking ellipse. This is a proxy for the major axis diameter of the fitted ellipse.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `max(width, height)` of the pupil tracking ellipse fit. NaN values (from blinks or invalid frames) are handled by interpolating only over finite samples. The valid pupil diameter values are linearly interpolated onto the trial's 30 Hz time grid, then discretized into 5 equal-percentile bins using global quintile edges.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    if valid.sum() == 1:
        return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. The AI documented that blink-related NaN gaps are filled by interpolation across valid timestamps, matching the reference SDK's blink masking approach.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 bins (0-4) using the same global quintile approach as running speed: 20th, 40th, 60th, and 80th percentile edges computed across all valid timepoints.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same justification as running speed binning.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz `grid` as the neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same alignment approach as all other outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB `intervals/trials` table.

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

iii. The AI documented these as the standard trial outcome flags matching the SDK's `Trial._get_trial_data` definitions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Each trial is assigned one of four outcome categories (hit=0, miss=1, false_alarm=2, correct_reject=3) based on which boolean flag is True. This static per-trial label is then broadcast across all time bins in the trial to create a time-varying representation.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The AI justified broadcasting the static label: "To satisfy the decoder format and simplify training, even static trial outcome will be repeated across all bins in a trial."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Missing eye tracking**: 3 active sessions lacking the `EyeTracking` group are excluded entirely before conversion.
- **NaN pupil values**: Blink/invalid frames produce NaN values in pupil data; these are handled by interpolating only over finite samples.
- **Sparse/zero neural events**: 2,467 trials (4.83%) have all-zero neural event matrices; these are kept as-is since they reflect genuine sparsity in the calcium event signal.
- **Degenerate quintile edges**: If percentile edges coincide, a small epsilon (1e-6) is added to prevent degenerate bins.
- **Sessions with too few trials**: Sessions with fewer than 2 valid trials after filtering are skipped.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
def robust_quintile_edges(values):
    ...
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. The AI documented each edge case discovery and resolution in CONVERSION_NOTES Steps 9-10, including verifying that all-zero neural trials are genuine raw-data zeros via `np.allclose` checks.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is neural interpolation in pass 2, where event traces for every neuron must be resampled onto the 30 Hz trial grid for every trial. The AI estimated pass 2 at ~18 minutes for the full dataset (scaling by neuron-bin work), compared to ~2 minutes for pass 1. The actual full conversion completed in ~356 seconds (~6 minutes).

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The AI documented timing per session and estimated full-run time, noting that "neural interpolation is still the dominant expected cost."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loop is the per-trial loop within each session (iterating over `np.flatnonzero(raw["keep_mask"])`). Within each trial, interpolation is vectorized across neurons, but the trial loop itself could potentially be vectorized by processing all trials for a session at once. Additionally, the `stimulus_identity_codes` function contains a Python for-loop over stimulus presentations within each trial that could be vectorized.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    # Per-trial processing loop

# Inner loop in stimulus_identity_codes:
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The AI noted that "trial-level interpolation is vectorized over neurons within each trial" as a speedup, but acknowledged the per-trial loop structure remains.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice (once in pass 1 for global statistics, once in pass 2 for full conversion). In each pass, it opens the HDF5 file, reads the trials table, stimulus presentations, running speed, and pupil data. It also computes the trial time grid and interpolates running speed and pupil data in both passes (pass 1 to collect global quintile edge statistics, pass 2 to construct the actual output arrays).

ii.
```python
# Pass 1:
def collect_global_statistics(sessions):
    for idx, session in enumerate(sessions):
        raw = read_session_raw(session, load_events=False)
        ...
        running_values.append(interpolate_vector(...))
        pupil_values.append(interpolate_pupil(...))

# Pass 2:
def convert_sessions(sessions, ...):
    for sess_num, session in enumerate(sessions):
        raw = read_session_raw(session)
        ...
        running_cont = interpolate_vector(...)
        pupil_cont = interpolate_pupil(...)
```

iii. The AI justified the two-pass design: "Two-pass design avoids storing all neural arrays while computing global bin edges" and "keeps memory bounded."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several aspects of processing are unnecessary for downstream decoding:
- **Reading stimulus columns not used**: The code reads `flashes_since_change` and `trials_id` from the stimulus table but never uses them for any output variable.
- **Storing empty input arrays**: The `input` field stores empty `(0, n_timepoints)` arrays for each trial since no decoder inputs are requested, adding structure overhead without information.
- **Broadcasting trial outcome across time**: Trial outcome is static per-trial but is broadcast to all time bins, creating redundant data. The decoder could use a single scalar per trial instead.
- **Loading `change_time` from trials**: The `change_time` column is read but never used in the final code (it was used in an earlier version with single-bin impulse change labels).

ii.
```python
stim = read_interval_table(stim_group, [
    ..., "trials_id", ..., "flashes_since_change",
])
...
input_trial = np.zeros((0, len(grid)), dtype=np.float32)
...
trials["change_time"] = trials["change_time"].astype(np.float64)
```

iii. The AI did not explicitly justify reading unused columns; these appear to be artifacts from exploratory development.
