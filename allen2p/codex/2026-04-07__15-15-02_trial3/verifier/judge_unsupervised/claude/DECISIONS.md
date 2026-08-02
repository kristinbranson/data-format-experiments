# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly using `h5py` rather than the AllenSDK high-level API. It first loads `ophys_experiment_table.csv` to discover available experiment files and their metadata (mouse_id, targeted_structure, session_type, etc.), then filters to only locally available, non-passive experiment NWB files. Each NWB file is opened with `h5py.File` and the relevant data groups are read: `intervals/trials`, `intervals/*_presentations`, `processing/ophys/event_detection`, `processing/running/speed`, `acquisition/EyeTracking/pupil_tracking`, and `processing/ophys/image_segmentation/cell_specimen_table`.

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
    # ... builds SessionMeta list
```

iii. The AI justified using `h5py` directly because the local AllenSDK/NWB stack could not instantiate the NWB files due to `external_resources`/version mismatch. It documented this in CONVERSION_NOTES.md Step 4 as an environment mismatch and chose to "mirror the AllenSDK/whitepaper semantics from the processed NWB contents."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mouse_id` from `ophys_experiment_table.csv`. A sorted list of unique mouse IDs across kept sessions forms the `subjects` list. Each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
# ...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The AI noted that mouse_id from the experiment metadata table is the canonical subject identifier, consistent with AllenSDK conventions.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` NWB file is treated as one session. This means each imaging plane within a multi-plane recording is a separate session.

ii.
```python
# In get_local_session_metadata:
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            # ...
            filepath=available_files[int(row.ophys_experiment_id)],
        )
    )
```

iii. The AI justified this by noting it "matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session." This is documented in CONVERSION_NOTES.md Step 5, Key Decision 2.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB `intervals/trials` table. Each row with appropriate trial type flags constitutes one trial, bounded by `start_time` and `stop_time`.

ii.
```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id", "start_time", "stop_time", "go", "catch", "aborted",
        "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "initial_image_name", "change_image_name",
    ]
    trials = read_interval_table(f["intervals"]["trials"], columns)
    # ...
    return trials
```

iii. The AI documented in Step 5 that it would "use the NWB trials table directly as the authoritative processed trial definition" rather than re-deriving trial logic from lower-level files.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only GO and CATCH trials, excluding aborted and auto-rewarded trials. Sessions with fewer than 2 kept trials are excluded entirely. Sessions missing eye tracking or task presentations are also excluded.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
# ...
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip ...")
    continue
```

iii. The AI referenced `trial_masks.contingent_trials` from the AllenSDK which defines contingent trials as GO and CATCH only, and noted this matches the user requirement to exclude aborted and auto-rewarded trials (CONVERSION_NOTES.md Step 1, Step 4).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the event detection traces stored at `processing/ophys/event_detection/data` in each NWB file, along with their timestamps at `processing/ophys/event_detection/timestamps`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI documented in Step 4 that "Paper methods explicitly use detected calcium events for neural analyses" and chose to "Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F."

## 2-b. How is the `neural` data processed?

i. The raw event detection matrix (time x ROI) is linearly interpolated onto a synthetic 30 Hz trial grid using vectorized `searchsorted`-based interpolation. The output is transposed to (n_neurons, n_timepoints) per trial.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1
    t0 = src_time[idx_lo]
    t1 = src_time[idx_hi]
    denom = np.where(t1 > t0, t1 - t0, 1.0)
    w = ((dst_time - t0) / denom).astype(np.float32)
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
# Called as:
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI justified 30 Hz resampling by noting that "native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane)" and that "30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not explicitly filter by `valid_roi`. It reads all cells from the `cell_specimen_table` and all rows from `event_detection/data`. The AI documented that all 29,168 cells in the local NWB files were already marked as valid, so no filtering was needed in practice.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The AI noted in Step 10 Check 3 that "included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed), so event matrices matched converted neuron counts exactly."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to trial start time. For each trial, bin centers are constructed from `start_time` to `stop_time` at 30 Hz intervals. Neural event traces are then linearly interpolated from their native ophys timestamps onto these bin centers.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
# In convert_session:
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI justified alignment to trial start: "For each trial, create bin centers from trial start_time to stop_time at 30 Hz and sample/interpolate all streams onto that grid." The metadata records `temporal_alignment_event` as "trial start" with `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a 30 Hz grid (33.333 ms bins). This is NOT the native ophys sampling rate. It is a synthetic common grid. All signals (neural, running, pupil, stimulus) are resampled onto this grid via linear interpolation (for continuous signals) or time-window masking (for categorical signals).

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
# In metadata:
"time_bin_size": TIME_BIN_MS,  # 33.333 ms
"sampling_grid_hz": 30.0,
```

iii. The AI chose 30 Hz because native rates differ (31 Hz single-plane, 11 Hz multi-plane) and this "preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the stimulus presentation interval tables (`intervals/*_presentations`), filtered to rows within the `change_detection` stimulus block. The `omitted` flag is also used to identify omitted presentations.

ii.
```python
# In get_task_presentations:
block_names = decode_str_array(group["stimulus_block_name"][:])
keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
# ...
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", ...]
```

iii. The AI documented that stimulus identity should come from the processed NWB stimulus presentation tables, consistent with AllenSDK's `get_stimulus_presentations` function.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A categorical trace is built per trial on the 30 Hz grid. Default value is "gray" (index 0). For each non-omitted stimulus presentation within the trial, time bins overlapping the presentation window are set to the corresponding image name index. Omitted presentations and gaps between stimuli are left as "gray". All unique image names across sessions are collected and sorted to create a global categorical encoding.

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

iii. The AI noted that "image_identity will include a gray category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned using the same 30 Hz trial bin centers as neural data, via time-window masking (`centers >= start_time & centers < stop_time`). This ensures temporal alignment with the resampled neural traces.

ii. Same code as 3-b above. The `centers` array is shared between neural interpolation and output construction.

iii. The AI planned consistent alignment across all data streams on the common 30 Hz grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentation tables within the `change_detection` block.

ii.
```python
# In get_task_presentations columns:
columns = [..., "is_change", ...]
# In convert_session:
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI documented that the `is_change` flag from stimulus presentations would be used, consistent with AllenSDK's `is_change_event` function.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is initialized to 0 for each trial. For each stimulus presentation flagged as `is_change`, all time bins overlapping that presentation's window are set to 1. This means the "change" signal persists for the duration of the change stimulus presentation (250 ms), not just a single time bin.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
# In the loop over presentations:
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The instructions state "Have value of 1 right after a change in image identity, otherwise 0." The AI interpreted this as marking the entire change stimulus window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1) and requires no thresholding. The output values are `["no_change", "change"]`.

ii.
```python
"output_values": [
    # ...
    ["no_change", "change"],
    # ...
]
```

iii. Binary by definition per the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz trial grid as neural data, using time-window masking on presentation start/stop times.

ii. Same `centers` and `mask` logic as image identity (see 3-c).

iii. Consistent alignment approach across all outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in each NWB file.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The AI noted this is the processed (already lowpass-filtered) running speed stored in the NWB, matching the `RunningSpeed` data object in the SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The pre-processed running speed from the NWB is linearly interpolated onto the 30 Hz trial grid using `np.interp`. Then it is discretized into 5 equal-percentile bins using globally computed quantile edges. The global edges are computed in Pass 1 across all finite running speed samples from all kept sessions and trials.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
# Global edges from Pass 1:
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The AI chose global quantile binning so that "class semantics are consistent dataset-wide" (CONVERSION_NOTES.md Step 5, Key Decision 10).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins (q1 through q5, encoded as integers 0-4). Bin edges are computed using `np.quantile` with probabilities [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] across all finite running speed values globally. Values are clipped to edge bounds before binning.

ii.
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    if np.unique(edges).size < edges.size:
        eps = np.finfo(np.float64).eps
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
    return edges

def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```

iii. The instructions specify "discretized into five equal percentile bins."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid used for neural data via `linear_resample_vector` (which calls `np.interp`).

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same common grid alignment approach as all other signals.

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

iii. The AI documented it would "Compute pupil diameter as `2 * max(width, height)` after blink filtering."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `2.0 * max(pupil_width, pupil_height)`. NaN values (from blink filtering already applied in the NWB) are filled by temporal interpolation. The result is then linearly interpolated onto the 30 Hz trial grid and globally discretized into 5 equal-percentile bins.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
# ...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The reference SDK computes `pupil_area = pi * max(width, height)^2` (circular area assumption). The AI instead computes diameter as `2 * max(width, height)`. Since the instructions say "Pupil diameter," the AI's choice to use diameter rather than area is defensible. However, since the quantile binning is a monotonic transformation, the bin assignments would be identical whether diameter or area is used.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins computed globally across all finite pupil diameter values from all kept sessions/trials.

ii. Same `compute_quantile_edges` and `digitize_with_edges` functions as running speed.

iii. Consistent with instruction to discretize into five equal percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 30 Hz trial grid alignment via `linear_resample_vector`.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Consistent alignment across all signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the mutually exclusive boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB `intervals/trials` table.

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

iii. The AI referenced the AllenSDK `Trial._get_trial_data` which defines these as mutually exclusive outcome categories for contingent trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is encoded as a single integer per trial (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). Despite being a static per-trial variable, it is represented as a constant time-varying trace (same value repeated for all time bins in the trial) to maintain a consistent `(n_output, T)` output format.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
output_trial = np.vstack([
    image_identity,
    image_change,
    running_bin,
    pupil_bin,
    outcome_trace,
])
```

iii. The instructions specify trial outcome as "Static per-trial." The AI chose to represent it as a constant trace to keep one consistent output format. This means the decoder sees it as time-varying even though it's constant within each trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies:
- **Missing eye tracking**: Sessions without `EyeTracking` acquisition are excluded entirely (3 active sessions).
- **Blink-related NaN in pupil data**: NaN values are filled by temporal interpolation using `fill_nan_by_time()` before resampling.
- **Zero-event trials**: Trials where the event detection matrix is all zeros are retained. The AI verified these come from genuine sparse event detections in the source data, not conversion bugs (4.84% of kept trials).
- **Missing trial bins**: Trials where `stop_time <= start_time` or with non-finite bounds produce empty bin arrays and are effectively skipped (though sessions must still have >= 2 valid trials).
- **Duplicate quantile edges**: When quantile bin edges are not unique, machine epsilon offsets are added to ensure strict monotonicity.

ii.
```python
def fill_nan_by_time(time_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("All values are NaN")
    if finite.all():
        return values
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values
```

iii. The AI documented investigation of zero-event trial warnings in Step 10, confirming via raw NWB inspection that "the underlying event-detection segment itself was exactly zero across all neurons and frames."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Two-pass NWB I/O**: Each NWB file is read twice -- once in Pass 1 (collecting global running/pupil statistics) and once in Pass 2 (converting trials). This doubles I/O time.
2. **Per-trial interpolation**: Within each session, every trial requires separate interpolation of neural events, running speed, and pupil diameter onto the trial grid.
3. **Full-dataset conversion**: 199 sessions processed sequentially, total ~395 seconds.

ii.
```python
# Pass 1:
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
# Pass 2:
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The AI estimated ~2.22 seconds per session and ~7.5 minutes total, noting I/O as the primary bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several inner loops could be vectorized:
1. **Stimulus presentation loop**: The per-presentation loop in `convert_session` iterates over each stimulus presentation row within a trial to set image_identity and image_change masks. This could be vectorized using `np.searchsorted` on presentation boundaries.
2. **Trial loop in Pass 1**: The per-trial resampling in `collect_global_statistics` could potentially be replaced by session-level concatenation of running/pupil values without per-trial binning.
3. **`decode_str_array`**: The element-wise string decoding loop could use vectorized numpy string operations.

ii.
```python
# Per-presentation loop (could be vectorized):
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if not mask.any():
        continue
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The AI noted "Vectorized linear interpolation for neural event matrices using searchsorted + broadcasting" as a speedup, but did not vectorize the stimulus presentation or string decoding loops.

## 9-c. What processing does the code repeat multiple times?

i. The primary repeated processing is:
1. **Reading each NWB file twice**: Pass 1 reads trials, presentations, running, and pupil data to compute global statistics. Pass 2 reads the same data again to perform the actual conversion.
2. **Trial table filtering**: The same `(go | catch) & ~aborted & ~auto_rewarded` filter is applied both in Pass 1 and Pass 2.
3. **Running and pupil resampling**: In Pass 1, running and pupil are resampled per-trial to collect global statistics. In Pass 2, the same resampling is done again for the actual conversion.

ii.
```python
# Pass 1 - collect_global_statistics:
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)

# Pass 2 - convert_session (same data re-read):
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. The AI acknowledged this: "Global binning requires a first pass over sessions, so conversion reads each file twice."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several aspects of the processing are unnecessary for the final output:
1. **Per-trial resampling in Pass 1**: Running and pupil data are resampled per-trial in Pass 1 just to collect global statistics for binning. Instead, the raw session-level running/pupil values could be used directly (without per-trial windowing) to compute quantile edges, since the bin edges should reflect the overall distribution.
2. **Image name collection in Pass 1**: Pass 1 collects all image names across sessions to build the categorical encoding. This requires reading stimulus presentation tables in Pass 1 even though the actual image traces are only constructed in Pass 2.
3. **Processing plots**: The `--show-processing` mode generates diagnostic plots that are not used in the final pickle output.

ii.
```python
# Pass 1 unnecessarily resamples per-trial:
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
    running_values.append(running_trial)
    pupil_values.append(pupil_trial)
```

iii. The AI did not explicitly document these inefficiencies. The two-pass design was chosen for correctness (global bins needed before per-trial discretization) but the per-trial resampling in Pass 1 adds unnecessary computation.
