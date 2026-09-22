# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly using `h5py` rather than going through the AllenSDK's `VisualBehaviorOphysProjectCache`. It discovers available experiments by reading `ophys_experiment_table.csv` from the local data directory, cross-references with actually present `.nwb` files on disk, and filters to active (non-passive) sessions. Each NWB file is opened individually with `h5py.File` to extract neural, behavioral, and trial data.

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
    ...
```

iii. The AI documented that the local AllenSDK/NWB stack could not instantiate these NWB files due to version mismatches (`external_resources` issue), so direct `h5py` reading was used as a workaround. The AI mirrors AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on high-level loading.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table. They are collected from the kept sessions list after filtering.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one NWB file = one imaging plane) is treated as an independent session. The AI does **not** group multiple imaging planes from the same `ophys_session_id` into a single session.

ii.
```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    ...
    for row in exp_table.itertuples(index=False):
        sessions.append(
            SessionMeta(
                ophys_experiment_id=int(row.ophys_experiment_id),
                ...
            )
        )
    return sessions
```

iii. The AI stated: "Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session." This is documented in CONVERSION_NOTES.md Step 5 Key Decision #2.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. For each non-aborted, non-auto-rewarded go or catch trial, a time window from `start_time` to `stop_time` is used. Within that window, a 30 Hz grid of bin centers is constructed to create the trial's time axis.

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

iii. The trial table from the NWB file provides pre-computed trial metadata matching Allen processing. The 30 Hz grid gives a common temporal resolution for all trials/sessions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only go and catch trials, excluding aborted and auto-rewarded trials. Sessions missing eye tracking data are excluded entirely. Sessions with fewer than 2 valid trials are excluded. Additionally, passive sessions are excluded at the experiment-metadata level.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
```
And for eye tracking:
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

iii. The filtering matches the instruction requirements: include go and catch trials, exclude aborted and auto-rewarded. The eye tracking requirement ensures pupil diameter output is available. The <2 trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection` in the NWB files -- i.e., calcium event magnitudes from the Allen event detection pipeline, **not** dF/F traces.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI justified this by noting: "Paper analyses frequently use discrete calcium events rather than raw dF/F" (from methods.txt), and "Use raw event magnitude traces ... as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F." (CONVERSION_NOTES Step 4).

## 2-b. How is the `neural` data processed?

i. The event magnitude matrix (time x neurons in NWB) is transposed to (neurons x time). For each trial, the event traces are linearly interpolated from native ophys timestamps onto a common 30 Hz grid of bin centers within the trial window.

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

iii. The resampling to 30 Hz was justified as giving a common bin size across sessions with different native acquisition rates (31 Hz single-plane, 11 Hz multiplane), consistent with the behavior/eye-tracking rate of 30 Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond what is already in the NWB files. The AI noted that the NWB `cell_specimen_table` already contains only valid ROIs for the available data subset. Sessions missing eye tracking are excluded entirely (not a neural quality filter per se).

ii. No specific filtering code; the neural data matrix is read as-is from `event_detection/data`.

iii. The AI verified that all listed cells in the available NWB files had `valid_roi == True` (29,168 total valid of 29,168 total listed), so no additional ROI filtering was needed for this data subset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start time. A 30 Hz grid of bin centers is created from `start_time` to `stop_time`, and the event traces are linearly interpolated onto those bin centers.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Alignment is to trial start, and the variable-length window captures both pre- and post-change periods. The AI metadata records `temporal_alignment_event: "trial start"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to a fixed 30 Hz grid (DT = 1/30 s = 33.33 ms bins). This is an explicit rebinning from the native ophys rate.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
```

iii. The AI chose 30 Hz because: "Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate." (CONVERSION_NOTES Step 4/5).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB file (`intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `stimulus_block_name` fields. Only presentations from the `change_detection` stimulus block are used.

ii.
```python
presentations = get_task_presentations(f)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. Using stimulus presentations gives frame-by-frame image identity that correctly captures gray inter-stimulus intervals and omitted flashes. The AI includes a "gray" category for periods when no image is on screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all sessions, with "gray" placed at index 0. Time bins within stimulus presentation intervals get the corresponding image code; all other bins (gray screen, ISI, omissions) get the "gray" code.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
```

iii. A global mapping ensures consistent integer codes across sessions. The "gray" category is prioritized at index 0 for the default/baseline state.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz bin centers as the neural data. For each trial, stimulus presentation intervals are overlaid onto the bin centers to determine which image (or gray) is present at each time point.

ii.
```python
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. Both neural and image identity use the same `centers` array, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation table's `is_change` flag, which marks presentation intervals where the image changed from the previous non-omitted image.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The `is_change` flag is computed by the Allen SDK processing and embedded in the NWB stimulus presentation table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is constructed per trial: 1 during stimulus presentation intervals marked `is_change`, 0 elsewhere. The change indicator covers only the duration of the change-image flash (typically ~250ms), not a full stimulus cycle.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. No additional processing beyond applying the pre-computed `is_change` flag from the presentation table.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii. See 4-b above.

iii. The binary representation directly matches the instruction: "Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz bin centers as neural data and image identity. The change indicator is set during the stimulus presentation interval marked as a change.

ii. See 4-b above.

iii. All outputs share the same temporal grid.

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

iii. This is the standard filtered running speed from the Allen SDK processing pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps onto the 30 Hz trial grid using `np.interp`, then discretized into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
# linear_resample_vector uses np.interp
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Linear interpolation preserves the signal shape. Global percentile binning ensures consistent categories across sessions. `np.interp` extrapolates to edge values rather than producing NaN.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins are computed from all finite running speed values across all sessions. Values are then digitized using `np.searchsorted` on the inner bin edges. Values outside the range are clipped to the outermost bins.

ii.
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

iii. Equal-percentile binning ensures roughly balanced class counts, which is important for decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz bin centers used for neural data and all other outputs.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Shared temporal grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking` in the NWB file, specifically the `width` and `height` fields. The diameter is computed as `2 * max(width, height)`.

ii.
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
    ...
```

iii. The AI chose to compute diameter from the pupil ellipse fit dimensions, taking `2*max(width, height)` as the diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After computing `2*max(width, height)`, NaN values (from blinks or tracking failures) are filled by time-based interpolation (`fill_nan_by_time`). The resulting trace is then linearly resampled onto the 30 Hz trial grid using `np.interp`, and finally discretized into 5 equal-percentile bins globally.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. NaN filling before resampling prevents NaN propagation. Global percentile binning is consistent with running speed processing.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins computed globally across all sessions, then digitized with `searchsorted`.

ii. Same as 5-c but with pupil values.

iii. Equal-percentile binning for balanced classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 30 Hz bin centers as neural and all other outputs.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Shared temporal grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trial table.

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

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task. They are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed priority order (hit=0, miss=1, false_alarm=2, correct_reject=3). The code is replicated across all time bins within a trial as a constant trace.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The mapping is deterministic and matches `OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]`. The constant trace allows a consistent `(n_output, T)` format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` acquisition in the NWB are excluded entirely (3 active sessions skipped).
- **NaN pupil values**: Filled by time-based linear interpolation before resampling via `fill_nan_by_time`.
- **Empty trials**: If `build_trial_bins` returns an empty array (invalid start/stop times), the trial is skipped.
- **Sessions with too few trials**: Sessions with fewer than 2 valid trials after filtering are excluded.
- **All-zero neural trials**: Retained since they are valid in the source data (sparse event detection).

ii.
```python
# fill_nan_by_time for pupil
def fill_nan_by_time(time_axis, values):
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    ...
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values

# Missing eye tracking
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

iii. The approaches are conservative: exclude sessions without required data, interpolate small gaps, retain all valid trials even if neural activity is sparse.

## 9-a. What are the most time-consuming steps of the code?

i. The two-pass design means each NWB file is read twice (once for global statistics, once for conversion). Each pass involves reading large event matrices and behavioral data from disk, which is I/O bound.

ii. From timing output: Pass 1 takes ~0.36s/session, Pass 2 takes ~1.86s/session. Total conversion was ~395s for 199 sessions.

iii. The AI documented this in CONVERSION_NOTES Step 7 Run Time Estimates.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each valid trial sequentially, constructing bin centers and interpolating neural/behavioral data for each trial. The per-presentation loop within each trial also iterates sequentially. These could theoretically be vectorized.

ii.
```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(...)
    neural_trial = linear_resample_matrix(...)
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The AI noted that data loading dominates runtime, so vectorizing these loops would provide marginal speedup. The neural interpolation itself is already vectorized across neurons.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is read twice: once in `collect_global_statistics` (pass 1) for running/pupil percentile edges, and again in `convert_session` (pass 2) for the actual conversion. This means all data loading, trial filtering, running speed reading, pupil reading, and stimulus presentation parsing is done twice.

ii.
```python
# Pass 1
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
# Pass 2
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session, ...)
```

iii. The two-pass approach is a design trade-off: it avoids holding all raw data in memory at once, but doubles the I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores detailed metadata about each session (ophys_experiment_ids, behavior_session_ids, etc.) that is not used by the decoder. The `fill_nan_by_time` processing fills NaN values that would otherwise be handled by the discretization step. The two-pass design reads neural data in pass 1 even though only running/pupil statistics are needed.

ii. Pass 1 reads neural data (via `get_neural_data`) even though it's only needed in pass 2 -- though looking more carefully, pass 1 doesn't call `get_neural_data`, it only reads running and pupil data. The unnecessary processing is relatively minimal.

iii. The code is reasonably efficient for its design. The main inefficiency is the two-pass I/O rather than unnecessary computation.
