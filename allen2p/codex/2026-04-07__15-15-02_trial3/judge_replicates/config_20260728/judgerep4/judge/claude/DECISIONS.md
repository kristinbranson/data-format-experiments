# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly with `h5py` rather than using the AllenSDK's `VisualBehaviorOphysProjectCache`. It discovers available experiments from a CSV metadata table (`ophys_experiment_table.csv`) cross-referenced with locally available `.nwb` files on disk. Only experiments with local NWB files present are included. Passive sessions are filtered out. Each NWB file is opened with `h5py.File` and relevant groups (trials, neural, running, pupil, stimulus presentations) are read directly.

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
```

iii. The AI explains in CONVERSION_NOTES.md Step 4 that the local AllenSDK/NWB stack could not instantiate the NWB files due to an `external_resources`/version mismatch, so it chose to read NWB files directly with `h5py` and mirror AllenSDK semantics from the processed NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table. Each unique mouse_id becomes a subject.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard Allen SDK identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (i.e., each NWB file, corresponding to one imaging plane) is treated as a separate session. This means a single recording session with multiple imaging planes produces multiple "sessions" in the converted data.

ii.
```python
# From get_local_session_metadata - each experiment file becomes a session
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ...
    ))
```

iii. The AI states in CONVERSION_NOTES.md Step 5: "Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session."

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table from the NWB file. For each non-aborted, non-auto-rewarded trial that is either a go or catch trial, a uniform 30 Hz time grid is built from `start_time` to `stop_time` using `build_trial_bins`. This produces variable-length trials.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]

trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
```

iii. The AI explains that this mirrors the Allen reference processing without re-deriving trial logic from lower-level files. It uses 30 Hz bin centers within each trial's start-to-stop window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only GO and CATCH trials, excluding aborted and auto-rewarded trials. Sessions with fewer than 2 usable trials are excluded. Sessions missing eye tracking data are also excluded.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```
In `collect_global_statistics`:
```python
if len(trials) < 2:
    print(f"... skip ... fewer than 2 kept trials")
    continue
```

iii. Per the instructions, aborted and auto-rewarded trials are excluded. The minimum 2-trial threshold prevents degenerate sessions. Sessions without eye tracking are excluded because pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `event_detection` data in the NWB file (`processing/ophys/event_detection`), which contains calcium event magnitudes extracted by the Allen pipeline's FastLZero algorithm.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI explains in CONVERSION_NOTES.md Step 4: "Paper methods explicitly use detected calcium events for neural analyses" and references `methods.txt:179` and `methods.txt:208`. The AI chose events over dF/F to match the paper's neural analysis approach.

## 2-b. How is the `neural` data processed?

i. The raw event detection data (time x ROI matrix) is linearly resampled from native ophys timestamps onto a common 30 Hz trial grid using a vectorized interpolation method. The result is transposed to (n_neurons, n_timepoints).

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

neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI states that resampling to 30 Hz gives a shared bin size while remaining close to the behavior/eye-tracking rate and is consistent with the paper's event-triggered interpolation approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons present in the NWB file's event detection matrix are included. The AI notes that the NWB files already contain only valid ROIs. No additional quality filtering is applied beyond what the Allen pipeline already did.

ii. No explicit filtering code - the AI reads all neurons from the event_detection group.

iii. The AI verified in Step 10 that "included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. A 30 Hz grid of bin centers is constructed from `start_time` to `stop_time`, and the event data is linearly interpolated onto this grid.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI states this aligns by absolute ophys time, cutting into trials based on the NWB trial table boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to a common 30 Hz grid (33.33 ms bins). This is a change from the native ophys sampling rate, which varies between 11 Hz (multiplane) and 31 Hz (single-plane).

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

iii. The AI explains: "Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB file (`intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, `omitted`, `is_change`, `trials_id`, and `stimulus_block_name` columns. This is filtered to the `change_detection` stimulus block.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
```

iii. The AI chose to use the stimulus presentation tables because they provide per-flash timing, enabling frame-level image identity including gray inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global mapping of image names to integer codes is built, with `gray` at index 0. For each trial, each time bin is assigned the image code based on which stimulus presentation (if any) covers that time point. Bins not covered by any presentation, or covered by omitted presentations, are assigned `gray`. The image values include `gray` plus all observed image names.

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

iii. The AI states that this approach captures the actual visual stimulus on screen at each moment, including gray periods between flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid (`centers`) as the neural data, so alignment is inherent.

ii. Both use the same `centers` array from `build_trial_bins`.

iii. The AI notes that all streams are sampled onto the same 30 Hz grid, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentation table. When a presentation row has `is_change == True`, the corresponding time bins are set to 1.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI uses the stimulus presentations table's `is_change` flag, which is computed by the Allen SDK pipeline.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is initialized to 0 for each trial. For each stimulus presentation within the trial that has `is_change == True`, the time bins overlapping that presentation's `[start_time, stop_time)` window are set to 1.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. This marks the change flash period as 1, covering approximately 250ms (the stimulus duration).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. See 4-b above.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the same 30 Hz trial grid.

ii. Uses the same `centers` array.

iii. Same alignment mechanism as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides timestamps and speed values.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. This is the Allen SDK's standard running speed data stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps onto the 30 Hz trial grid using `np.interp`. It is then discretized into 5 equal-percentile bins computed globally across all sessions. Bin edges are computed from all finite values across all included sessions/trials.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Percentile-based binning ensures roughly equal class counts. Global bin edges maintain consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global percentile edges. Values are clipped to the edge range before digitization.

ii.
```python
def digitize_with_edges(values, edges):
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```

iii. Clipping prevents out-of-range values from producing invalid bin indices.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid (`centers`) used for neural data.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same alignment as all other outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking` in the NWB file, specifically the `width` and `height` fields. Diameter is computed as `2 * max(width, height)`.

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

iii. The AI uses `2 * max(width, height)` as an estimate of pupil diameter from the ellipse fit. It fills NaN values by time interpolation before returning.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (`2 * max(width, height)`) is first computed, NaN values are filled by linear interpolation in time (`fill_nan_by_time`), then the result is linearly resampled onto the 30 Hz trial grid. Finally, it is discretized into 5 equal-percentile bins computed globally.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. NaN filling before interpolation prevents missing data from propagating. The AI notes that sessions without eye tracking are excluded entirely.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 global percentile bins with clipping.

ii. Same `digitize_with_edges` function as running speed.

iii. Same approach for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - interpolated onto the same 30 Hz trial grid.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same alignment mechanism.

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

iii. These are the standard Allen SDK trial outcome labels, checked in priority order.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is encoded as an integer (0-3) corresponding to hit/miss/false_alarm/correct_reject. It is replicated as a constant trace across all time bins in the trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The outcome is static per trial, so it is repeated across all time bins to maintain the `(n_output, T)` format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` in the NWB are excluded entirely (KeyError caught in pass 1).
- **NaN pupil values**: Filled by time interpolation via `fill_nan_by_time` before resampling.
- **Empty trials**: Trials where `build_trial_bins` produces no bins (invalid start/stop) are skipped.
- **Sessions with < 2 trials**: Excluded with a RuntimeError in pass 2 (caught in pass 1 filtering).
- **All-zero neural trials**: Retained because they reflect genuine sparse event detections in the source data.

ii.
```python
# Missing eye tracking
except KeyError as exc:
    print(f"... skip ... {exc}")

# NaN pupil
diameter = fill_nan_by_time(timestamps, diameter)

# Empty trials
if centers.size == 0:
    continue
```

iii. The AI documented that 4.84% of trials had all-zero event traces, verified from raw NWB data to be genuine rather than a conversion bug.

## 9-a. What are the most time-consuming steps of the code?

i. The code uses a two-pass approach. Pass 1 reads each NWB file to collect running/pupil statistics and filter sessions. Pass 2 reads each NWB file again to perform the full conversion. The most time-consuming step is I/O: reading the NWB files from disk, particularly the neural event matrices. Full conversion completed in ~396 seconds for 199 sessions.

ii. N/A

iii. The AI notes this in CONVERSION_NOTES.md Step 6 and estimates ~2.22 seconds per session.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial, performing stimulus presentation matching via a nested loop over presentation rows. This inner loop could potentially be vectorized using interval-based operations.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The AI vectorized the neural interpolation with `searchsorted` + broadcasting, but the stimulus presentation matching loop was not vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design reads each NWB file twice: once in pass 1 for global statistics collection, and once in pass 2 for full conversion. This includes re-reading neural data, running speed, pupil data, and trial tables.

ii.
```python
# Pass 1: collect_global_statistics
for idx, session in enumerate(sessions, start=1):
    with h5py.File(session.filepath, "r") as f:
        ...

# Pass 2: convert_session
for idx, session in enumerate(kept_sessions, start=1):
    # Opens same file again
    with h5py.File(session.filepath, "r") as f:
        ...
```

iii. The AI notes: "Global binning requires a first pass over sessions, so conversion reads each file twice."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code rebins neural data from native ophys rate to 30 Hz. For single-plane experiments at 31 Hz, this is a very slight downsampling that introduces interpolation artifacts without meaningfully changing the data. For multiplane experiments at 11 Hz, the 30 Hz resampling upsamples data, creating interpolated values that don't add real information.

ii.
```python
DT = 1.0 / 30.0  # 30 Hz resampling
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI justifies this as giving "one shared bin size while remaining close to behavior/eye-tracking rate."
