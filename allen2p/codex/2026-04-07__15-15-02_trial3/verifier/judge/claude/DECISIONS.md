# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly with `h5py` rather than using the AllenSDK's `VisualBehaviorOphysProjectCache`. It discovers available experiments by reading `ophys_experiment_table.csv` from the local data directory, then filters to only locally-present active NWB files. Each file is read twice: once in Pass 1 to collect global running/pupil statistics, and once in Pass 2 for full trial conversion.

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
```

iii. The AI justified using h5py because the local AllenSDK/NWB stack could not instantiate these NWB files due to environment/version incompatibility. The AI documented this in CONVERSION_NOTES.md Step 4: "Read NWB files directly with h5py and mirror the AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in this environment."

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment metadata table. The AI collects unique mouse IDs from the kept sessions after filtering.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Mouse IDs are the canonical subject identifiers from the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` (i.e., each NWB file / imaging plane) as a separate session, rather than grouping multiple imaging planes by `ophys_session_id`. This means multi-plane recording sessions are split into multiple "sessions" in the converted data, each with its own set of neurons from one plane.

ii.
```python
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(
        session=session, ...)
    neural_all.append(neural_trials)
    ...
    subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
    brain_region_idx_all.append(brain_region_idx)
```

iii. The AI documented this in CONVERSION_NOTES.md Step 5: "Treat each ophys_experiment_id file as one converted session: This matches the AllenSDK object granularity (BehaviorOphysExperiment) and yields a single imaging plane / neuron set / brain region per session."

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. Go and catch trials are kept; aborted and auto-rewarded trials are excluded. Each trial spans from `start_time` to `stop_time` (variable length). The AI builds a uniform 30 Hz time grid within each trial using `build_trial_bins`.

ii.
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
```

```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

iii. The AI documented using the NWB trials table directly as the "authoritative processed trial definition" and filtering to GO/CATCH only, matching Allen reference processing.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only go and catch trials, (2) excluding aborted trials, (3) excluding auto-rewarded trials, (4) requiring non-empty trial bins from `build_trial_bins`, (5) requiring at least 2 usable trials per session. Additionally, passive sessions are excluded at the session level, and sessions without eye tracking are excluded.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. The AI justified these filters based on the instructions (exclude aborted and auto-rewarded), reference code (`trial_masks.contingent_trials`), and practical requirements (need at least 2 trials for decoder evaluation, need eye tracking for pupil output).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `event_detection` data (calcium event magnitudes) from the NWB file's `processing/ophys/event_detection` group, NOT the `dff_traces` (dF/F) used by the reference solution.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI justified this in CONVERSION_NOTES.md Step 4: "Paper methods explicitly use detected calcium events for neural analyses" and "Use raw event magnitude traces from processing/ophys/event_detection/data as neural. Do not use filtered_events (visualization-only) and do not recompute dF/F."

## 2-b. How is the `neural` data processed?

i. The neural event data is linearly resampled from its native ophys timestamps onto a common 30 Hz trial grid using vectorized linear interpolation. Each trial gets a `(n_neurons, n_timepoints)` matrix at the resampled rate.

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
...
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI justified the 30 Hz resampling in CONVERSION_NOTES.md Step 4: "Resample all streams to a common 30 Hz grid... This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural quality filtering is applied in the conversion code. The NWB files in the local subset already had all listed cells as valid (29,168 valid of 29,168 total listed). Passive sessions are excluded at the session level.

ii. N/A (no filtering code)

iii. The AI documented in CONVERSION_NOTES.md Step 10: "included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed), so event matrices matched converted neuron counts exactly."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, uniform 30 Hz bin centers are generated from `start_time` to `stop_time`, and the neural event traces are linearly interpolated onto those bin centers.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI stated: "Align by absolute ophys time, then cut into trials: For each trial, create bin centers from trial start_time to stop_time at 30 Hz and sample/interpolate all streams onto that grid."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a uniform 30 Hz grid (DT = 1/30 s = 33.33 ms bins). This is a rebinning from the native ophys rate (31 Hz single-plane, 11 Hz multiplane) to a common rate matching the behavior/eye-tracking sampling frequency.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
```

iii. The AI justified in CONVERSION_NOTES.md Step 4: "Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB file (`intervals/*_presentations`), specifically from the `image_name`, `start_time`, `stop_time`, `omitted`, and `trials_id` columns. A `gray` category is used for inter-stimulus intervals and omitted flashes.

ii.
```python
presentations = get_task_presentations(f)
...
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The AI justified using stimulus presentations rather than trial-level image names: "Use stimulus presentation interval tables to build time-varying image identity and image-change signals on the resampled trial grid."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is computed as a time-varying categorical trace. The default is `gray` (for ISI/omitted periods), and during each non-omitted stimulus flash, the trace is set to the corresponding image code. A global mapping from image names to integer codes is built, with `gray` as index 0.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The AI documented: "Encode gray/omission periods explicitly: image_identity will include a gray category for ISI and omitted-image periods."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz bin centers as the neural data for each trial. Each bin center is checked against the start/stop times of stimulus presentations to determine which image (if any) is being shown.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. Same 30 Hz trial grid ensures alignment with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` flag in the stimulus presentation interval tables, combined with the flash timing (`start_time`, `stop_time`).

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI used the stimulus presentation `is_change` flag to mark change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary time-varying trace. It is 0 by default, and set to 1 during the time bins that fall within a stimulus presentation marked as `is_change`. This means it is 1 only during the change flash period (~250 ms), not during the entire post-change window.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI used the stimulus presentation timing to determine when the change occurs.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz bin centers as neural data. Each bin is checked against the change flash presentation timing.

ii. Same as 3-c and 4-b.

iii. Same trial grid ensures alignment.

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

iii. This is the SDK's standard running speed data, equivalent to `dataset.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`, then discretized into 5 equal-percentile bins computed globally across all sessions. Values are clipped to the bin edge range before digitization.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
def digitize_with_edges(values, edges):
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```

iii. Global percentile-based binning ensures consistent bin semantics across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. 5 equal-percentile bins are computed globally from all finite running speed values across all sessions. `compute_quantile_edges` computes edges at 0%, 20%, 40%, 60%, 80%, 100% percentiles, with a tie-breaking epsilon adjustment if edges are not unique.

ii.
```python
def compute_quantile_edges(values, nbins):
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    if np.unique(edges).size < edges.size:
        eps = np.finfo(np.float64).eps
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
    return edges
```

iii. Equal-percentile bins give roughly balanced class counts for the decoder.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz trial bin centers as the neural data, ensuring temporal alignment.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same 30 Hz grid as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking` in the NWB file, using `width` and `height` fields. It is computed as `2 * max(width, height)`.

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

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Compute pupil diameter as 2 * max(width, height) after blink filtering."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `2 * max(width, height)`, then NaN values (from blinks) are filled by time-based linear interpolation (`fill_nan_by_time`). The filled signal is then linearly resampled to the 30 Hz trial grid, and discretized into 5 equal-percentile bins globally.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. NaN filling by interpolation prevents missing data from propagating into the discretized output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same 5 equal-percentile bin approach as running speed, computed globally across all sessions.

ii. Same `compute_quantile_edges` and `digitize_with_edges` functions.

iii. Same rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 30 Hz trial bin centers as the neural data.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same 30 Hz grid as neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

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

iii. These four outcomes are the SDK's canonical labels for the change detection task. They are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0-3) and repeated as a constant trace across all time bins in the trial. The mapping is: hit=0, miss=1, false_alarm=2, correct_reject=3.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. Constant across all bins because trial outcome is per-trial, not time-varying.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed sessions**: Sessions that raise `KeyError` during Pass 1 are skipped (e.g., missing `EyeTracking` group).
- **Empty trials**: Trials with no valid bin centers (`centers.size == 0`) are skipped.
- **Sessions with few trials**: Sessions with fewer than 2 usable trials raise a `RuntimeError` and are skipped.
- **Missing pupil data**: NaN values in pupil diameter are filled by time-based linear interpolation before resampling.
- **Tied quantile edges**: If percentile edges are not unique (degenerate distributions), an epsilon adjustment is applied.

ii.
```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
...
if centers.size == 0:
    continue
...
diameter = fill_nan_by_time(timestamps, diameter)
...
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. The AI documented: "Sessions missing EyeTracking are excluded up front because pupil output is required." The NaN-fill approach ensures continuous signals for interpolation.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is I/O: reading neural event matrices and behavioral data from NWB files. Additionally, the two-pass approach means each NWB file is read twice (once for global statistics, once for conversion).

ii. N/A

iii. The AI documented timing: ~0.36 s/session for Pass 1 and ~1.86 s/session for Pass 2, with full conversion completing in ~395s for 199 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially, performing interpolation and output construction. However, the AI already vectorized the neural interpolation across all neurons simultaneously using `linear_resample_matrix`. The per-presentation loop for image identity/change within each trial could potentially be vectorized.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The AI noted: "Vectorized event interpolation: Avoided per-neuron interpolation loops."

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design means each NWB file is opened and read twice: once in Pass 1 to collect running/pupil statistics for global binning, and once in Pass 2 for full conversion. The trial table is also read and filtered in both passes.

ii.
```python
# Pass 1:
def collect_global_statistics(sessions):
    for idx, session in enumerate(sessions):
        with h5py.File(session.filepath, "r") as f:
            trials = get_trial_table(f)
            ...
# Pass 2:
def convert_session(session, ...):
    with h5py.File(session.filepath, "r") as f:
        trials = get_trial_table(f)
        ...
```

iii. The AI acknowledged this: "Global binning requires a first pass over sessions, so conversion reads each file twice."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes a `gray` category in image identity that represents inter-stimulus intervals and omitted flashes. This may be considered unnecessary if downstream analyses only care about image identity during stimulus presentations. The code also computes and stores extensive metadata that may not be used by the decoder.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
```

iii. The AI justified including gray: "image_identity will include a gray category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time."
