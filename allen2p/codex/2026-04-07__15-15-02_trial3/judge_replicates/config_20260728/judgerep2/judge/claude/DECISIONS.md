# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly with `h5py` from the local data directory (`data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`). It discovers available experiments from `ophys_experiment_table.csv`, filters to locally available NWB files, excludes passive sessions, and reads each file individually. This differs from the reference which uses the AllenSDK `VisualBehaviorOphysProjectCache.from_s3_cache()` and `get_behavior_ophys_experiment()`.

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

iii. The AI documented in CONVERSION_NOTES.md Step 4 that the local AllenSDK/NWB stack could not instantiate the NWB files due to `external_resources`/version mismatch, so it chose to read them directly with `h5py` and mirror the AllenSDK semantics from the processed NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table. Sorted unique mouse IDs from kept sessions form the subjects list.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard identifier for each animal. This matches the reference approach.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` (one imaging plane) as a separate session. This differs from the reference, which groups experiments by `ophys_session_id` to combine multiple imaging planes from the same behavioral session into one session.

ii.
```python
# Each experiment file = one session
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(
        session=session, ...)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session." The AI noted the AllenSDK treats each experiment as a self-contained unit.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table, filtering to go and catch trials while excluding aborted and auto-rewarded. Each trial spans from `start_time` to `stop_time`, giving variable-length trials. The AI constructs a 30 Hz time grid within each trial window using `build_trial_bins`.

ii.
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

iii. The AI uses the NWB processed trials table directly, consistent with the Allen pipeline. The 30 Hz grid provides a common temporal resolution across all sessions regardless of native acquisition rate.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be `go` or `catch` and not `aborted` or `auto_rewarded`. Sessions must have at least 2 valid trials. Sessions missing eye tracking are excluded. Trials with empty time bins are skipped.

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

iii. The filtering matches the instruction requirements (include go and catch, exclude aborted and auto-rewarded). The reference additionally filters on `change_time.notna()`, which the AI does not explicitly do but achieves a similar effect since go/catch trials should have valid change times.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `processing/ophys/event_detection/data` (calcium event magnitudes) from the NWB files. This differs from the reference which uses `dff_traces` (dF/F calcium fluorescence traces).

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI documented in CONVERSION_NOTES.md Steps 3-4 that the paper's neural analyses "frequently use discrete calcium events rather than raw dF/F" and chose events to "best match the paper's neural analyses." The whitepaper event detection pipeline is embedded in the NWB files.

## 2-b. How is the `neural` data processed?

i. Event magnitudes are linearly resampled from their native ophys timestamps onto a common 30 Hz trial grid using vectorized interpolation. This differs from the reference which applies no processing beyond stacking dF/F traces from multiple planes.

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

iii. The AI resamples to 30 Hz to provide a common bin size across all sessions (some are 31 Hz single-plane, some 11 Hz multiplane). The reference avoids this by keeping native frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are taken from the NWB `cell_specimen_table`, which in the available data already contained only valid ROIs. No additional filtering is applied beyond what the NWB pipeline provides.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The AI noted in CONVERSION_NOTES.md Step 10 that "included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed)." This is consistent with the reference which also relies on SDK-level filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. A 30 Hz grid of bin centers is constructed from `start_time` to `stop_time`, and event magnitudes are linearly interpolated onto this grid.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The reference aligns using `np.searchsorted` on ophys timestamps to find frame indices within the trial window. The AI instead constructs a uniform 30 Hz grid and interpolates. Both align to trial start, but the time grids differ.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a fixed 30 Hz grid (33.33 ms bins). The reference keeps the native ophys frame rate (approximately 11 Hz for multiplane or 31 Hz for single-plane sessions) with no rebinning.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
```

iii. The AI documented in CONVERSION_NOTES.md Step 4: "Resample all streams to a common 30 Hz grid (33.333... ms bins). This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size." The reference avoids this issue because all sessions in the `VisualBehavior` project are single-plane at ~31 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from the stimulus presentation interval tables in the NWB (`intervals/*_presentations`), using `image_name`, `start_time`, `stop_time`, `omitted`, `is_change`, and `trials_id` columns. The reference derives it from the trials table `initial_image_name` and `change_image_name` with `change_time`.

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

iii. The AI chose stimulus presentations to build a more granular time-varying signal, capturing exact flash timing including gray inter-stimulus intervals. The reference uses a simpler approach with just initial/change image names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI maps stimulus presentations onto the 30 Hz trial grid: during image flash intervals, the bin gets the image code; during gray/omitted periods, it gets a "gray" code. This includes a "gray" category as an explicit image identity value. The reference only maps initial_image_name before change_time and change_image_name after, with no gray category.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Encode gray/omission periods explicitly: image_identity will include a gray category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid as the neural data, so alignment is inherent.

ii.
```python
# Same `centers` array used for both neural and image identity
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
```

iii. Alignment is guaranteed because all signals share the same 30 Hz grid per trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derives image change from the `is_change` column in the stimulus presentation table. The reference derives it from `change_time` and `go` in the trials table.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI uses the stimulus presentation's `is_change` flag which marks the actual change flash interval. The reference uses a 750ms window starting at `change_time` for go trials only.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is set to 1 during the time bins that fall within a stimulus presentation interval where `is_change == True`. This marks only the change flash itself (250 ms). The reference marks a 750 ms window (flash + ISI).

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. The AI marks the change during the actual stimulus presentation window based on the `is_change` flag. The reference marks a 750 ms window (one flash duration + one ISI) and restricts to go trials only. The AI's approach includes change markers for both go and catch trials where `is_change` is True.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1), no thresholding needed.

ii. See 4-b.

iii. The variable is inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz grid as neural data.

ii. See 4-b.

iii. Same grid alignment as all other signals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file (timestamps and data arrays). The reference uses `dataset.running_speed` from the AllenSDK, which wraps the same underlying data.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. Both access the same processed running speed data; the AI reads it directly from NWB while the reference goes through the SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `np.interp`, then discretized into 5 equal-percentile bins computed globally across all sessions. The reference interpolates to the ophys timebase using `scipy.interpolate.interp1d`.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Both approaches interpolate running speed to the neural time grid and discretize into 5 percentile bins globally. The AI uses `np.interp` which clamps at boundaries instead of producing NaN.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins are computed globally from all finite running speed values across all included sessions and trials, using `np.quantile`. Values are then digitized using `np.searchsorted` with clipping.

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

iii. Percentile-based binning ensures roughly equal class counts. The reference uses `np.percentile` and `np.digitize`, which is functionally equivalent. The AI clips values to edges; the reference maps NaN to bin 0.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid as neural data.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives pupil diameter from `acquisition/EyeTracking/pupil_tracking` using `width` and `height`, computing diameter as `2 * max(width, height)`. The reference uses `pupil_width` from `dataset.eye_tracking`.

ii.
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
```

iii. The AI computes diameter from ellipse fit parameters (2*max of semi-axes). The reference uses `pupil_width` directly. These measure different aspects of the pupil ellipse.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI fills NaN values (blinks) by time-based linear interpolation using `fill_nan_by_time`, then linearly resamples to the 30 Hz grid and discretizes into 5 percentile bins. The reference removes blink frames first, then interpolates with `scipy.interpolate.interp1d` (bounds_error=False, fill_value=NaN).

ii.
```python
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The AI interpolates over NaN/blink periods before resampling. The reference removes blink-flagged frames and then interpolates. Both approaches handle blinks by interpolation, but differ in the specific implementation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile bins computed globally, digitized with clipping.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Same approach as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 30 Hz grid as all other signals.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same grid-based alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table. This matches the reference.

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

iii. These four outcome types are the standard SDK trial outcomes for the change detection task, matching the reference approach.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and replicated as a constant trace across all time bins within a trial. This matches the reference.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. Same categorical encoding as reference. The order matches `OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: Sessions lacking `EyeTracking` in the NWB are excluded entirely (`KeyError` caught in pass 1).
- **Pupil NaNs/blinks**: NaN values in pupil data are filled by time-interpolation before resampling via `fill_nan_by_time`.
- **Empty trial bins**: Trials where `build_trial_bins` returns empty array are skipped.
- **Few trials**: Sessions with <2 usable trials raise a `RuntimeError` (caught by caller in pass 1; would crash in pass 2 but such sessions were already excluded).
- **Zero neural events**: Retained as valid trials; documented that 4.84% of trials had all-zero events in source data.

ii.
```python
# Missing eye tracking
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip ...")

# Pupil NaN fill
def fill_nan_by_time(time_axis, values):
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])

# Empty trials
if centers.size == 0:
    continue
```

iii. The AI documented its handling extensively in CONVERSION_NOTES.md Steps 9-10, including raw-data verification that zero-event trials are genuine source data characteristics.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are reading NWB files from disk (I/O bound) and the two-pass design requiring each file to be read twice: once for global statistics and once for conversion.

ii. N/A (architectural observation)

iii. The AI documented in CONVERSION_NOTES.md Step 7: "Pass 1 bin-stat collection: ~0.36 s/session" and "Pass 2 conversion: ~1.86 s/session", total ~7.5 min for 202 sessions. The two-pass design is the main overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial to build time grids, resample, and assign stimulus labels. The stimulus presentation matching within each trial could potentially be vectorized.

ii.
```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(...)
    neural_trial = linear_resample_matrix(ophys_time, events, centers)
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The AI noted vectorized event interpolation as a speedup already implemented. The remaining per-trial loop is straightforward and I/O is the dominant cost.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in `collect_global_statistics` (pass 1) for running/pupil bin edge computation, and once in `convert_session` (pass 2) for full trial conversion. In pass 1, it also resamples running and pupil for bin edge computation, then repeats this in pass 2.

ii.
```python
# Pass 1
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)

# Pass 2 (same resampling repeated)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The two-pass design is a deliberate architectural choice to compute global bin edges before the final conversion, at the cost of reading and resampling data twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes running and pupil resampled values in pass 1 just to collect them for bin edge computation, then discards and recomputes them in pass 2. Additionally, the entire `get_task_presentations` table is loaded in both passes (pass 1 for session filtering/image name collection, pass 2 for stimulus alignment).

ii. See 9-c.

iii. This redundancy is a consequence of the two-pass architecture. A single-pass approach would avoid the duplication but would require a different strategy for global bin edge computation.
