# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly using `h5py` rather than the AllenSDK. It discovers available experiments from `ophys_experiment_table.csv`, filters to locally available NWB files, excludes passive sessions, and reads each file individually. A two-pass approach is used: pass 1 collects global statistics (running/pupil bin edges), pass 2 converts each session's trials.

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

iii. The AI could not use the AllenSDK's high-level loader (`VisualBehaviorOphysProjectCache`) due to an environment mismatch (NWB/AllenSDK version incompatibility). It opted to read the processed NWB files directly with `h5py`, mirroring the AllenSDK semantics from the processed NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table (`ophys_experiment_table.csv`). The AI collects all unique mouse IDs from the kept sessions after filtering.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` (i.e., each NWB file, corresponding to one imaging plane) as a separate session. This differs from the reference approach of grouping multiple imaging planes (experiments) by `ophys_session_id` into a single session.

ii.
```python
# Each SessionMeta corresponds to one ophys_experiment_id / one NWB file
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ...
        )
    )
```

iii. The AI justified this by saying each experiment file is one imaging plane with its own neuron set, and this matches the AllenSDK's `BehaviorOphysExperiment` granularity.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. For each non-aborted, non-auto-rewarded trial that is either go or catch, the trial window from `start_time` to `stop_time` is used. Trials are resampled onto a fixed 30 Hz grid using `build_trial_bins`.

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

iii. The AI uses the NWB trials table directly, filtering to GO/CATCH and excluding aborted/auto-rewarded trials. The trial grid is constructed at 30 Hz (DT = 1/30 s) with bin centers from `start_time` to `stop_time`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only `go` or `catch` trials, (2) excluding `aborted` trials, (3) excluding `auto_rewarded` trials, (4) requiring at least 2 usable trials per session. Sessions missing eye tracking are excluded entirely.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. The filtering matches the instructions (include GO and CATCH, exclude aborted and auto-rewarded). The AI additionally excludes sessions without eye tracking data since pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection` data in the NWB file (`processing/ophys/event_detection`), which contains calcium event magnitudes, not dF/F traces.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI chose event detection traces because the paper's neural analyses use extracted calcium events rather than raw dF/F (citing `methods.txt:179`, `methods.txt:208`). The AI noted that the whitepaper event detection uses FastLZero on fluorescence-derived traces.

## 2-b. How is the `neural` data processed?

i. Event detection data is linearly interpolated (resampled) from native ophys timestamps to a common 30 Hz grid defined by trial bin centers. A custom vectorized interpolation is used (searchsorted + linear weighting).

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

iii. The AI resamples to 30 Hz to ensure all sessions share a common time bin size, since native acquisition rates vary (31 Hz single-plane, 11 Hz multiplane).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. The AI uses all cells present in the NWB event detection data. It notes that the NWB files already had all listed cells as valid (`valid_roi`).

ii. N/A (no filtering code)

iii. The AI checked that raw NWB files already contained only valid ROIs (`29,168 total valid of 29,168 total listed`), so no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `trial_start` (the `start_time` from the trials table). A 30 Hz grid of bin centers is constructed from `start_time` to `stop_time`, and neural event data is linearly interpolated onto these centers.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The instructions say "temporally align based on ophys timestamp." The AI aligns by interpolating onto a uniform 30 Hz grid within each trial's time window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a fixed 30 Hz grid (33.33 ms bins). This involves temporal rebinning from native ophys rates (31 Hz single-plane or 11 Hz multiplane) to a common 30 Hz rate.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

iii. The AI chose 30 Hz because it is close to both the behavior/eye-tracking rate and the single-plane ophys rate, ensuring a consistent bin size across all sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations intervals table in the NWB file (under `intervals/*_presentations`), filtering to change detection blocks. The `image_name` field from each presentation is used, with a `gray` category for inter-stimulus intervals and omitted flashes.

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

iii. The AI used stimulus presentations to capture the temporal structure of image flashes (250 ms on, 500 ms gray ISI) and to include a `gray` category for periods when no image is shown.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all sessions, with `gray` placed at index 0. The integer code varies within a trial based on which stimulus is being presented at each 30 Hz time bin.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The AI includes both gray-screen periods and actual images in the identity variable, so the decoder sees the full temporal pattern of image presentations.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid as the neural data by checking which stimulus presentations overlap each time bin center.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. Both neural and image identity share the same `centers` array, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI uses the stimulus presentations' `is_change` flag to mark time bins where a change image is being displayed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created for each trial. Time bins overlapping a stimulus presentation with `is_change == True` are set to 1; all others are 0.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. This marks the change image flash itself (250 ms duration) as the change event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1), so no thresholding is needed. It is 1 during the change flash presentation and 0 otherwise.

ii. See 4-b above.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz grid alignment as image identity and neural data.

ii. See 4-b above.

iii. Same alignment mechanism as all other time-varying outputs.

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

iii. This is the SDK's standard running speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated (`np.interp`) from its native timestamps to the 30 Hz trial grid, then discretized into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
def linear_resample_vector(src_time, src_value, dst_time):
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
```

iii. Percentile-based binning ensures roughly equal class counts. Bin edges are computed globally from all sessions in pass 1.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using quantile edges computed globally. `np.searchsorted` on inner edges assigns bin indices.

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

iii. Values are clipped to the edge range before digitization.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid (`centers`) as the neural data.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same alignment mechanism as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking` in the NWB file, using `width` and `height` fields. Diameter is computed as `2 * max(width, height)`.

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

iii. The AI computed pupil diameter from the ellipse fit parameters (width and height), taking `2 * max(width, height)` as the diameter. Blink-related NaNs are filled by time interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (`2 * max(width, height)`) has NaN values filled by time-based linear interpolation (`fill_nan_by_time`), is then interpolated to the 30 Hz trial grid, and finally discretized into 5 percentile bins computed globally.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Same discretization approach as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same quantile-based 5-bin discretization as running speed, computed globally across all sessions.

ii. See 5-c for the discretization functions.

iii. Same approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as neural data.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same alignment mechanism as neural and running data.

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

iii. These four columns are the SDK's canonical trial outcome labels. They are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and repeated across all time bins within a trial as a constant trace.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The outcome is static per trial but represented as a time-varying trace to maintain a consistent `(n_output, T)` output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- Sessions missing eye tracking (`EyeTracking` key not in NWB) are excluded entirely.
- Pupil NaN values (from blinks) are filled by time-based interpolation before resampling.
- Sessions with fewer than 2 valid trials raise a RuntimeError and are skipped during pass 1.
- Trials with empty time bins (`centers.size == 0`) are skipped.
- Sessions that fail during pass 1 (KeyError) are skipped with a warning.

ii.
```python
# Missing eye tracking
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")

# NaN filling for pupil
diameter = fill_nan_by_time(timestamps, diameter)

# Empty trials
if centers.size == 0:
    continue

# Session-level errors
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

iii. The AI fills pupil NaN values by interpolation rather than mapping to a default bin. Sessions without eye tracking are excluded because pupil diameter is a required output.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is I/O: reading NWB files with h5py. The full conversion reads each file twice (pass 1 for global statistics, pass 2 for conversion). Total conversion time was ~395 seconds for 199 sessions.

ii. N/A

iii. Per-session time was ~2 seconds, dominated by NWB file reads.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial to build stimulus traces from presentation rows. The inner loop over stimulus presentations per trial could potentially be vectorized. The neural resampling was already vectorized using searchsorted + broadcasting.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The per-presentation inner loop is necessary due to the variable structure of stimulus presentations, though a fully vectorized approach using interval assignment could be faster.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in pass 1 (to collect global running/pupil statistics) and once in pass 2 (to convert trials). This means neural data, running data, pupil data, and trial tables are all loaded twice per session.

ii.
```python
# Pass 1
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
# Pass 2
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session, ...)
```

iii. The two-pass design is chosen to compute global percentile bin edges before the conversion pass, but it doubles I/O time.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes pupil diameter as `2 * max(width, height)` using both width and height from the pupil tracking ellipse fit, then fills NaN values by interpolation. This is more complex than simply using `pupil_width` as the reference does. The NaN filling step is also potentially unnecessary since the reference solution maps NaN to bin 0 during discretization instead.

ii. See 6-a code snippet.

iii. The additional computation of diameter from ellipse parameters and NaN interpolation adds complexity without clear benefit over using `pupil_width` directly.
