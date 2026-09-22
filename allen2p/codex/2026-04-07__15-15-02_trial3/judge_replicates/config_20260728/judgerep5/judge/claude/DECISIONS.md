# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from local NWB files using `h5py`, bypassing the AllenSDK high-level loader (which failed due to environment incompatibilities). It discovers available sessions from a CSV experiment table (`ophys_experiment_table.csv`) cross-referenced with locally available `.nwb` files on disk. It filters to active (non-passive) sessions only. A two-pass approach is used: pass 1 collects global statistics (running/pupil bin edges) and filters sessions; pass 2 converts each session's trials.

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

iii. The AI noted that the local AllenSDK/NWB stack could not instantiate these NWB files due to `external_resources`/version mismatch, so it read NWB files directly with `h5py` and mirrored the AllenSDK/whitepaper semantics from the processed NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the experiment metadata table, sorted alphabetically. Each session's mouse_id is looked up in the kept session list.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. `mouse_id` is the SDK's unique identifier for each animal. Subjects are derived from the set of kept sessions after filtering.

## 1-c. How are the data split into sessions?

i. Each individual `ophys_experiment_id` NWB file is treated as one session. This means each imaging plane is a separate session, even when multiple planes were recorded simultaneously in the same `ophys_session_id`. Sessions are not grouped by `ophys_session_id`.

ii.
```python
all_sessions = get_local_session_metadata(data_root)
# Each SessionMeta corresponds to one ophys_experiment_id / one NWB file
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session=session, ...)
```

iii. The AI noted: "Multiple experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK experiment granularity and each file has its own neuron set." This was partly driven by the constraint of working with individual NWB files via h5py.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table, read directly with h5py. Go and catch trials are kept; aborted and auto-rewarded trials are excluded. Trial windows span from `start_time` to `stop_time` (variable length). A 30 Hz grid of bin centers is constructed for each trial.

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

iii. The NWB `trials` table is the authoritative processed trial definition from the Allen processing pipeline. Using `go | catch` and excluding `aborted` and `auto_rewarded` matches the reference definition of contingent trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only go and catch trials, (2) excluding aborted trials, (3) excluding auto-rewarded trials, (4) requiring at least 2 usable trials per session. Sessions without eye tracking are excluded. Passive sessions are excluded. Sessions that fail to load are skipped.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
```

iii. The AI justified excluding passive sessions because "they are not task performance and collapse trial-outcome variability." Sessions without eye tracking were excluded because pupil diameter is a required output variable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` in the NWB files — the extracted calcium event magnitudes, not dF/F traces.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI's CONVERSION_NOTES.md states: "Paper analyses frequently use discrete calcium events rather than raw dF/F" and "Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F." The trajectory confirms: "use the Allen/whitepaper processing choices already embedded in NWB, use event traces as neural activity."

## 2-b. How is the `neural` data processed?

i. Neural event data is linearly resampled from the native ophys timestamps onto a common 30 Hz trial grid using vectorized linear interpolation (searchsorted + broadcasting). Each experiment/session has its own set of neurons from a single imaging plane.

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

iii. The AI chose to resample to 30 Hz to give a common bin size across sessions with different native acquisition rates (31 Hz single-plane, 11 Hz multiplane). This was documented as consistent with the paper's event-triggered analyses that interpolate onto a common time base.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural data filtering is applied beyond what is already in the NWB files. The AI noted that the included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed).

ii. N/A (no filtering code)

iii. The AI documented: "included-session raw NWB files already had all listed cells valid, so event matrices matched converted neuron counts exactly."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to absolute time via the ophys timestamps. For each trial, a 30 Hz grid of bin centers is constructed from `start_time` to `stop_time`, and the neural events are linearly interpolated onto this grid. Alignment is to trial start.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The metadata records `temporal_alignment_event: "trial start"` and `off_start: 0.0`. The 30 Hz grid starts at `start_time + 0.5 * DT` (bin center of first bin).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All data is resampled to a 30 Hz grid (33.33 ms bins), regardless of the native ophys acquisition rate. This is a rebinning from native rates (31 Hz single-plane, ~11 Hz multiplane) to a common 30 Hz.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

iii. The AI justified this as: "Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables (`intervals/*_presentations`) in the NWB files, specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `stimulus_block_name` columns. The tables are filtered to the `change_detection` stimulus block.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials": continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
```

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

iii. Using stimulus presentations rather than the trial-level `initial_image_name`/`change_image_name` allows more precise temporal placement of image flashes and explicit representation of gray inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is a time-varying categorical variable. The default value is "gray" (for inter-stimulus intervals and omitted flashes). During each stimulus flash (from `start_time` to `stop_time` of each presentation row), the image code is set to the corresponding image name's integer index. A global sorted mapping of all unique image names (including "gray") is used for consistent encoding across sessions.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. Including "gray" as a category ensures that inter-stimulus intervals and omitted flashes are explicitly represented, since the decoder instruction asks for "image identity of the image presented during the non-grey screen."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz bin centers as the neural data. For each bin center, if it falls within a stimulus presentation's `[start_time, stop_time)`, the image code is set accordingly; otherwise it remains "gray".

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. Using the same `centers` array for both neural and output data guarantees temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentation tables, filtered to the `change_detection` block.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The `is_change` flag in the stimulus presentations table identifies which stimulus flash is the change stimulus.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary time-varying variable. It is 0 by default. During the temporal window of any stimulus presentation where `is_change` is True, image change is set to 1. This covers the 250ms flash duration of the change stimulus.

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

iii. The AI uses the stimulus presentation's actual timing to mark the change event, rather than constructing a fixed-duration window from `change_time`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change), so no thresholding is needed.

ii.
```python
["no_change", "change"]
```

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed on the same 30 Hz bin centers as the neural data.

ii. See 4-b code above — uses the same `centers` array.

iii. Same frame-level alignment as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB files, which contains timestamps and speed data.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The `running/speed` group in the NWB processing module contains the Allen-processed filtered running speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly resampled from its native timestamps to the 30 Hz trial grid using `np.interp`, then discretized into 5 equal-percentile bins. Bin edges are computed globally across all sessions in pass 1.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

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

iii. Percentile-based binning ensures roughly equal class counts. Global computation across all sessions maintains consistent categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins (labeled q1-q5) using globally computed bin edges. Values are clipped to the edge range before digitizing.

ii.
```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. Clipping before digitizing ensures no out-of-range values.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial bin centers used for neural data.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Same temporal grid guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height` in the NWB files. The diameter is computed as `2 * max(width, height)`.

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

iii. The AI chose to derive pupil diameter "in the same way the reference stack defines the pupil geometry rather than inventing a new measure." NaN values from blinks are interpolated over time.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `2 * max(width, height)` from the ellipse fits, NaN values are filled by linear time interpolation, then the signal is linearly resampled to the 30 Hz trial grid and discretized into 5 equal-percentile bins globally.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. NaN filling by interpolation prevents missing data from propagating into the discretized output. Global percentile binning maintains consistent categories across sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — 5 equal-percentile bins computed globally across all sessions from finite values.

ii.
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. Same as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto the same 30 Hz trial bin centers.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same temporal grid guarantees alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]): return 0
    if bool(trial_row["miss"]): return 1
    if bool(trial_row["false_alarm"]): return 2
    if bool(trial_row["correct_reject"]): return 3
    raise ValueError("Trial has no valid outcome label")
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task. They are mutually exclusive for contingent trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3: hit, miss, false_alarm, correct_reject) and replicated as a constant trace across all time bins within a trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The instruction specifies trial outcome as "static per-trial," so the code fills the entire trial with the same value.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` in the NWB acquisition group are excluded entirely.
- **NaN pupil values**: Blink-related NaN values in pupil data are filled by time interpolation (`fill_nan_by_time`).
- **Empty trials**: Trials where `build_trial_bins` returns an empty array (invalid start/stop times) are skipped.
- **Failed sessions**: Sessions that raise exceptions during pass 1 are skipped with a warning.
- **Sessions with <2 trials**: Excluded from the output.
- **Duplicate quantile edges**: When quantile edges are not unique, small epsilon offsets are added to ensure monotonicity.

ii.
```python
# Missing eye tracking
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")

# NaN interpolation
def fill_nan_by_time(time_axis, values):
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])

# Empty trial bins
if centers.size == 0:
    continue

# Duplicate quantile edges
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. The AI documented: "Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the I/O-heavy NWB file reads. The two-pass design means each file is read twice — once for global statistics collection and once for conversion.

ii. N/A

iii. The AI noted: "Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk. Global binning requires a first pass over sessions, so conversion reads each file twice."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-presentation loop within each trial (iterating over stimulus presentation rows to set image identity and change indicators) could potentially be vectorized. The per-trial loop within `convert_session` iterates sequentially.

ii.
```python
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
```

iii. The AI did vectorize the neural interpolation (using searchsorted + broadcasting), but the stimulus presentation loop remains iterative.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is read twice — once in pass 1 (global statistics collection) and once in pass 2 (session conversion). In pass 1, the running/pupil data is resampled per trial to compute global bin edges; in pass 2, the same resampling is done again during actual conversion.

ii.
```python
# Pass 1
for trial in trials.itertuples(index=False):
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)

# Pass 2
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The two-pass design was a deliberate trade-off: global bin edges must be known before discretization, requiring a first pass over the data.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `2 * max(width, height)` for pupil diameter, which is a more complex computation than needed if only `pupil_width` is a sufficient measure. The NaN interpolation for pupil data is computed session-wide even though only trial segments are used.

ii. See `get_pupil_data` function above.

iii. The full-session NaN interpolation is done before trial segmentation for simplicity, even though only within-trial values are ultimately used.
