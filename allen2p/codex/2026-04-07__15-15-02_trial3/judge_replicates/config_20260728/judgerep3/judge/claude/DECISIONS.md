# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads local NWB files directly using `h5py` rather than the AllenSDK. It reads session metadata from `ophys_experiment_table.csv`, finds locally available NWB files on disk, filters to active (non-passive) sessions, and loads each experiment file individually. A two-pass approach is used: pass 1 collects global running/pupil statistics and filters sessions; pass 2 converts each session's trials.

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

iii. The AI could not use the AllenSDK high-level loader due to local NWB/SDK version incompatibility, so it reads the processed NWB contents directly with `h5py`. This mirrors the same data that the AllenSDK would expose, just through a lower-level interface. The AI documented this in CONVERSION_NOTES.md Step 4 as an environment workaround.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table. Each kept session's mouse_id is mapped to a subject index.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. `mouse_id` is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (i.e., each NWB file corresponding to one imaging plane) is treated as a separate session. This differs from grouping experiments by `ophys_session_id`.

ii.
```python
sessions = []
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ...
            filepath=available_files[int(row.ophys_experiment_id)],
        )
    )
```

iii. The AI chose this because each NWB file is a single imaging plane with its own neuron set, matching the `BehaviorOphysExperiment` granularity of the AllenSDK. The AI noted this in CONVERSION_NOTES.md Step 5 Key Decision #2.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. The AI filters to keep only GO and CATCH trials, excluding aborted and auto-rewarded trials. For each kept trial, time bins are constructed from `start_time` to `stop_time` on a 30 Hz grid.

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

iii. Using the NWB trials table mirrors the Allen SDK processing. Filtering to GO and CATCH while excluding aborted and auto-rewarded matches the task instructions. The 30 Hz grid creates uniform time bins across trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only GO and CATCH trials, (2) excluding aborted trials, (3) excluding auto-rewarded trials, (4) requiring non-empty time bins, (5) sessions with fewer than 2 usable trials are rejected. Sessions missing eye tracking data are also excluded.

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

In `collect_global_statistics`:
```python
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip ...")
    continue
```

Sessions without eye tracking:
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

iii. These filters match the task instructions (exclude aborted and auto-rewarded, include GO and CATCH). The minimum 2-trial threshold ensures the decoder has enough data per session. Excluding sessions without eye tracking is necessary since pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` in the NWB files -- the detected calcium event magnitudes, NOT dF/F traces.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The AI chose event traces because the paper's neural analyses use calcium events rather than raw dF/F (documented in CONVERSION_NOTES.md Step 3 and Step 4). The whitepaper describes the event detection pipeline (FastLZero algorithm).

## 2-b. How is the `neural` data processed?

i. Neural event data is linearly resampled from native ophys timestamps onto a common 30 Hz grid defined by trial bin centers. This is done via a vectorized linear interpolation across all neurons simultaneously.

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

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Resampling to 30 Hz creates a common time bin size across sessions with different native acquisition rates (31 Hz single-plane, 11 Hz multiplane). The AI documented this decision in CONVERSION_NOTES.md Step 4 and Step 5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied beyond what the NWB files already contain. The AI noted that all cells in the local NWB files had `valid_roi == True`.

ii. N/A (no explicit filtering code)

```python
brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
# Simply counts cells, no filtering
```

iii. The AI verified that the NWB files already contain only valid ROIs (29,168 valid out of 29,168 listed), so no additional filtering was needed. The Allen SDK pipeline already applies ROI curation upstream.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, a 30 Hz grid of bin centers is constructed from `start_time` to `stop_time`, and neural events are interpolated onto this grid.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The instructions say to "temporally align based on ophys timestamp." The AI uses ophys timestamps as the source time axis and constructs a trial grid starting at each trial's `start_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a fixed 30 Hz grid (33.33 ms bins), regardless of the native ophys acquisition rate. This is a significant rebinning step.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

iii. The AI justified this by noting that native acquisition rates vary (31 Hz single-plane, 11 Hz multiplane) and that behavior/eye tracking is at 30 Hz. A common 30 Hz grid satisfies the decoder requirement that all trials share a common bin size. This was documented in CONVERSION_NOTES.md Step 4 and Step 5.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB (`intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `trials_id` columns. A `gray` category is used for inter-stimulus intervals and omitted flashes.

ii.
```python
presentations = get_task_presentations(f)
...
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. Using stimulus presentations provides frame-accurate image identity that reflects the actual 250ms on / 500ms off stimulus cadence, including omitted flashes and gray periods. The AI included a `gray` category to represent inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all sessions, with `gray` placed at index 0. The time-varying trace is built by marking each time bin with the image shown during its corresponding stimulus presentation, defaulting to `gray`.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
```

iii. A global mapping ensures consistent codes across sessions. The `gray` default covers inter-stimulus intervals naturally.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid (bin centers) as the neural data, by checking which stimulus presentation interval each bin center falls within.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. Using the same `centers` array guarantees temporal alignment between neural and image identity data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation table's `is_change` column, which marks the change-image flash.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The `is_change` flag in the stimulus presentations is set by the Allen SDK processing pipeline for the flash where the image identity changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created: 1 during the time window of the change flash (as defined by the stimulus presentation's `start_time` to `stop_time`), 0 elsewhere. This applies to both go and catch trials if `is_change` is True.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The `is_change` flag marks the actual change event in the stimulus presentation timeline.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii. Output values: `["no_change", "change"]`

iii. The binary nature of image change makes thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- computed on the same 30 Hz bin centers as neural data, using the stimulus presentation time windows.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. Same alignment mechanism as all other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB files, which provides timestamps and speed values.

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

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial bin centers using `np.interp`, then discretized into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
def linear_resample_vector(src_time, src_value, dst_time):
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
```

```python
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. Linear interpolation resamples to the common timebase. Global percentile binning ensures balanced class counts across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile (quantile) bins using edges computed from all finite running speed values across all included sessions and trials.

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

iii. Percentile-based binning produces approximately equal class counts, which is important for balanced decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial bin centers as neural data.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Using the same `centers` array as neural data guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking` in the NWB files, specifically `width` and `height` fields, plus timestamps from `eye_tracking`.

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

iii. The AI computes pupil diameter as `2 * max(width, height)` from the pupil ellipse fit, then fills NaN values by time interpolation before resampling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `2 * max(width, height)`, NaN values are filled by time-based interpolation, then the signal is linearly resampled to the 30 Hz trial grid and discretized into 5 equal-percentile bins computed globally.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The NaN fill-by-interpolation handles blink artifacts. Global percentile binning ensures balanced classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile bins computed globally across all finite values.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Same rationale as running speed -- balanced class counts for decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed -- interpolated onto the same 30 Hz trial bin centers.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same alignment mechanism as all time-varying outputs.

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

iii. These four columns are the standard trial outcome labels from the Allen SDK for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0-3) and broadcast as a constant trace across all time bins within the trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. Trial outcome is a per-trial static variable, so it is constant across all time bins within a trial. The ordering matches `OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` in the NWB acquisition group are excluded entirely (KeyError caught in pass 1).
- **NaN pupil values**: Filled by time-based interpolation before resampling (`fill_nan_by_time`).
- **Empty trials**: Trials where `build_trial_bins` produces no bins are skipped.
- **Sessions with too few trials**: Sessions with fewer than 2 usable trials are skipped.
- **Duplicate quantile edges**: When percentile edges are not unique, small epsilon offsets are added.

ii.
```python
def fill_nan_by_time(time_axis, values):
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values
```

```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

iii. Missing eye tracking makes pupil diameter unavailable, requiring session exclusion. NaN interpolation for pupil prevents artifacts from propagating. The epsilon offset for duplicate edges prevents degenerate bin boundaries.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading NWB files from disk, which happens twice (pass 1 for statistics collection and pass 2 for conversion). Each file contains large neural event matrices and behavioral data.

ii. N/A (I/O-bound operations)

iii. The two-pass design doubles the file I/O. However, the AI noted this was necessary to compute global binning edges before final conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over trials sequentially, with per-presentation inner loops for building image identity traces. The inner loop over stimulus presentations could potentially be vectorized.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        ...
```

iii. The nested loop is not a major bottleneck compared to file I/O, but vectorizing the stimulus presentation matching could provide modest speedup.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is read twice: once in pass 1 (for global statistics) and once in pass 2 (for actual conversion). This includes reading the trials table, running data, pupil data, and neural data (in pass 2).

ii.
```python
# Pass 1:
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)

# Pass 2:
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. The AI acknowledged this inefficiency in CONVERSION_NOTES.md but noted it was a design tradeoff for memory efficiency (not keeping all data in memory between passes).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `fill_nan_by_time` for pupil data (interpolating NaN values), which fills in blink-period values. These interpolated values still get discretized and included in the output. The code also reads and processes stimulus presentations in pass 1 to collect image names, even though this information isn't used for the global statistics computation.

ii.
```python
# In collect_global_statistics, image names are collected but only used later:
image_names.update(x for x in presentations["image_name"].unique() if x != "omitted")
```

iii. The image name collection in pass 1 is a minor overhead. The main unnecessary work is the double file reading.
