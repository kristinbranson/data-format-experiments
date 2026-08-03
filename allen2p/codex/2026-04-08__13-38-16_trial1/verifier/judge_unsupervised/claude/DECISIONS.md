# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB/HDF5 files stored under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` using `h5py`. It globs for all files matching `behavior_ophys_experiment_*.nwb`, then performs a two-pass workflow: a preview pass to identify eligible sessions and compute global statistics, followed by a conversion pass that opens each NWB file again to extract neural, behavioral, and stimulus data. The AI chose to bypass `pynwb` due to environment compatibility issues and read HDF5 fields directly.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
```
Preview pass via `session_preview()`, then conversion pass via `convert_session()` which opens each NWB file with `h5py.File(preview.path, "r")`.

iii. Justified in CONVERSION_NOTES.md Step 6: "Uses direct HDF5 reads from local NWB files rather than pynwb because the installed NWB stack is incompatible with these files in this environment." The two-pass design is explained as intentional to avoid storing large neural matrices before global percentile/bin definitions are known.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from each NWB file's `/general/subject/subject_id` field. A running dictionary `subject_to_idx` maps string subject IDs to indices. Each session's subject ID is looked up and the index is stored in `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
# ...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The AI identified subjects from the NWB metadata, consistent with the AllenSDK data model. 38 unique subjects were found in the local data subset.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file corresponds to one session (one imaging plane in one ophys session). The AI treats each experiment file as an independent session. Sessions are processed sequentially, with each producing a list of trial arrays.

ii.
```python
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
# Each file becomes one session in the output:
neural_sessions.append(neural_trials)
input_sessions.append(input_trials)
output_sessions.append(output_trials)
```

iii. From CONVERSION_NOTES.md Step 2: "One experiment corresponds to one imaging plane in one session." 281 eligible sessions were included (3 excluded for missing eye tracking).

## 1-d. How are the data split into trials?

i. Trials are defined from the `/intervals/trials` group in each NWB file. The `build_trial_specs()` function iterates over all trials, filters out aborted and auto-rewarded trials, and creates `TrialSpec` objects with start/stop times and outcome information. Each valid trial becomes one entry in the session's trial list.

ii.
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    raw_count = len(trials["id"])
    aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
    auto_rewarded = np.nan_to_num(trials["auto_rewarded"], nan=0.0).astype(bool)
    # ...
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        # ...
        specs.append(TrialSpec(...))
```

iii. Consistent with the instructions ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials") and the SDK trial semantics documented in Step 1 notes.

## 1-e. How are trials filtered based on quality controls?

i. Two levels of filtering: (1) Session-level: sessions missing eye tracking are excluded (3 sessions), and sessions with fewer than 2 valid trials or insufficient pupil samples are excluded. (2) Trial-level: aborted and auto-rewarded trials are excluded. Non-aborted, non-auto-rewarded trials must have exactly one valid outcome flag (hit, miss, false_alarm, correct_reject).

ii.
```python
# Session filtering:
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"

# Trial filtering in build_trial_specs():
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
```

iii. Session exclusion for missing eye tracking is justified because pupil diameter is a required decoder output. Trial filtering matches the instructions and SDK semantics.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the event-detection output stored at `/processing/ophys/event_detection/data` in the NWB files, filtered by the `valid_roi` flag from `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`.

ii.
```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
    )
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
    return event_data, event_rois[valid_mask]
```

iii. From CONVERSION_NOTES.md Step 4/5: "Prefer event-detection outputs for neural activity because the analysis paper explicitly uses them" and "the paper explicitly states that analyses were performed on discrete calcium events."

## 2-b. How is the `neural` data processed?

i. The raw event-detection data (at native ophys frame rate) is rebinned into 100 ms time bins by summing event magnitudes within each bin. Bins are defined using `np.arange(start_time, stop_time, 0.1)` and ophys timestamps are mapped to bins via `np.searchsorted`.

ii.
```python
BIN_SIZE_SEC = 0.1
# ...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI chose 100 ms bins as a compromise between the 31 Hz single-plane and 11 Hz multi-plane ophys rates, while still resolving 250 ms stimulus flashes. Metadata describes this as "sum event magnitudes within each 100 ms trial bin."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs marked as `valid_roi == True` are retained. No additional neural quality filtering (e.g., signal-to-noise thresholds, minimum firing rate) is applied beyond the SDK's valid_roi flag.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. From CONVERSION_NOTES.md Step 1: "invalid ROIs are removed when exclude_invalid_rois=True (default in BehaviorOphysExperiment.from_nwb/from_lims)." This mirrors the SDK's default behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. The instructions say to "temporally align based on ophys timestamp." The AI uses the trial's start_time and stop_time to define the trial window, then creates bins from start_time to stop_time. Ophys timestamps are used via `np.searchsorted` to find which ophys frames fall within each 100 ms bin.

ii.
```python
ophys_timestamps = np.asarray(
    f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
)
# ...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The AI notes in metadata: `temporal_alignment_event: "trial start"` with `off_start: 0.0` and `off_end: None`. Alignment is to the trial window defined by the trials table.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (0.1 s). Yes, temporal rebinning is applied: the native ophys frame rate (31 Hz single-plane or 11 Hz multi-plane) is rebinned to uniform 100 ms bins by summing events within each bin.

ii.
```python
BIN_SIZE_SEC = 0.1
# In metadata:
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. From CONVERSION_NOTES.md Step 5: "Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation tables found in `/intervals/*_presentations` groups (excluding `/intervals/trials`). Specifically, the `image_name`, `start_time`, `stop_time`, and `omitted` fields are used.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    # reads from /intervals/* groups (excluding trials)
    keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]
```

iii. The AI identified stimulus presentations as the source for image identity, consistent with the AllenSDK `Presentations` object.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, the AI finds presentation rows overlapping the trial window. For each time bin center, if it falls within a non-omitted stimulus presentation window, the bin is assigned the corresponding image's integer index. Otherwise (during ISI, omissions, or no-image periods), it is assigned to "gray" (index 0). The global set of image categories includes "gray" plus all unique non-omitted image names found across eligible sessions.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        # ...
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. From CONVERSION_NOTES.md Step 5: "Project stimulus presentations onto trial bins; assign image category during image flashes and gray during ISI/omissions/no-image periods."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity uses the same time bin centers as the neural data. The bin centers are computed from the trial start/stop times, and presentation intervals are projected onto these bin centers.

ii.
```python
# Same centers used for both neural and image:
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
# Neural uses these for binning, image_series uses these for assignment:
in_window = (centers >= start) & (centers < stop)
```

iii. Alignment is achieved by using the same trial-relative time bins for all data streams.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` and `omitted` fields in the stimulus presentations table.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI uses the SDK's pre-computed `is_change` flag rather than computing change from consecutive image names.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is constructed. For each presentation row overlapping the trial, if `is_change` is True and the presentation is not omitted, all time bin centers falling within that presentation's time window are set to 1. All other bins are 0.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
# ...
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. From CONVERSION_NOTES.md Step 5: "Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. The output values are defined as `["no_change", "change"]`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    # ...
]
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity — uses the same time bin centers derived from trial start/stop times.

ii. Same `centers` array is used in `make_image_series()` for both image identity and change series.

iii. All outputs share the same time base as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` with timestamps from `/processing/running/speed/timestamps`.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Consistent with AllenSDK's `RunningSpeed` object.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are interpolated to the trial bin centers using linear interpolation (`np.interp`). The interpolated values are then discretized into 5 bins using global percentile-based bin edges computed across all valid trials in all eligible sessions.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```
Global edges computed as:
```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
```

iii. From CONVERSION_NOTES.md Step 5: "Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile edges are computed from all finite running speed samples across all valid trials in eligible sessions. The 20th, 40th, 60th, and 80th percentiles define 5 bins. Values are discretized using `np.digitize`.

ii.
```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. Instructions specify "discretized into five equal percentile bins." Global bin edges from the conversion: `[-0.023, 0.117, 10.039, 31.567]`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same time bin centers used for neural data, ensuring temporal alignment.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. All outputs share the same trial-relative time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), with blink filtering from `/acquisition/EyeTracking/likely_blink/data`, and timestamps from `/acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. Consistent with AllenSDK's `EyeTrackingTable` object and the whitepaper description.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are masked (set to NaN), then pupil area is converted to diameter via `2 * sqrt(area / pi)`. The diameter values are then interpolated to bin centers and discretized into 5 percentile bins using global edges.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
# ...
def pupil_area_to_diameter(area):
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
# ...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. From CONVERSION_NOTES.md Step 5: "Use processed pupil area after blink filtering, convert to equivalent diameter 2*sqrt(area/pi), interpolate to rebinned trial axis, discretize into 5 global percentile bins."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: global percentile edges computed from all finite pupil diameter values across all valid trials, then discretized with `np.digitize` into 5 bins.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
```

iii. Global pupil bin edges from the conversion: `[72.634, 83.376, 93.055, 104.904]`.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same time bin centers as neural data.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean fields in `/intervals/trials`.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
# ...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The four outcome classes match the SDK's trial table semantics. Order: `["hit", "miss", "false_alarm", "correct_reject"]`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is a static per-trial categorical value. Each trial gets exactly one outcome index (0-3). This value is repeated across all time bins in the trial to maintain a uniform `(n_output, n_timepoints)` shape.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins,
    pupil_bins,
    outcome_series,
])
```

iii. From CONVERSION_NOTES.md Step 5: "Single categorical value per trial, repeated across all time bins in that trial to keep output arrays uniformly time-varying."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) NaN values in trial flags are treated as 0/False via `np.nan_to_num`. (2) Sessions missing eye tracking are excluded entirely. (3) Blink frames in pupil data are set to NaN before interpolation; interpolation fills NaN gaps via `np.interp` which skips non-finite values. (4) Non-finite change_time values are stored as `math.nan`. (5) All-zero neural trials (from sparse event detection) are kept — 3947 such trials (4.68%) were verified against raw NWB data. (6) Omitted stimulus presentations are mapped to "gray" rather than a separate class.

ii.
```python
# NaN handling in trial flags:
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)

# Blink handling:
pupil_area[likely_blink] = np.nan

# Interpolation handles NaN:
finite = np.isfinite(times) & np.isfinite(values)
```

iii. From CONVERSION_NOTES.md Step 10: "all-zero neural warnings arise from genuine zero-event trials in the raw event-detection matrices rather than off-by-one errors at trial boundaries."

## 9-a. What are the most time-consuming steps of the code?

i. The conversion pass is the most time-consuming at ~686 seconds for 281 sessions. The preview pass takes ~65 seconds. Per-session conversion time averages ~2.4 seconds. Total elapsed ~758 seconds.

ii. From conversion output:
```
Preview pass completed in 65.0s
Conversion pass completed in 686.0s
```

iii. From CONVERSION_NOTES.md Step 7: per-session conversion takes ~5.95s for large sessions, with the full conversion conservatively estimated under 30 minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the per-bin neural rebinning loop, which iterates over each time bin to sum events:

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```
This could be replaced with a cumulative sum approach. Similarly, `make_image_series` loops over presentation rows.

iii. From CONVERSION_NOTES.md Step 6: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design means each NWB file is opened and partially read twice: once in the preview pass (to extract trial specs, running/pupil values for global edges) and once in the conversion pass (full data extraction). Running speed and pupil data are loaded in both passes.

ii.
```python
# Preview pass: session_preview() opens each file
def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        # reads trials, running, pupil...

# Conversion pass: convert_session() opens same file again
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        # reads trials, running, pupil, neural again...
```

iii. From CONVERSION_NOTES.md: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several potentially unnecessary operations: (1) The `input` arrays are all empty `(0, T)` matrices since no decoder inputs are specified, yet they are created for every trial. (2) Trial outcome is repeated across all time bins despite being a static per-trial variable — the decoder framework may only use one value per trial. (3) The `change_time` field is stored in `TrialSpec` but never used in the conversion (image change is derived from presentations' `is_change` flag instead). (4) Processing plots are generated for sample mode but may not be examined.

ii.
```python
# Empty input arrays:
input_trial = np.empty((0, T), dtype=np.float32)

# Repeated outcome:
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)

# change_time stored but unused:
change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
```

iii. The empty input arrays are required by the target format specification. The repeated outcome is a design choice to keep all outputs as `(n_output, n_timepoints)`.
