# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, listing all `.nwb` files under the experiment directory. It does NOT use the AllenSDK cache. A two-pass workflow is used: a lightweight preview pass scans all NWB files to identify eligible sessions and collect global statistics, then a conversion pass reopens each eligible file to extract neural and behavioral data.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        experiment_id = int(decode_scalar(f["/identifier"][()]))
        ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
        subject_id = decode_scalar(f["/general/subject/subject_id"][()])
        ...
```

iii. The AI explains in CONVERSION_NOTES.md Step 6: "Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment." The AI noted the SDK semantics are mirrored by reading the same NWB fields directly.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject_id` field read from `/general/subject/subject_id` in each NWB file. Unique subjects are accumulated in a list as new subject IDs are encountered during processing.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. Subject IDs are read from the NWB metadata, consistent with the Allen SDK's subject identification.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. The AI does NOT group multiple experiments by `ophys_session_id`. This means if a session has multiple imaging planes (multiple experiment files), each plane becomes its own "session" in the output.

ii.
```python
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The AI treats each experiment file as a session unit. The CONVERSION_NOTES don't explicitly discuss whether to group by `ophys_session_id`. Since VisualBehavior is single-plane imaging, most sessions have exactly one experiment, but the data shows some sessions have up to 7 experiments (mean 1.15).

## 1-d. How are the data split into trials?

i. Trials are read from the `/intervals/trials` table in each NWB file. The `build_trial_specs` function extracts valid (non-aborted, non-auto-rewarded) trials with exactly one outcome flag set. Trial boundaries are defined by `start_time` and `stop_time`, giving variable-length trials.

ii.
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    ...
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        ...
```

iii. The AI uses the NWB trials table directly, mirroring SDK trial semantics. Trials with ambiguous outcomes (not exactly one flag set) are treated as errors rather than silently skipped.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials must have exactly one outcome flag set. Sessions without eye tracking are excluded entirely. Sessions with fewer than 2 valid trials or insufficient valid pupil samples are also excluded.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
...
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The AI justifies excluding sessions without eye tracking because pupil diameter is a required decoder output. The minimum 2-trial and 2-valid-pupil thresholds prevent degenerate sessions. Note: unlike the reference, the AI does NOT filter on `change_time.notna()`. Trials with NaN change_time are included as long as they have a valid outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the **event detection** data (`/processing/ophys/event_detection/data`), NOT from dF/F traces. Only neurons with `valid_roi == True` are included.

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

iii. The AI chose events over dF/F because CONVERSION_NOTES Step 4 states: "Prefer event-detection outputs for neural activity because the analysis paper explicitly uses them." Step 5 reaffirms: "Neural signal = event-detection output, not dF/F: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events."

## 2-b. How is the `neural` data processed?

i. Event detection magnitudes are summed within 100ms time bins spanning each trial. For each bin, all ophys frames falling within that bin's time window contribute their event magnitudes via summation.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI explains in CONVERSION_NOTES Step 5: "Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions." The metadata describes this as "sum event magnitudes within each 100 ms trial bin."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` flag from the cell specimen table. Only neurons where `valid_roi == True` are included. No additional quality filtering is applied.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI follows the SDK's default ROI filtering behavior. CONVERSION_NOTES Step 4 notes: "Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Time bins are computed from `start_time` to `stop_time` using 100ms steps. Ophys frame indices are found via `np.searchsorted` against the bin boundaries, and event magnitudes within each bin are summed.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. Alignment is to the trial start time, as stated in metadata: `temporal_alignment_event: "trial start"`, `off_start: 0.0`, `off_end: None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to **100ms** time bins (`BIN_SIZE_SEC = 0.1`). This is a deliberate rebinning from the native ophys frame rate. The justification is that native rates differ across session types (31 Hz single-plane vs 11 Hz multi-plane) and the target format requires uniform bin size.

ii.
```python
BIN_SIZE_SEC = 0.1
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

Metadata:
```python
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. CONVERSION_NOTES Step 5 states: "Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions. I will rebin all trials to a common bin width based on ophys-time alignment." And: "Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** (`/intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields. This is different from the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    interval_root = f["/intervals"]
    keep_names = [
        "start_time", "stop_time", "image_name", "omitted", "is_change", ...
    ]
    ...
```

iii. The AI uses the presentations table because it provides flash-by-flash stimulus timing, allowing the image identity to reflect the actual stimulus state at each time bin (image during flash, gray during ISI/omission).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Stimulus presentations are projected onto the 100ms trial time bins. For each bin center, the overlapping stimulus presentation determines the image identity. During inter-stimulus intervals or omitted flashes, the identity is "gray." A global mapping from image names to integer indices is built, with "gray" as index 0.

ii.
```python
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        ...
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. CONVERSION_NOTES Step 5: "Image identity will include a `gray` class: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers as the neural data, using the same `build_bin_centers` function. The stimulus presentations overlapping each bin center determine the image identity.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. All output variables share the same time axis (bin centers), ensuring alignment with the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table. A bin is marked as change (1) if any overlapping presentation has `is_change == True` and is not omitted.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI uses the presentations table's `is_change` flag, which marks the actual change stimulus presentation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The change series is binary (0/1). It is 1 during the time bins overlapping a change presentation (where `is_change == True` and `omitted == False`), and 0 otherwise. The duration of the change marker corresponds to the presentation duration (~250ms flash) rather than a fixed 750ms window.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
for idx in row_idx:
    start = float(presentations["start_time"][idx])
    stop = float(presentations["stop_time"][idx])
    ...
    is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. CONVERSION_NOTES Step 5: "Image-change target will mark the changed-image presentation, not only a single instant: Marking the post-change image flash interval is more robust after binning."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No additional thresholding is needed.

ii.
```python
image_change_value_names = ['no_change', 'change']
```

iii. The binary nature directly follows from the instruction: "Image change, binary variable."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100ms bin centers as neural data and other outputs.

ii. See 3-c code snippet.

iii. All outputs share the same temporal grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This matches the SDK's running speed data source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 100ms bin centers using `np.interp`. Edge values are used for extrapolation (no NaN). Then discretized into 5 equal percentile bins computed globally across all eligible sessions.

ii.
```python
def interpolate_series(times, values, query_times):
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. CONVERSION_NOTES Step 5: "Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global percentile-based edges. The `compute_bin_edges` function calculates 4 inner edges at the 20th, 40th, 60th, and 80th percentiles of all valid running speed values across eligible sessions. `np.digitize` maps values to bins 0-4.

ii.
```python
def compute_bin_edges(values, nbins):
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values, edges):
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. Global bin edges ensure consistent categories across sessions, as required for decoder outputs.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as the neural data.

ii. See 5-b code snippet.

iii. All data streams share the same temporal grid per trial.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), converted to equivalent diameter. Blink frames are identified from `/acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The AI chose to convert area to diameter, consistent with the whitepaper stating "pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Raw pupil area is loaded, blink frames are set to NaN, area is converted to equivalent circle diameter via `2*sqrt(area/pi)`, then interpolated to 100ms bin centers and discretized into 5 global percentile bins. Sessions without eye tracking are excluded entirely.

ii.
```python
def pupil_area_to_diameter(area):
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 global percentile-based bins using `compute_bin_edges` and `discretize_with_edges`.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Global bin edges computed across all eligible sessions for consistent categories.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers as neural and other output data.

ii. See 5-b code pattern.

iii. Shared temporal grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean flags `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table (`/intervals/trials`). The AI requires exactly one of these to be True per valid trial.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

iii. The four outcome types are the SDK's canonical trial outcomes for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcome is mapped to an integer code (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and repeated across all time bins in the trial to maintain the `(n_output, n_timepoints)` format.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. CONVERSION_NOTES Step 5: "Trial outcome will be repeated across time bins: Although static per-trial, repeating it across the trial keeps every output array in `(n_output, n_timepoints)` form."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: Sessions without eye tracking (3 files) are excluded entirely.
- **NaN change_time**: Stored as NaN in the TrialSpec; these trials are still included (unlike the reference which excludes them).
- **Blinks**: Set pupil area to NaN before interpolation; interpolation fills gaps via linear interpolation.
- **Extrapolation**: Running and pupil interpolation uses edge values (no NaN produced); non-finite interpolation results raise ValueError.
- **All-zero neural trials**: ~4.68% of trials have all-zero event data; these are kept as-is.
- **Sessions with <2 trials or insufficient pupil**: Excluded.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
pupil_area[blink] = np.nan
...
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
```

iii. CONVERSION_NOTES Step 5: "Exclude sessions with missing eye tracking: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data." The all-zero neural warnings are documented in Step 9 as "consistent with sparse event-detection matrices."

## 9-a. What are the most time-consuming steps of the code?

i. The preview pass (scanning all 284 NWB files) takes ~65s, and the conversion pass takes ~686s. The conversion pass is the bottleneck, dominated by reading neural event data and rebinning per trial.

ii. From CONVERSION_NOTES Step 9: Preview pass 65.0s, Conversion pass 686.0s, Total 757.7s.

iii. The two-pass design means each NWB file is opened twice (once in preview, once in conversion). The preview avoids loading neural matrices to stay lightweight.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural event summation loop in `convert_session` iterates over each time bin sequentially:

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 6 acknowledges: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU." A cumulative sum approach could vectorize this.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass workflow opens each NWB file twice: once during the preview pass (to collect global statistics) and again during the conversion pass (to extract neural data and build trial arrays). Trial tables and presentations are read in both passes.

ii.
```python
# Pass 1 (preview)
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f: ...

# Pass 2 (conversion)
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f: ...
```

iii. CONVERSION_NOTES Step 6: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and processes the full stimulus presentations table including fields like `is_sham_change`, `trials_id`, and `active` that are not used in the final output construction. The preview pass also collects running and pupil values within trial windows for global percentile computation, which duplicates some of the interpolation work done in the conversion pass.

ii.
```python
keep_names = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "is_sham_change", "trials_id", "active",
]
```

iii. The extra fields are read for potential sanity checks but not used in the final output.
