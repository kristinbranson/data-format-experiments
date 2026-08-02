# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB/HDF5 files using `h5py` rather than the AllenSDK Python API (because `pynwb` was incompatible in the runtime environment). It enumerates all `.nwb` experiment files under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` with a glob pattern. A two-pass approach is used: first a lightweight "preview" pass scans all files to identify eligible sessions and collect global statistics (running speed, pupil values, image names), then a "conversion" pass reopens each eligible file to extract full neural and behavioral data.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

# Preview pass
eligible, excluded = collect_previews(files=files, ...)

# Conversion pass
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(preview=preview, ...)
```

iii. The AI documented in CONVERSION_NOTES.md Step 6 that direct HDF5 reads were used because the installed NWB stack was incompatible. The two-pass design avoids storing large neural matrices before global bin edges are computed.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the NWB file metadata field `/general/subject/subject_id`. Each unique subject ID is added to a `subjects` list as sessions are processed. A `subject_idx` array maps each session to its subject.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])

# In main():
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. This follows the standard AllenSDK approach where each experiment NWB file contains a subject identifier.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one session. The AI does not group multiple experiments from the same ophys session; each experiment file becomes one entry in the `neural`, `input`, and `output` session lists.

ii.
```python
# Each file processed independently:
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(preview=preview, ...)
    neural_sessions.append(neural_trials)
    ...
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that the local data contains 284 NWB experiment files across 247 unique ophys sessions (some sessions have multiple imaging planes). Each experiment file = one imaging plane in one session, and is treated as a separate session entry.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB `/intervals/trials` group. The trial table fields include `start_time`, `stop_time`, `go`, `catch`, `aborted`, `auto_rewarded`, outcome flags, `change_time`, and image names. Each non-excluded trial becomes one entry in the per-session trial list with its own time window defined by `start_time` to `stop_time`.

ii.
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    names = ["id", "start_time", "stop_time", "go", "catch", "aborted", "auto_rewarded",
             "hit", "miss", "false_alarm", "correct_reject", "change_time", "change_frame",
             "initial_image_name", "change_image_name"]
    return read_interval_group(group, names)

def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    # Iterates through all trials, filters, creates TrialSpec objects
```

iii. This matches the SDK's trial definitions from the behavior stimulus trial log, consistent with the whitepaper and methods text.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes `aborted` trials and `auto_rewarded` trials, keeping only GO and CATCH trials. Additionally, each remaining trial must have exactly one outcome flag set (hit, miss, false_alarm, or correct_reject). Sessions missing eye tracking are excluded entirely (3 sessions). Sessions with fewer than 2 valid trials would also be excluded.

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

# Session exclusion:
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. The AI justified this in CONVERSION_NOTES.md Step 5: "Keep GO and CATCH trials, exclude aborted and auto_rewarded exactly as instructed and consistent with the trial table semantics."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the event-detection output at `/processing/ophys/event_detection/data` in the NWB files, not from dF/F traces.

ii.
```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    ...
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 1: "Neural signal = event-detection output, not dF/F: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events."

## 2-b. How is the `neural` data processed?

i. Event detection magnitudes are rebinned into 100ms time bins by summing all event values from ophys frames that fall within each bin. Each trial's time window is divided into bins starting from trial start_time to stop_time. For each bin, ophys frames within the bin boundaries are identified via `searchsorted`, and event magnitudes are summed.

ii.
```python
def convert_session(...):
    ...
    for spec in preview.trial_specs:
        starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
        frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
        frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
        T = centers.size
        n_neurons = event_data.shape[1]
        neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
        for b in range(T):
            lo = int(frame_starts[b])
            hi = int(frame_ends[b])
            if hi > lo:
                neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI noted this is the raw event-detection output (not the filtered_events used for visualization), summed within bins to create a consistent time resolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` flag from the cell specimen table. Only ROIs marked as valid are included. This is done by indexing event_rois against the valid_roi array.

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

iii. CONVERSION_NOTES.md Step 5: "The SDK code path filters invalid ROIs (valid_roi == True) at load time; this is consistent with the whitepaper's QC emphasis."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. Each trial's time window runs from `start_time` to `stop_time` as defined in the trials table. Bins are constructed starting from `start_time` in increments of the bin size (100ms). Ophys timestamps are used to identify which frames fall within each bin via binary search (`searchsorted`).

ii.
```python
def build_bin_centers(start_time: float, stop_time: float, bin_size_sec: float):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...
    centers = starts + 0.5 * widths
    return starts, ends, centers

# In convert_session:
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The instructions say "Temporally align based on ophys timestamp" and the AI's metadata sets `temporal_alignment_event` to "trial start" with `off_start = 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 100ms (0.1 seconds). Yes, temporal rebinning is applied: native ophys timestamps (31 Hz single-plane or 11 Hz multi-plane) are rebinned into uniform 100ms bins. Event magnitudes within each bin are summed, and behavioral signals are interpolated to bin centers.

ii.
```python
BIN_SIZE_SEC = 0.1

# metadata:
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table(s) found under `/intervals/*_presentations` in the NWB files, specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    ...
    keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]

def unique_nonempty_images(presentations: Dict[str, np.ndarray]) -> List[str]:
    names = [str(x) for x in presentations["image_name"]
             if str(x) not in {"", "nan", "None", "omitted"}]
    return sorted(set(names))
```

iii. The AI documented this in CONVERSION_NOTES.md Step 5 variable mapping table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, stimulus presentations overlapping the trial window are identified. For each 100ms bin, the bin center is compared to presentation intervals. If a bin center falls within a non-omitted stimulus presentation, the image name is assigned; otherwise it is "gray" (index 0). Omitted presentations are also treated as "gray". The global vocabulary is "gray" + all sorted unique image names.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        start = float(presentations["start_time"][idx])
        stop = float(presentations["stop_time"][idx])
        ...
        in_window = (centers >= start) & (centers < stop)
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
    return image_series, change_series
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: "Image identity will include a gray class: Trials contain gray ISI and omission periods; using an explicit gray/blank class keeps the categorical time series defined at every bin."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is projected onto the same 100ms bin centers used for the neural data, both derived from the same `build_bin_centers()` call for each trial. This ensures temporal alignment.

ii.
```python
# In convert_session, same centers used for neural and output:
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
# Neural uses these for frame binning
# Image identity uses centers for presentation overlap check
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. Alignment is inherent since both neural and output data are constructed on the same per-trial time grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, combined with the `omitted` field and presentation timing (`start_time`, `stop_time`).

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    ...
    change_series = np.zeros(centers.shape, dtype=np.int16)
    for idx in row_idx:
        ...
        is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
        if is_change and not omitted:
            change_series[in_window] = 1
    return image_series, change_series
```

iii. CONVERSION_NOTES.md Step 5: "Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation within a trial window, if `is_change` is True and the presentation is not omitted, all bins whose centers fall within that presentation's time window are set to 1. Otherwise bins remain 0.

ii. (Same code as 4-a above)

iii. CONVERSION_NOTES.md Step 5 Key Decision 9: "Image-change target will mark the changed-image presentation, not only a single instant: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed. It is 1 during the changed-image presentation bins and 0 otherwise.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
# ...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The instructions specify "binary variable" with value 1 right after a change in image identity, otherwise 0.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Aligned using the same 100ms bin centers as the neural data, identical to image identity alignment.

ii. (Same `centers` variable used for both neural and change_series)

iii. Same justification as 3-c.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` with timestamps from `/processing/running/speed/timestamps` in the NWB files.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "Running speed timeseries -> output[2] (running_speed_bin)"

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are interpolated from their native timestamps to the 100ms bin centers using linear interpolation (`np.interp`). The interpolated values are then discretized into 5 quintile bins using global bin edges computed from all running speed values across all eligible sessions and valid trial windows.

ii.
```python
def interpolate_series(times, values, query_times):
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)

running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. CONVERSION_NOTES.md Step 5: "Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile bin edges are computed from the pooled running speed values across all eligible sessions within valid trial time windows. Values are digitized into 5 bins (quintiles) using `np.digitize` with edges at the 20th, 40th, 60th, and 80th percentiles.

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

running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 10: "Running and pupil bin edges will be global, not per-session: One consistent categorical definition across all sessions is required for decoder outputs and output_values."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers used for the neural data, ensuring temporal alignment.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. Same alignment approach as all other time-varying outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), `/acquisition/EyeTracking/eye_tracking/timestamps`, and `/acquisition/EyeTracking/likely_blink/data` (blink indicator).

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. CONVERSION_NOTES.md Step 5: "Use processed pupil area after blink filtering, convert to equivalent diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing steps: (1) Load raw pupil area. (2) Set blink frames to NaN using the `likely_blink` indicator. (3) Convert area to diameter using `2 * sqrt(area / pi)`. (4) Interpolate to 100ms bin centers using linear interpolation. (5) Discretize into 5 quintile bins using global edges.

ii.
```python
pupil_area[likely_blink] = np.nan

def pupil_area_to_diameter(area):
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)

pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. CONVERSION_NOTES.md Step 5: "Use processed pupil area after blink filtering, convert to equivalent diameter 2*sqrt(area/pi), interpolate to rebinned trial axis, discretize into 5 global percentile bins."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: global percentile bin edges computed from pooled pupil diameter values across all eligible sessions within valid trial windows, then digitized into 5 quintile bins.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
```

iii. Same as running speed justification (Key Decision 10).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers used for the neural data.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same alignment approach as all other time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-level boolean flags `hit`, `miss`, `false_alarm`, and `correct_reject` in the `/intervals/trials` group.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)

outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. CONVERSION_NOTES.md Step 5: "Trial outcome flags hit, miss, false_alarm, correct_reject -> output[4] (trial_outcome)"

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is determined by which of the four boolean flags is True for each trial (exactly one must be True after filtering). The outcome index is then repeated across all time bins in the trial to maintain the `(n_output, n_timepoints)` shape.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 11: "Trial outcome will be repeated across time bins: Although static per-trial, repeating it across the trial keeps every output array in (n_output, n_timepoints) form."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data strategies are employed:
- Sessions without eye tracking (3 sessions) are excluded entirely since pupil diameter is a required output.
- NaN values in trial flags (aborted, auto_rewarded, outcomes) are treated as 0/False via `np.nan_to_num`.
- NaN values in change_time are preserved as NaN in the TrialSpec.
- Blink frames in pupil data are set to NaN, and NaN values are excluded from interpolation.
- Running and pupil interpolation uses the nearest valid value for extrapolation (`left=v[0], right=v[-1]`).
- The code raises a ValueError if non-finite values remain in running or pupil after interpolation.
- All-zero neural trials (from sparse event detection) are left as-is rather than excluded.

ii.
```python
# NaN handling in trial flags:
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)

# Session exclusion for missing eye tracking:
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"

# Blink handling:
pupil_area[likely_blink] = np.nan

# Interpolation NaN handling:
finite = np.isfinite(times) & np.isfinite(values)
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
```

iii. CONVERSION_NOTES.md Step 10: "all neural data is zero warnings remained after the rerun: investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation."

## 9-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, the full conversion took 757.7 seconds total:
- Preview pass: 65 seconds (scanning all 284 NWB files for metadata and global statistics)
- Conversion pass: 686 seconds (processing 281 eligible sessions)
- Per-session conversion averages ~2.4 seconds, with larger sessions (more neurons, more trials) taking up to 6 seconds.

ii.
```python
# Timing instrumentation:
preview_start = time.perf_counter()
...
log(f"Preview pass completed in {time.perf_counter() - preview_start:.1f}s")
...
convert_start = time.perf_counter()
...
log(f"Conversion pass completed in {time.perf_counter() - convert_start:.1f}s")
```

iii. CONVERSION_NOTES.md Step 7 documents timing estimates and the two-pass overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural rebinning loop is the most obvious candidate. For each trial, it iterates over every time bin individually to sum ophys frames. This could be vectorized using cumulative sums. The per-presentation loop in `make_image_series` also iterates over individual stimulus presentations rather than using vectorized array operations.

ii.
```python
# Per-bin loop that could be vectorized:
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)

# Per-presentation loop that could be vectorized:
for idx in row_idx:
    start = float(presentations["start_time"][idx])
    stop = float(presentations["stop_time"][idx])
    ...
    in_window = (centers >= start) & (centers < stop)
    ...
```

iii. CONVERSION_NOTES.md Step 6 Code inefficiencies: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once during the preview pass (to read metadata, trial info, running/pupil values) and once during the conversion pass (to read neural data and reconstruct full trial arrays). Running speed timestamps/values and pupil timestamps/values are read in both passes.

ii.
```python
# Preview pass reads running/pupil:
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
        ...

# Conversion pass reads them again:
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
```

iii. CONVERSION_NOTES.md Step 6: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. The `collect_time_window_values` function concatenates raw running/pupil samples within trial windows during the preview pass purely for global percentile computation; these raw concatenated values are discarded.
2. The `sort_files_for_mode` function reads the cells metadata CSV to sort by neuron count in sample mode, but this sorting has no effect in full mode.
3. The `session_preview` also reads trial-level data and builds full TrialSpec objects to count valid trials, even for sessions that will be excluded.
4. `read_task_presentations` reads multiple presentation fields (like `active`, `is_sham_change`, `trials_id`) that are not used in the conversion.
5. Input arrays are created as empty `(0, T)` matrices for every trial, which carry no information.

ii.
```python
# Extra presentation fields read but unused:
keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change",
              "is_sham_change", "trials_id", "active"]

# Empty input arrays:
input_trial = np.empty((0, T), dtype=np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 12 that empty inputs are required by the target format specification even though no decoder inputs were requested.
