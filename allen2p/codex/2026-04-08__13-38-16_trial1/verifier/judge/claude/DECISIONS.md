# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, listing all `.nwb` experiment files under the `behavior_ophys_experiments` directory. It does NOT use the AllenSDK `VisualBehaviorOphysProjectCache`. Instead, it scans all NWB files, builds lightweight previews of each, filters for eligibility (eye tracking present, >= 2 valid trials), then does a second conversion pass to extract neural/behavioral data.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
```
And loading within `convert_session()`:
```python
with h5py.File(preview.path, "r") as f:
    ophys_timestamps = np.asarray(f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64)
    event_data, _ = load_neural_events(f)
    running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
    ...
```

iii. The AI chose direct HDF5 reads because `pynwb` was incompatible in the runtime environment. The CONVERSION_NOTES.md states: "Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment."

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `/general/subject/subject_id` from each NWB file during the preview pass. Unique subject IDs are collected and indexed.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The subject ID is extracted directly from the NWB metadata, which is the canonical identifier for each mouse.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. Multi-plane sessions (where multiple experiments share the same `ophys_session_id`) are NOT grouped together — each experiment file becomes its own session in the output.

ii.
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The AI reads `ophys_session_id` from each NWB file but does not use it to group experiments into sessions. Each experiment file is processed independently. The CONVERSION_NOTES.md notes the experiment count (281 eligible) but does not discuss grouping experiments into sessions.

## 1-d. How are the data split into trials?

i. Trials are defined from the `/intervals/trials` group in each NWB file. Aborted and auto-rewarded trials are excluded. The remaining go and catch trials are included. Trial boundaries use `start_time` to `stop_time`, giving variable-length trials.

ii.
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    ...
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(...)
        ...
        specs.append(TrialSpec(
            trial_idx=idx,
            start_time=float(trials["start_time"][idx]),
            stop_time=float(trials["stop_time"][idx]),
            ...
        ))
```

iii. The AI uses the SDK's built-in trial table stored in the NWB file. Aborted and auto-rewarded trials are excluded per instructions. The AI additionally validates that each included trial has exactly one valid outcome flag.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted trials, (2) excluding auto-rewarded trials, (3) requiring exactly one outcome flag to be true. Sessions are excluded if: (1) eye tracking is missing, (2) fewer than 2 valid trials, (3) insufficient valid pupil samples (< 2 finite values).

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
...
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The AI excludes sessions without eye tracking because pupil diameter is a required output. The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `/processing/ophys/event_detection/data` (event-detection magnitudes), NOT from dF/F traces. Only neurons with `valid_roi == True` in the cell specimen table are included.

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

iii. From CONVERSION_NOTES.md: "Prefer events over dF/F because the analysis paper explicitly uses discrete calcium events." The AI justified this by noting the paper states analyses were performed on discrete calcium events.

## 2-b. How is the `neural` data processed?

i. The event-detection data is rebinned from native ophys timestamps into 100ms bins. Within each bin, event magnitudes are summed across all ophys frames falling into that bin.

ii.
```python
BIN_SIZE_SEC = 0.1
...
def build_bin_centers(start_time, stop_time, bin_size_sec):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...
    return starts, ends, centers

# In convert_session:
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. From CONVERSION_NOTES.md: "Common time base via uniform rebinned trial bins: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions. I will rebin all trials to a common bin width." The 100ms bin was chosen as "coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons with `valid_roi == True` in the cell specimen table are included. No additional quality filtering is applied.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. From CONVERSION_NOTES.md: "The SDK code path identified in Step 1 also filters invalid ROIs (`valid_roi == True`) at load time; this is consistent with the whitepaper's QC emphasis."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to trial start. The bin centers are computed from `start_time` to `stop_time` using 100ms bins. Ophys frames are mapped to bins via `np.searchsorted`.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. From CONVERSION_NOTES.md: "Alignment event = trial start: The trial itself is the natural unit requested by the user."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins to 100ms (0.1s) bins. This differs from the native ophys frame rate. All data streams (neural, running, pupil, image identity, image change) are aligned to the same 100ms time base.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. CONVERSION_NOTES.md: "Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`/intervals/*_presentations`), using the `image_name`, `start_time`, `stop_time`, `omitted`, and `is_change` fields. It is NOT derived from the trials table `initial_image_name`/`change_image_name` fields.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    ...
    keep_names = ["start_time", "stop_time", "image_name", "omitted",
                  "is_change", "is_sham_change", "trials_id", "active"]
    ...
```

iii. The AI used the stimulus presentations table to get frame-by-frame image identity, which provides more precise timing than the trial-level `initial_image_name`/`change_image_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is constructed as a time-varying categorical variable. For each 100ms bin, the code checks which stimulus presentation overlaps that bin center. If a non-omitted presentation is active, its image name is used; otherwise the bin is labeled "gray". A global mapping from image names to integer codes is built, with "gray" as index 0.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    change_series = np.zeros(centers.shape, dtype=np.int16)
    for idx in row_idx:
        ...
        in_window = (centers >= start) & (centers < stop)
        ...
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
        ...
```
And the global mapping:
```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The AI includes a "gray" category for inter-stimulus intervals and omitted flashes. This is a notable addition compared to the reference, which only tracks the current trial image (initial or change) without a gray category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers as the neural data, ensuring perfect alignment. The stimulus presentation windows are projected onto the bin centers.

ii. See 3-b code — `centers` are the same time points used for neural binning.

iii. Same time base ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table. It is 1 during the entire duration of the changed stimulus presentation, and 0 otherwise.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    ...
    is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
    if is_change and not omitted:
        change_series[in_window] = 1
    ...
```

iii. The AI uses the presentation-level `is_change` flag rather than deriving the change from trial-level `change_time` and `go` flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation that overlaps the trial window, if `is_change` is True and the presentation is not omitted, all bin centers within that presentation's `[start_time, stop_time)` window are set to 1. Otherwise they remain 0.

ii. See 4-a code.

iii. This approach marks the entire changed stimulus flash duration (250ms), not a fixed 750ms window. It also does not distinguish between go and catch trials for the change marker — it uses `is_change` which is only True for actual image changes (go trials), not sham changes.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100ms bin centers as neural data.

ii. See 3-c.

iii. Same time base ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This is the standard running speed data from the Allen SDK pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native timestamps to the 100ms bin centers using `np.interp` (linear interpolation with edge clamping). Then discretized into 5 percentile-based bins using global bin edges computed across all eligible sessions.

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

iii. Linear interpolation preserves the signal shape. Percentile-based binning ensures roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five bins are computed using global percentile edges (20th, 40th, 60th, 80th percentiles) from all valid running speed values across all included trials. `np.digitize` maps values to bins 0-4.

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

iii. Global percentile edges ensure consistent categories across sessions. The `np.clip` ensures values stay within valid bin range.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as the neural data.

ii. See 5-b.

iii. Same time base ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), converted to diameter via `2 * sqrt(area / pi)`. Blink frames are identified from `/acquisition/EyeTracking/likely_blink/data` and their pupil area values are set to NaN before conversion.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

```python
def pupil_area_to_diameter(area):
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter." The AI derives diameter from raw area using `2*sqrt(area/pi)`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After blink removal and area-to-diameter conversion, pupil diameter is interpolated from eye tracking timestamps to 100ms bin centers using `np.interp` with edge clamping. Then discretized into 5 percentile-based bins using global edges.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Same approach as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile-based bins using global edges computed from all valid (finite) pupil diameter values across all included trials.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
```

iii. Global percentile edges ensure consistent categories across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers as the neural data.

ii. See 6-b.

iii. Same time base ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The four outcome types are the SDK's canonical trial outcome labels for the change detection task. The AI validates that exactly one outcome flag is True per trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome index (0-3 mapping to hit/miss/false_alarm/correct_reject) is repeated across all time bins within a trial, creating a static-per-trial but time-varying-in-format output.

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

iii. The integer mapping follows the order `["hit", "miss", "false_alarm", "correct_reject"]`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye tracking data (3 sessions) are excluded entirely.
- **Blink frames**: Pupil area during blinks is set to NaN before diameter conversion; NaN values are handled during interpolation.
- **NaN trial fields**: `np.nan_to_num` converts NaN boolean fields to False.
- **Non-finite interpolation**: The code raises a `ValueError` if running or pupil interpolation produces non-finite values.
- **Empty bins**: Neural bins with no ophys frames remain zero.
- **Trial validation**: Trials without exactly one valid outcome flag raise an error.

ii.
```python
# Missing eye tracking
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
# Blink handling
pupil_area[likely_blink] = np.nan
# NaN handling
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
# Non-finite check
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
```

iii. The AI chose to exclude sessions without eye tracking rather than fill pupil values, since pupil diameter is a required output.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and processing each NWB file during the conversion pass. The preview pass took 65s and the conversion pass took 686s for the full dataset (281 sessions). The code uses a two-pass approach: a lightweight preview pass and a heavier conversion pass, which means each NWB file is opened twice.

ii. N/A

iii. From CONVERSION_NOTES.md: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural rebinning loop iterates over each time bin sequentially to sum event data within that bin. This could be vectorized using cumulative sums.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. From CONVERSION_NOTES.md: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once during the preview pass to determine eligibility and collect global statistics, and again during the conversion pass to extract neural and behavioral data.

ii. N/A

iii. This is intentional to avoid storing large neural matrices in memory before global percentile/bin definitions are known.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `ophys_session_id` and `session_type` metadata during the preview pass, which are used for logging but not included in the final output. The two-pass architecture also means running speed and pupil data are collected in the preview pass for global bin edge computation, then re-read and re-interpolated during the conversion pass.

ii. N/A

iii. The preview pass collects values needed for global bin edge computation, but re-reads them during conversion since only summary statistics (not per-frame values) are retained from the preview.
