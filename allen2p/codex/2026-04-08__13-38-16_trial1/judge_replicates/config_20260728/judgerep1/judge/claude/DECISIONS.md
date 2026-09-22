# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, bypassing the AllenSDK's Python API. It discovers all experiment files by globbing `behavior_ophys_experiments/behavior_ophys_experiment_*.nwb` under the data root. A two-pass approach is used: first a "preview" pass reads lightweight metadata (trials, running, pupil, image names) from each NWB file, then a "conversion" pass re-opens each eligible file to extract full neural event data.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

# Preview pass
for idx, path in enumerate(files, start=1):
    preview = session_preview(path, bin_size_sec)

# Conversion pass
with h5py.File(preview.path, "r") as f:
    ophys_timestamps = np.asarray(f["/processing/ophys/dff/traces/timestamps"], ...)
    event_data, _ = load_neural_events(f)
    running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
    ...
```

iii. The AI notes in CONVERSION_NOTES.md that it uses direct HDF5 reads because "the installed NWB stack is incompatible with these files in this environment." The two-pass design avoids holding large neural matrices in memory before global percentile bins are known.

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `/general/subject/subject_id` from each NWB file during the preview pass. Each unique `subject_id` string is assigned an index.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
# ...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The NWB `subject_id` field is the canonical identifier for each mouse. This is functionally equivalent to using the SDK's `mouse_id`.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. The AI does NOT group multiple experiments (imaging planes) that share the same `ophys_session_id` into a single session. Each file = one session in the output.

ii.
```python
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The AI's CONVERSION_NOTES.md states the local subset has "284 NWB experiment files" and "247 unique local ophys sessions," meaning some sessions have multiple planes. By treating each experiment as a session, the AI produces 281 sessions (after excluding 3 without eye tracking) rather than grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the `/intervals/trials` table in each NWB file. The AI builds `TrialSpec` objects from non-aborted, non-auto-rewarded trials. Each trial spans from `start_time` to `stop_time`, but is rebinned into 100ms bins rather than using native ophys frames.

ii.
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        ...
        specs.append(TrialSpec(
            start_time=float(trials["start_time"][idx]),
            stop_time=float(trials["stop_time"][idx]),
            change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
            outcome_idx=outcome_idx,
            is_go=bool(go[idx]),
            is_catch=bool(catch[idx]),
        ))

# In convert_session:
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. Trial definitions match the SDK semantics: go and catch trials are included, aborted and auto-rewarded are excluded. The AI additionally requires exactly one valid outcome flag per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `aborted` and `auto_rewarded` trials, and requiring exactly one valid outcome flag (`hit`, `miss`, `false_alarm`, or `correct_reject`). Sessions without eye tracking are excluded entirely. Sessions with fewer than 2 eligible sessions after filtering are rejected (though this minimum check is at the dataset level, not per-session).

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")

# Session-level exclusion
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The AI's trial filtering matches the instructions (exclude aborted and auto-rewarded). The additional check requiring exactly one outcome flag is a stricter validation. Excluding sessions without eye tracking is justified because pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `/processing/ophys/event_detection/data` in the NWB files — the event-detection output, NOT dF/F traces. Only neurons with `valid_roi == True` in the cell specimen table are included.

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

iii. The AI's CONVERSION_NOTES.md explains: "the paper explicitly states that analyses were performed on discrete calcium events" and therefore chose event-detection outputs over dF/F. The SDK's `valid_roi` filtering is applied.

## 2-b. How is the `neural` data processed?

i. Event detection magnitudes are summed within each 100ms time bin to produce a rebinned neural activity matrix. The bin centers are computed relative to trial start/stop times.

ii.
```python
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

iii. The AI justifies 100ms binning as "coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons with `valid_roi == True` are included. No additional quality filtering is applied beyond that.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The SDK's `valid_roi` filtering is the standard ROI quality control for this dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. The 100ms bins span from `start_time` to `stop_time`, producing variable-length trials.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
# build_bin_centers:
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
ends = np.minimum(starts + bin_size_sec, stop_time)
centers = starts + 0.5 * widths
```

iii. The instructions say "temporally align based on ophys timestamp." The AI aligns to trial start (which is defined relative to ophys timestamps), using ophys frame indices for binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to a common 100ms time bin size (`BIN_SIZE_SEC = 0.1`). This differs from the native ophys frame rate (~93ms for 11Hz multi-plane, ~32ms for 31Hz single-plane).

ii.
```python
BIN_SIZE_SEC = 0.1
# ...
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. The AI justifies this in CONVERSION_NOTES.md: "Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`/intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields. A "gray" class is used during ISI periods, omissions, and non-stimulus times.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        start = float(presentations["start_time"][idx])
        stop = float(presentations["stop_time"][idx])
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI uses the presentations table to construct a fine-grained time-varying image identity that includes ISI gray periods. This differs from using the trials table's `initial_image_name`/`change_image_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global mapping. The mapping includes "gray" as index 0 plus all unique non-omitted image names. Time bins are initialized to "gray" and then overwritten with the appropriate image code during stimulus presentation intervals.

ii.
```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The "gray" class allows the decoder to distinguish ISI periods from stimulus presentations. The total number of classes is 17 (1 gray + 16 task images from two 8-image sets).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is projected onto the same 100ms bin centers used for neural data. Each bin center is checked against the presentation start/stop times to determine the image displayed at that moment.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. By using the same `centers` array as the neural binning, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table's `is_change` field. It is 1 during the time bins overlapping with a change-presentation interval where `is_change` is true and the stimulus is not omitted.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI uses the presentations table `is_change` flag rather than the trials table `change_time` + `go` flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is constructed: 1 during the time bins overlapping with a stimulus-change presentation, 0 elsewhere. The change signal spans the duration of the change-stimulus flash (250ms image + possible ISI).

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
# ...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. No additional thresholding is needed — it is inherently binary.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is applied.

ii. N/A — binary by construction.

iii. The output values are `["no_change", "change"]`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100ms bin centers as neural data. Bins overlapping the change presentation are marked as 1.

ii. See 4-a code — uses same `centers` array.

iii. Alignment is guaranteed by sharing the time base.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This is the SDK's standard running speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to the 100ms bin centers using `np.interp` (linear interpolation with edge-value extrapolation), then discretized into 5 percentile-based bins using globally computed bin edges.

ii.
```python
def interpolate_series(times, values, query_times):
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)

# Global bin edges
running_edges = compute_bin_edges(running_pool, 5)

# Per-trial
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Percentile-based bins ensure roughly equal class counts. Edge-value extrapolation (rather than NaN) is used, and any non-finite values are explicitly checked and raise errors.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using quantile-based edges computed globally across all valid trial data. `np.digitize` assigns each value to a bin.

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

iii. The bin edges are 4 interior boundaries (20th, 40th, 60th, 80th percentiles), producing 5 bins numbered 0-4.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as neural data, ensuring alignment.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. Same time base guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), with blink frames identified from `/acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The AI uses pupil area (not pupil width) and converts to diameter using the formula `2 * sqrt(area / pi)`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN in the pupil area. Area is converted to equivalent circular diameter via `2 * sqrt(area / pi)`. The resulting diameter is interpolated to 100ms bin centers, then discretized into 5 percentile bins.

ii.
```python
def pupil_area_to_diameter(area):
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. The AI notes the whitepaper states "pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter." The AI reverses this: computing diameter from area.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile-based bins using globally computed edges.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Equal-count bins ensure balanced classes for decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers as neural data.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same time base guarantees alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the `/intervals/trials` table.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)

outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The four outcome categories are the standard SDK outcomes for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one of the four outcome flags must be True. The outcome index (0-3 mapping to hit/miss/false_alarm/correct_reject) is repeated across all time bins in the trial.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Trial outcome is static per-trial but repeated across time bins to maintain uniform `(n_output, n_timepoints)` shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 sessions excluded entirely (`excluded_reason = "missing_eye_tracking"`).
- **NaN values in trial fields**: `np.nan_to_num` converts NaN to 0 for boolean fields.
- **Non-finite interpolation**: Running and pupil interpolation results are checked for finiteness, raising `ValueError` if non-finite values exist.
- **Few trials**: Sessions with fewer than 2 valid trials are excluded.
- **Insufficient pupil data**: Sessions with fewer than 2 finite pupil samples are excluded.
- **Empty bins**: `build_bin_centers` ensures at least one bin even if `stop_time - start_time < bin_size_sec`.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"

# Non-finite check
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
```

iii. The AI is strict about data quality — raising errors rather than silently filling NaN values, which is a defensive approach.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are (1) the preview pass, which reads metadata from all 284 NWB files (~65s), and (2) the conversion pass, which re-reads each file and processes neural event data (~686s for 281 sessions).

ii. N/A

iii. Total elapsed time was ~758s. The conversion pass dominates because it loads full neural event matrices from disk.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural rebinning loop iterates over each time bin within a trial, performing a slice-and-sum on the event matrix. This could be vectorized using cumulative sums.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI acknowledges this in CONVERSION_NOTES.md: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once in the preview pass (for metadata, trial specs, running/pupil values, image names) and once in the conversion pass (for full neural data and trial construction). Running and pupil values are also read in both passes.

ii.
```python
# Preview pass
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
        ...

# Conversion pass
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
        ...
```

iii. The two-pass design is intentional to avoid holding large neural matrices in memory before global percentile bins are known, but it does duplicate I/O for running, pupil, and trial data.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects running and pupil data during the preview pass for computing global bin edges, then discards these samples and re-reads/re-interpolates the same data during the conversion pass. Also, the `make_image_series` function processes the stimulus presentations table in detail even though a simpler approach using `initial_image_name`/`change_image_name` from the trials table would suffice.

ii. N/A

iii. The double-read is a trade-off for memory efficiency. The presentations-based image identity construction is more complex than necessary but produces a more detailed output (including gray periods).
