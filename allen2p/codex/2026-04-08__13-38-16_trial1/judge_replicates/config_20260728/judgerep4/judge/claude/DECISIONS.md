# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py` rather than the AllenSDK Python API. It lists all `.nwb` files under the `behavior_ophys_experiments` directory, performs a two-pass workflow: first a "preview" pass to identify eligible sessions and collect global statistics, then a "conversion" pass to build the final arrays.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

# Preview pass
eligible, excluded = collect_previews(
    files=files,
    bin_size_sec=BIN_SIZE_SEC,
    sample_mode=args.sample,
    required_eligible=2,
)

# Conversion pass
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
```

iii. The AI chose to use direct `h5py` reads because the installed NWB/pynwb stack was incompatible with the files in the environment. The two-pass design allows computing global percentile bin edges before the conversion pass, at the cost of reading each NWB file twice.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `subject_id` read from `/general/subject/subject_id` in each NWB file. Each unique subject ID is assigned an index.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
# ...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The subject ID is read directly from the NWB file metadata, which is the standard identifier for each mouse.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. The AI does NOT group multiple experiments (imaging planes) from the same `ophys_session_id` into a single session. Each experiment file = one session in the output.

ii.
```python
for i, preview in enumerate(eligible, start=1):
    # Each preview corresponds to one NWB experiment file
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The AI noted in CONVERSION_NOTES.md that each experiment corresponds to one imaging plane in one session. The code treats each experiment as a separate session rather than grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined from the `/intervals/trials` table in each NWB file. The `build_trial_specs()` function filters out aborted and auto-rewarded trials, and requires exactly one valid outcome (hit/miss/false_alarm/correct_reject). Trial time windows use `start_time` to `stop_time` from the trials table, giving variable-length trials. The time window is then rebinned into 100ms bins.

ii.
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        # ...

# Rebinning into 100ms bins
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The AI followed the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. The requirement for exactly one valid outcome is a stricter check than the reference.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they are aborted, auto-rewarded, or do not have exactly one valid outcome flag. Sessions are excluded if they have missing eye tracking, fewer than 2 valid trials, or insufficient valid pupil samples. The AI also excludes sessions without eye tracking since pupil diameter is a required output.

ii.
```python
# Trial filtering
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)

# Session filtering
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The AI documented that 3 NWB files lacked eye tracking and were excluded. The exact-one-outcome check is stricter than the reference, which just checks boolean flags sequentially.

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

iii. The AI justified using event detection over dF/F because "the analysis paper explicitly uses discrete calcium events." The CONVERSION_NOTES.md Step 4 states: "Use the available processed neural signal in a way that matches the paper/code path. Prefer event-detection outputs."

## 2-b. How is the `neural` data processed?

i. Event detection data is rebinned from native ophys timestamps into 100ms bins by summing event magnitudes within each bin. Only valid ROIs are included.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI chose 100ms bins as a common time base across single-plane (31 Hz) and multi-plane (11 Hz) recordings, noting it "is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons with `valid_roi == True` in the cell specimen table are included. No additional quality filtering is applied beyond this SDK-standard filter.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI noted: "SDK code path identified in Step 1 also filters invalid ROIs (`valid_roi == True`) at load time; this is consistent with the whitepaper's QC emphasis."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, uniform 100ms bins are created from `start_time` to `stop_time`, and ophys frames falling within each bin are summed.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
# Sum events within each bin
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI chose trial start as the alignment event, documenting: "The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to 100ms bins (`BIN_SIZE_SEC = 0.1`). This is a deliberate choice, different from the native ophys frame rate.

ii.
```python
BIN_SIZE_SEC = 0.1
# ...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```
Metadata: `"time_bin_size": BIN_SIZE_SEC * 1000.0` (100.0 ms)

iii. The AI justified this: "Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions." The 100ms bin was chosen as a compromise.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** (`/intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields. A `gray` class is used for inter-stimulus intervals and omitted flashes.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    interval_root = f["/intervals"]
    keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]
    # ...

def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        # ...
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI used the stimulus presentations table to get per-flash image identity with precise timing, rather than using the trial-level `initial_image_name`/`change_image_name` fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping. The mapping includes a `gray` class at index 0 for ISI periods and omitted flashes, plus all unique image names found across eligible sessions. Image identity is time-varying: each 100ms bin is assigned the image code of any non-omitted presentation overlapping that bin's center, defaulting to `gray`.

ii.
```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The AI included a `gray` class because "Trials contain gray ISI and omission periods; using an explicit gray/blank class keeps the categorical time series defined at every bin."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is projected onto the same 100ms bin centers as the neural data. Each bin center is checked against presentation start/stop times to determine which image (if any) is on screen.

ii.
```python
in_window = (centers >= start) & (centers < stop)
if not np.any(in_window):
    continue
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. Both neural and output data share the same bin centers derived from `build_bin_centers()`, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI used the `is_change` flag from stimulus presentations rather than the trial-level `go` flag combined with `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary time-varying variable. It is 1 during the presentation window of any flash marked `is_change` (and not omitted) in the stimulus presentations table, and 0 otherwise.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
for idx in row_idx:
    # ...
    is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. The change window corresponds to the duration of the changed image presentation (~250ms), not a fixed 750ms window as in the reference.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — image change is inherently binary (0 = no change, 1 = change).

ii. N/A (binary by construction)

iii. The variable is binary by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity — projected onto the shared 100ms bin centers.

ii. Same as 3-c.

iii. Both use the same `centers` array from `build_bin_centers()`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB files.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This matches the standard SDK running speed data source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to the 100ms bin centers using `np.interp` (which extrapolates using the first/last values rather than NaN), then discretized into 5 percentile-based bins using global edges.

ii.
```python
def interpolate_series(times, values, query_times):
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)

# Global bin edges
running_edges = compute_bin_edges(running_pool, 5)

# Discretization
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Global percentile edges ensure consistent bin definitions across sessions. The `np.interp` extrapolation uses first/last values rather than NaN.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using `np.digitize` with globally computed quantile edges.

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

iii. The 4 internal edges (20th, 40th, 60th, 80th percentiles) create 5 bins with approximately equal sample counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers used for neural data.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. Same bin centers ensure alignment between neural and running speed data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area) and `/acquisition/EyeTracking/likely_blink/data` (blink detection). Eye tracking timestamps come from `/acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The AI chose `area_raw` and converted to diameter, rather than using a pre-computed width/diameter field directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then pupil area is converted to diameter via `2 * sqrt(area / pi)`. The result is interpolated to 100ms bin centers and discretized into 5 percentile-based bins.

ii.
```python
def pupil_area_to_diameter(area):
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. Converting area to diameter via circular assumption. Blink frames are masked before interpolation to avoid artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 equal-percentile bins using globally computed quantile edges from all finite pupil samples.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
# ...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Same approach as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers used for neural data.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the `/intervals/trials` table.

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

iii. Same four outcome categories as the reference, with NaN-to-False conversion for safety.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is a static per-trial categorical variable (0-3 corresponding to hit/miss/false_alarm/correct_reject). It is replicated across all time bins within the trial.

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

iii. Replicating the static value across time bins keeps the output array shape consistent at `(n_output, n_timepoints)`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye tracking data are excluded entirely (3 sessions).
- **Blink frames**: Set to NaN before pupil area-to-diameter conversion and interpolation.
- **Non-finite interpolation**: The code raises a `ValueError` if running or pupil interpolation produces non-finite values, preventing silent data corruption.
- **NaN outcome flags**: `nan_to_num` converts NaN booleans to False before trial filtering.
- **Trials with non-unique outcomes**: Trials without exactly one True outcome flag raise an error.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
# ...
pupil_area[likely_blink] = np.nan
# ...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
# ...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
```

iii. The AI documented these choices in CONVERSION_NOTES.md, noting that missing eye tracking is handled by exclusion since pupil diameter is a required output.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the conversion pass where each NWB file is opened, neural event data is loaded, and per-trial rebinning is performed. The preview pass is also I/O-bound but lighter since it skips neural data loading.

ii. N/A (architectural observation)

iii. The AI documented: "Preview pass: 65.0s, Conversion pass: 686.0s, Total: 757.7s" for the full dataset. The two-pass design means each NWB file is opened twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural rebinning loop iterates over each time bin sequentially, summing event data within each bin. This could be vectorized using cumulative sums.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI acknowledged this: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in the preview pass and once in the conversion pass. In the preview pass, it reads trials, running speed, pupil data, and presentations. In the conversion pass, it re-reads all of these plus neural event data.

ii.
```python
# Preview pass
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        # reads trials, running, pupil, presentations...

# Conversion pass
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        # reads same data again plus neural events...
```

iii. The AI documented this as intentional: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes pupil area-to-diameter conversion (`2*sqrt(area/pi)`) from raw area, which adds unnecessary computation. The reference uses `pupil_width` directly. Additionally, the two-pass architecture reads behavioral data (running, pupil, presentations) twice per session. The `interpolate_series` function sorts timestamps each time it is called, even though they are already sorted. The processing plots infrastructure exists but only runs when `--show-processing` is passed.

ii.
```python
def pupil_area_to_diameter(area):
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)

def interpolate_series(times, values, query_times):
    order = np.argsort(t)  # unnecessary if already sorted
```

iii. The area-to-diameter conversion is a deliberate choice by the AI based on the whitepaper stating "pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter."
