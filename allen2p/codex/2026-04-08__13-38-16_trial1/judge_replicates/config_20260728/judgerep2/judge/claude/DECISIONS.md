# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from the local filesystem using `h5py`, bypassing the AllenSDK entirely. It lists all `.nwb` files under the experiment directory, performs a two-pass workflow: a lightweight "preview" pass to identify eligible sessions and compute global statistics, and a "conversion" pass to build the final arrays. The AI noted that `pynwb` failed in the environment so it chose direct HDF5 access.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

# In session_preview():
with h5py.File(path, "r") as f:
    experiment_id = int(decode_scalar(f["/identifier"][()]))
    ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
    subject_id = decode_scalar(f["/general/subject/subject_id"][()])
    ...
```

iii. The AI justified using h5py because `pynwb` was incompatible with the NWB files in the environment. It stated this mirrors SDK semantics already serialized into the NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject_id` field in each NWB file's `/general/subject/subject_id`. Each unique subject_id is assigned an index as it is encountered during processing.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The AI uses the NWB file's subject metadata directly. This is functionally equivalent to the reference's use of `mouse_id` from the experiment table.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment file (one imaging plane) as a separate session. It does NOT group multiple imaging planes from the same `ophys_session_id` into a single session.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

# Each NWB file becomes one session:
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The AI's CONVERSION_NOTES.md acknowledges that multi-plane sessions exist (mean 1.15 experiments per session locally), but treats each experiment file as a separate session. The reference code groups experiments by `ophys_session_id` and merges imaging planes.

## 1-d. How are the data split into trials?

i. Trials are defined from the `/intervals/trials` group in each NWB file. The AI uses `build_trial_specs()` to extract valid go/catch trials (excluding aborted and auto-rewarded). Each trial spans `start_time` to `stop_time`. The trial is then rebinned into uniform 100ms bins using `build_bin_centers()`.

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
            start_time=float(trials["start_time"][idx]),
            stop_time=float(trials["stop_time"][idx]),
            ...))

# In convert_session():
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The AI uses the SDK-defined trial table directly from the NWB files. Aborted and auto-rewarded trials are excluded as instructed. The AI additionally validates that each kept trial has exactly one outcome flag set.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding aborted trials, auto-rewarded trials, and trials where the outcome flags are ambiguous (not exactly one of hit/miss/false_alarm/correct_reject). Sessions missing eye tracking are excluded entirely (3 sessions). Sessions with fewer than 2 valid trials or insufficient valid pupil samples are also excluded.

ii.
```python
# In build_trial_specs():
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)

# In session_preview():
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The AI justified excluding sessions without eye tracking because pupil diameter is a required decoder output. The reference code does not explicitly filter by `change_time.notna()`, but the AI's requirement for exactly one valid outcome achieves a similar purpose since aborted/auto-rewarded trials (which may lack change_time) are already excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses **event detection** data from `/processing/ophys/event_detection/data`, NOT dF/F traces. Additionally, it filters neurons by the `valid_roi` flag from the cell specimen table.

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

iii. The AI justified using event detection because "the analysis paper explicitly states that analyses were performed on discrete calcium events." The reference code uses `dff_traces` (dF/F).

## 2-b. How is the `neural` data processed?

i. Event detection data is loaded per experiment, filtered by valid_roi, then rebinned into 100ms time bins by summing event magnitudes within each bin. Each bin's neural value is the sum of event amplitudes from all ophys frames falling within that bin's time window.

ii.
```python
# In convert_session():
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

iii. The AI chose to sum event magnitudes per bin rather than averaging, since events are sparse point-like signals. The reference code does not rebin — it uses dF/F at the native ophys frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` flag from the cell specimen table. Only neurons marked as valid ROIs are included.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI noted the SDK filters invalid ROIs by default when `exclude_invalid_rois=True`. The reference code does not apply additional filtering (the SDK's `dff_traces` accessed via the API already includes only valid ROIs by default).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. Ophys frames within each 100ms bin (from trial start to trial stop) are summed. The alignment event is the trial start (`off_start = 0.0`).

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The AI justified this by noting the trial itself is the natural unit requested by the user, so alignment to trial start with `off_start = 0.0` and variable trial lengths is appropriate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to a uniform 100ms (0.1s) bin size. This is different from the native ophys frame rate (~11 Hz for multiplane, ~31 Hz for single-plane). The reference code keeps data at the native ophys frame rate (~93ms for 11 Hz).

ii.
```python
BIN_SIZE_SEC = 0.1

def build_bin_centers(start_time: float, stop_time: float, bin_size_sec: float):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...
```

iii. The AI justified the 100ms bin size as "coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`/intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields. A "gray" class is used for inter-stimulus intervals and omitted flashes.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        start = float(presentations["start_time"][idx])
        stop = float(presentations["stop_time"][idx])
        ...
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI used the presentations table to get fine-grained, frame-by-frame image identity that includes gray periods. The reference code uses `initial_image_name` and `change_image_name` from the trials table, which only tracks the non-gray image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names from the presentations table are mapped to integer codes. A global vocabulary of all unique non-omitted image names plus "gray" is built. Each time bin is assigned the image code of the presentation active during that bin center, or "gray" if no presentation or an omitted presentation covers that bin.

ii.
```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The AI includes a "gray" class (17 total classes: gray + 16 images). The reference has no "gray" class — it uses only the 8 actual image names visible in each session, creating a variable number of image categories.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the 100ms bin centers, matching the neural data's rebinned time axis. For each bin center, the active stimulus presentation is looked up.

ii.
```python
in_window = (centers >= start) & (centers < stop)
```

iii. Both neural and image identity share the same `centers` time axis from `build_bin_centers()`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, combined with the presentation timing. It is 1 during bins where a change presentation is active, 0 otherwise.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI uses the presentations table's `is_change` flag rather than the trial-level `change_time` and `go` flag used by the reference.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The change series is a binary time-varying array. It is 1 for bins whose center falls within a presentation marked as `is_change` (and not omitted). The presentation window is typically 250ms (one flash duration).

ii. See 4-a above.

iii. The reference marks a 750ms window (flash + gray ISI) starting at `change_time` only for go trials. The AI marks only the presentation window (~250ms) for any trial with `is_change=True`, including catch trials with sham changes.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1). No thresholding is needed — it is already categorical.

ii. `change_series = np.zeros(centers.shape, dtype=np.int16)` with values set to 1 during change presentations.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the 100ms bin centers shared with neural data.

ii. See 3-c.

iii. Same alignment approach as all other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB files.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Same source variable as the reference code's `dataset.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native timestamps to the 100ms bin centers using `np.interp` (linear interpolation, with edge values extrapolated). Then it is discretized into 5 bins using global percentile edges.

ii.
```python
def interpolate_series(times, values, query_times):
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)

running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The AI uses `np.interp` with edge-value extrapolation (no NaN fill). The reference uses `scipy.interpolate.interp1d` with `fill_value=np.nan` and maps NaN to bin 0.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global percentile edges computed from all valid running speed samples across all eligible sessions. The edges are computed from 4 internal quantile boundaries.

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

iii. Both the AI and reference use percentile-based binning with 5 bins. The AI stores 4 internal edges while the reference stores all 6 edges (including 0th and 100th percentile). The `np.digitize` call achieves equivalent discretization.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as the neural data, ensuring alignment.

ii. `running_interp = interpolate_series(running_times, running_values, centers)`

iii. Same alignment approach as all other time-varying outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), with blink frames identified by `/acquisition/EyeTracking/likely_blink/data`. The AI converts area to diameter using the formula `2 * sqrt(area / pi)`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)

def pupil_area_to_diameter(area):
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. The AI uses `area_raw` and converts to diameter. The reference uses `pupil_width` directly from `dataset.eye_tracking`. These are different raw variables and different processing paths.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN in the raw pupil area. Area is converted to equivalent diameter via `2*sqrt(area/pi)`. The diameter is then interpolated to 100ms bin centers and discretized into 5 percentile-based bins.

ii. See 6-a above and:
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The reference uses `pupil_width` directly (no area-to-diameter conversion needed). The AI's approach adds an unnecessary conversion step since `pupil_width` is available directly. Both filter blinks.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 bins using global percentile edges from all valid pupil diameter samples.

ii. Same `compute_bin_edges` and `discretize_with_edges` functions.

iii. Both AI and reference use the same percentile-based 5-bin approach.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers as the neural data.

ii. `pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)`

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean fields in the trials table (`/intervals/trials`).

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

iii. Same source variables as the reference code.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to an integer index (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and replicated across all time bins in the trial.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Same approach as the reference, which also replicates the outcome across all time frames.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- Sessions missing eye tracking (3 sessions) are excluded entirely.
- Trials without exactly one valid outcome flag raise an error.
- Running speed extrapolation uses edge values (no NaN).
- Pupil interpolation: the AI requires finite values and raises a ValueError if non-finite values remain after interpolation.
- Sessions with fewer than 2 valid trials or insufficient valid pupil samples are excluded.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The reference is more lenient — it uses `fill_value=np.nan` for interpolation and maps NaN to bin 0 during discretization. The AI is stricter, excluding sessions rather than filling missing data.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies two main bottlenecks: (1) the preview pass scans all NWB files to collect metadata, running/pupil values, and trial specs; (2) the conversion pass re-opens each NWB file to load neural event data and build trial arrays. The total conversion took ~757s for 281 sessions.

ii. N/A (timing is printed during execution)

iii. The AI documents preview pass at 65s, conversion pass at 686s. The two-pass approach means each NWB file is opened twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural rebinning loop iterates over each time bin within each trial, summing event data for that bin. This could be vectorized using cumulative sums.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI acknowledged this inefficiency in CONVERSION_NOTES.md: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums."

## 9-c. What processing does the code repeat multiple times?

i. The code opens each NWB file twice — once in the preview pass (to collect metadata, trials, running/pupil values) and once in the conversion pass (to load neural data and build trial arrays). Running speed and pupil data are read in both passes.

ii.
```python
# Preview pass:
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        ...
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)

# Conversion pass:
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
```

iii. The AI justified this as intentional to avoid storing large neural matrices before global percentile/bin definitions are known.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `image_to_idx["gray"]` as a category for image identity, adding a 17th class (gray) that represents inter-stimulus intervals. This may be unnecessary if the downstream decoder only cares about the 8 actual stimulus images. The reference code does not include gray periods as a separate image identity class.

ii.
```python
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The AI included gray because "trials contain gray ISI and omission periods; using an explicit gray/blank class keeps the categorical time series defined at every bin." However, the reference uses only the non-gray stimulus images.
