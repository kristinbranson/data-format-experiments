# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, bypassing the AllenSDK's `VisualBehaviorOphysProjectCache`. It lists all NWB files under the experiment directory, performs a two-pass workflow: a lightweight "preview" pass to collect metadata, trial specs, running/pupil values, and image names; then a "conversion" pass to load neural data and build trial arrays. Sessions without eye tracking or with fewer than 2 valid trials are excluded during the preview pass.

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
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
```

iii. The AI chose direct h5py reads because the installed NWB/pynwb stack was incompatible with the files in the environment. The AI noted this mirrors SDK semantics already serialized into NWB. The two-pass approach avoids storing large neural matrices before global percentile/bin definitions are known.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject_id` field read from each NWB file's `/general/subject/subject_id`. Each unique subject_id becomes a subject entry.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
# ...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The subject_id is the NWB file's canonical identifier for each animal, corresponding to the SDK's mouse_id concept.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate session. There is no grouping of multiple imaging planes (experiments) within the same ophys session. Each experiment file becomes one entry in the session list.

ii.
```python
for i, preview in enumerate(eligible, start=1):
    # Each preview corresponds to one NWB experiment file
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. In the CONVERSION_NOTES, the AI notes that local NWB files are individual experiment files. Since the local data has 284 experiment files for 247 unique ophys sessions (mean 1.15 experiments per session), most sessions have only one experiment, but some multi-plane sessions would have their planes split into separate "sessions" in the converted data. The AI documented this as yielding 281 sessions (after excluding 3 without eye tracking).

## 1-d. How are the data split into trials?

i. Trials are defined using the `/intervals/trials` group in the NWB file. Non-aborted, non-auto-rewarded trials with exactly one valid outcome flag (hit/miss/false_alarm/correct_reject) are included. Trial boundaries use `start_time` to `stop_time` from the trials table.

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
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        # ...
        specs.append(TrialSpec(...))
```

iii. The AI follows the SDK trial table semantics: aborted and auto-rewarded trials are excluded as instructed. The AI adds an additional validation that each trial must have exactly one outcome flag set, raising an error otherwise.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted trials, (2) excluding auto-rewarded trials, (3) requiring exactly one valid outcome flag. Sessions are excluded if they have missing eye tracking, or fewer than 2 valid trials. The AI does not filter based on `change_time` validity (unlike the reference).

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
# Session-level exclusion:
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The AI documented that excluding sessions without eye tracking ensures all sessions have valid pupil data. The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the event-detection output at `/processing/ophys/event_detection/data`, filtered to valid ROIs using `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`.

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

iii. The AI chose event-detection outputs over dF/F because the analysis paper explicitly states that analyses were performed on discrete calcium events. The AI reasoned this is the closest match to the paper's processing.

## 2-b. How is the `neural` data processed?

i. Event-detection amplitudes are rebinned into 100ms bins by summing event magnitudes within each bin. Bins are constructed using `build_bin_centers()` which creates uniform time bins from trial start to trial stop. For each bin, ophys frames falling within that bin are identified and their event values are summed.

ii.
```python
BIN_SIZE_SEC = 0.1

def build_bin_centers(start_time, stop_time, bin_size_sec):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ends = np.minimum(starts + bin_size_sec, stop_time)
    # ...
    return starts, ends, centers

# In convert_session:
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI chose 100ms bins as a compromise: coarse enough to avoid upsampling 11 Hz recordings, fine enough to resolve 250ms stimulus flashes, and compatible with 30 Hz behavior streams. The AI noted this rebinning is required by the target format's shared bin-size constraint across sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` flag from the cell specimen table. Only neurons marked as valid ROIs are included in the converted data.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI noted this matches the AllenSDK's default behavior of filtering to `valid_roi == True` (as identified in `CellSpecimens.__init__` with `exclude_invalid_rois`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, time bins are constructed from `start_time` to `stop_time` using 100ms bins. Ophys frames falling within each bin are identified using `np.searchsorted` and their event values are summed.

ii.
```python
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
    frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
    frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
    # ...
    for b in range(T):
        lo = int(frame_starts[b])
        hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI uses trial start as the alignment event, documented in metadata as `temporal_alignment_event: "trial start"` with `off_start: 0.0` and `off_end: None` (variable trial lengths).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to a uniform 100ms (0.1s) bin size. This is a deliberate rebinning from the native ophys frame rate (which varies between 11 Hz and 31 Hz depending on single-plane vs multi-plane imaging).

ii.
```python
BIN_SIZE_SEC = 0.1
# ...
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. The AI justified this choice: "coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams." The AI also noted this is needed because the target format requires one bin size across all sessions, and native rates differ.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`/intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields. This differs from the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    # reads from /intervals/*_presentations (not /intervals/trials)
    keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]

def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        # ...
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI uses the stimulus presentations table to get precise timing of each image flash, including gray inter-stimulus intervals and omitted flashes. This provides finer temporal resolution than the trials table approach.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each 100ms time bin is assigned an image identity based on which stimulus presentation overlaps that bin's center time. Bins during gray ISI periods, omitted flashes, or outside any presentation window are labeled "gray". A global mapping from image names to integer indices is built across all eligible sessions, with "gray" as index 0.

ii.
```python
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
# ...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
for idx in row_idx:
    in_window = (centers >= start) & (centers < stop)
    if not omitted:
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI includes a "gray" class for ISI/omission periods, resulting in 17 total classes (gray + 16 task images) vs the reference's 8 classes (just image names).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers used for neural data, ensuring alignment. Both use the same `centers` array from `build_bin_centers()`.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
# Neural uses these same centers for binning
# Image identity also uses these centers:
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. Same temporal grid ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, combined with presentation timing (`start_time`, `stop_time`) and the `omitted` flag.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    change_series = np.zeros(centers.shape, dtype=np.int16)
    for idx in row_idx:
        is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
        if is_change and not omitted:
            change_series[in_window] = 1
    return image_series, change_series
```

iii. The AI uses the presentations table's `is_change` flag rather than the trials table's `go` field and `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation within a trial that has `is_change == True` and is not omitted, all time bins whose centers fall within that presentation's time window are marked as 1. The presentation window is defined by the presentation's `start_time` and `stop_time` (typically 250ms for one image flash).

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. This marks the change during the actual changed-image presentation window rather than using a fixed 750ms window from change_time as the reference does.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1), not thresholded. It is 1 during the changed-image presentation and 0 otherwise.

ii. See 4-b above.

iii. Binary by construction.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- computed at the same 100ms bin centers used for neural data.

ii. See 3-c above.

iii. Same temporal grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native timestamps to the 100ms bin centers using `np.interp`, then discretized into 5 equal percentile bins using global bin edges computed across all eligible sessions.

ii.
```python
def interpolate_series(times, values, query_times):
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)

def compute_bin_edges(values, nbins):
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values, edges):
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. The AI uses `np.interp` with `left=v[0], right=v[-1]` (extrapolation to boundary values) instead of the reference's `interp1d` with `fill_value=np.nan`. The discretization uses `np.quantile` for 4 internal edges (yielding 5 bins) vs the reference's `np.percentile` for 6 edges (including 0th and 100th percentile).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using 4 quantile-based edges (20th, 40th, 60th, 80th percentiles). Values are assigned to bins using `np.digitize` with `right=False`, then clipped to [0, 4].

ii.
```python
def compute_bin_edges(values, nbins):
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]  # [0.2, 0.4, 0.6, 0.8]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values, edges):
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. This produces approximately equal-count bins (quintiles) across all sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as neural data, ensuring alignment.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. Same temporal grid as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), with blink frames identified from `/acquisition/EyeTracking/likely_blink/data`. The area is converted to diameter via `2 * sqrt(area / pi)`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan

def pupil_area_to_diameter(area):
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)

pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The AI uses `area_raw` and converts to diameter, while the reference directly uses `pupil_width` from the SDK's eye_tracking table. The AI also sets blink frames to NaN before the area-to-diameter conversion.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is loaded, blink frames are set to NaN, area is converted to diameter via `2*sqrt(area/pi)`, then the diameter is interpolated to 100ms bin centers using `np.interp`, and finally discretized into 5 percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
# ...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The area-to-diameter conversion follows the whitepaper's description that "pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter." The AI reverse-engineers this relationship. However, the reference solution simply uses `pupil_width` directly.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 bins using 4 quantile-based edges computed globally.

ii. Same code as running speed discretization (see 5-c).

iii. Produces approximately equal-count bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers as neural data.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same temporal grid as neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table (`/intervals/trials`).

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]

hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)

outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. Same source variables as the reference. The AI adds NaN-to-zero conversion for robustness and validates exactly one outcome flag is set.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0-3) corresponding to the outcome order [hit, miss, false_alarm, correct_reject]. The code is constant across all time bins within a trial.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
# outcome_idx comes from outcome_flags.index(True)
```

iii. Same approach as reference: static per-trial value repeated across time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye tracking data (3 sessions) are excluded entirely.
- **Blink frames**: Set to NaN before pupil processing.
- **NaN outcome flags**: Converted to False via `np.nan_to_num`.
- **Inconsistent outcome flags**: Raises an error if a trial doesn't have exactly one outcome.
- **Running/pupil extrapolation**: Uses boundary values (`left=v[0], right=v[-1]`) instead of NaN.
- **Non-finite interpolation**: Raises a ValueError if running or pupil interpolation produces non-finite values.
- **Insufficient pupil data**: Sessions with fewer than 2 finite pupil samples are excluded.

ii.
```python
# Missing eye tracking
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"

# NaN handling
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)

# Extrapolation
out = np.interp(query_times, t, v, left=v[0], right=v[-1])

# Non-finite check
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
```

iii. The AI takes a stricter approach than the reference, raising errors on unexpected data conditions rather than silently handling them (e.g., the reference maps NaN to bin 0, while the AI raises an error on non-finite values).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the conversion pass which opens each NWB file a second time and loads the full neural event-detection matrix. The two-pass architecture (preview + conversion) means each file is opened twice.

ii. N/A (architectural observation)

iii. The AI documented timing: preview pass took 65s, conversion pass took 686s for the full dataset (757.7s total).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural data summation loop iterates over each time bin sequentially, computing `event_data[lo:hi].sum(axis=0)` for each bin. This could be vectorized using cumulative sums.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI noted this trades CPU efficiency for reduced peak memory. The per-presentation loop in `make_image_series` also iterates sequentially.

## 9-c. What processing does the code repeat multiple times?

i. The code opens each NWB file twice: once during the preview pass (to collect metadata, trial specs, running/pupil values) and once during the conversion pass (to load neural data and build trial arrays). Running speed and pupil data are loaded in both passes.

ii.
```python
# Preview pass:
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
        # ...

# Conversion pass:
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        running_values = np.asarray(f["/processing/running/speed/data"], ...)
```

iii. The AI documented this as intentional to avoid storing large neural matrices before global percentile definitions are computed.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and includes a "gray" image identity class, which represents inter-stimulus intervals and omitted flashes. This is an additional class not present in the reference solution. The code also reads stimulus presentations data (with detailed per-flash timing) when simpler trial-level variables would suffice for the reference approach.

ii.
```python
image_values = ["gray"] + image_names
# This adds a gray class that wouldn't be needed if using the trials table approach
```

iii. The "gray" class adds complexity and an extra output category. Whether it's "unnecessary" depends on perspective -- the AI argues it provides more accurate temporal information about what's on screen.
