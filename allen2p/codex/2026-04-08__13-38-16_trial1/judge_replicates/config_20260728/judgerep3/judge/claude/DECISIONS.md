# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, bypassing the AllenSDK Python API. It lists all NWB files under the experiment directory, performs a two-pass workflow: first a "preview" pass to discover eligible sessions and compute global statistics, then a "conversion" pass to build the final arrays.

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

iii. The AI noted that `pynwb` failed in the environment, so it fell back to direct `h5py` reads of the same NWB fields the AllenSDK would use. This is documented in CONVERSION_NOTES.md Step 6.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `subject_id` read from `/general/subject/subject_id` in each NWB file. Unique subject IDs are collected as sessions are processed.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The AI uses the NWB subject metadata field, which corresponds to the SDK's `mouse_id`. Subjects are accumulated as sessions are encountered.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one session. The AI does NOT group multiple experiments (imaging planes) from the same `ophys_session_id` into a single session. Each experiment file = one session in the output.

ii.
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
    ...
```

iii. The AI's CONVERSION_NOTES.md notes that one experiment corresponds to one imaging plane in one session, and that multi-plane sessions have multiple experiments. However, the code treats each experiment as a separate session rather than grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined from the `/intervals/trials` group in the NWB file. Non-aborted, non-auto-rewarded trials are included (go and catch). Trial boundaries use `start_time` and `stop_time` from the trials table. Trials are then rebinned into 100ms time bins.

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

def build_bin_centers(start_time, stop_time, bin_size_sec):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...
```

iii. Trials are segmented using the SDK's trial table boundaries. Each trial uses its full `start_time` to `stop_time` window, rebinned into 100ms bins.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Each valid trial must have exactly one of hit/miss/false_alarm/correct_reject. Sessions missing eye tracking are excluded entirely. Sessions with fewer than 2 valid trials are excluded. Sessions with insufficient valid pupil samples are excluded.

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

iii. The AI documented that 3 sessions were excluded due to missing eye tracking. No sessions were excluded for other reasons. The AI does NOT filter on `change_time.notna()` unlike the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `/processing/ophys/event_detection/data`, the event-detection output, not from dF/F traces. Only neurons with `valid_roi == True` are included.

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

iii. The AI justified using event-detection data because "the paper explicitly states that analyses were performed on discrete calcium events" (CONVERSION_NOTES.md Steps 4 and 5).

## 2-b. How is the `neural` data processed?

i. Event-detection magnitudes are summed within each 100ms time bin to produce per-trial neural activity matrices. For each bin, all ophys frames falling within that bin's time window are summed.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI chose to sum event magnitudes within bins. This is a form of temporal rebinning. The AI documented this choice in CONVERSION_NOTES.md as "sum event magnitudes within each 100 ms trial bin."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` flag from the cell specimen table. Only neurons marked as valid ROIs are included.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI noted this matches the SDK's default behavior of filtering invalid ROIs at load time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. The trial window from `start_time` to `stop_time` is divided into 100ms bins, and ophys frames are mapped to these bins.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The AI uses `trial start` as the alignment event, documented in metadata as `temporal_alignment_event: "trial start"` with `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes, temporal rebinning is applied. All data is rebinned to 100ms time bins. This is different from the native ophys frame rate (~11 Hz for multi-plane or ~31 Hz for single-plane).

ii.
```python
BIN_SIZE_SEC = 0.1
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. The AI justified rebinning: "This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams" (CONVERSION_NOTES.md Step 5).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`/intervals/*_presentations`), specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields. This is NOT from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    ...
    keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]
    ...

def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    for idx in row_idx:
        ...
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. Using the presentations table gives a more accurate time-varying image identity that accounts for individual flash timing, gray inter-stimulus intervals, and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, presentation rows overlapping the trial window are found. For each 100ms time bin, the image shown during that bin is determined from presentation start/stop times. During ISI, omissions, or no-image periods, the value is set to "gray". Image names are mapped to integer codes via a global sorted mapping that includes "gray" as index 0.

ii.
```python
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
for idx in row_idx:
    ...
    omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The "gray" class captures ISI and omission periods. The global mapping ensures consistent codes across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers as the neural data. Presentations are projected onto the bin centers, so each bin's image identity corresponds to the stimulus shown at that bin's center time.

ii.
```python
in_window = (centers >= start) & (centers < stop)
```

iii. Since both neural and image identity use the same `centers` array, they are aligned by construction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, along with presentation timing (`start_time`, `stop_time`) and the `omitted` flag.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI uses the presentations table's `is_change` flag rather than the trial-level `change_time` and `go` flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each presentation row that overlaps the trial, if `is_change` is True and the flash is not omitted, the corresponding time bins are set to 1. Otherwise they remain 0.

ii. Same as 4-a.

iii. The change indicator marks the duration of the change-image flash (typically 250ms) rather than a fixed 750ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary: 0 = no_change, 1 = change. No thresholding is needed beyond the binary flag from presentations.

ii. N/A (binary variable).

iii. N/A.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity -- computed at the same 100ms bin centers.

ii. See 3-c and 4-a code.

iii. Aligned by construction via shared `centers` array.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This is the same running speed variable used by the AllenSDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to the 100ms bin centers using linear interpolation (`np.interp`), with edge values extended (left=first value, right=last value). Then discretized into 5 equal percentile bins using globally computed bin edges.

ii.
```python
def interpolate_series(times, values, query_times):
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    ...

running_edges = compute_bin_edges(running_pool, 5)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Global percentile-based bin edges ensure consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using quantile-based edges computed across all eligible sessions. The edges are computed using `np.quantile` at `[0.2, 0.4, 0.6, 0.8]` quantiles, then `np.digitize` assigns bin indices.

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

iii. This produces 5 bins (indices 0-4) with approximately equal population.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as the neural data.

ii. See 5-b.

iii. Aligned by construction via shared `centers` array.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw` (pupil area), the likely blink flag from `/acquisition/EyeTracking/likely_blink/data`, and eye tracking timestamps from `/acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The AI uses raw pupil area rather than `pupil_width`, converting area to diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN. Pupil area is converted to equivalent diameter using `2 * sqrt(area / pi)`. The resulting diameter is interpolated to 100ms bin centers and discretized into 5 percentile bins.

ii.
```python
def pupil_area_to_diameter(area):
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    ...
```

iii. The AI follows the whitepaper's description that "pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter." The AI works backwards from area to diameter.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 quantile-based bins using globally computed edges.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Global bin edges across all sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same alignment as running speed -- interpolated to the same 100ms bin centers.

ii. See interpolation in `convert_session()`.

iii. Aligned by construction.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table (`/intervals/trials`).

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
...
outcome_idx = outcome_flags.index(True)
```

iii. The four outcome columns are mutually exclusive for valid (non-aborted, non-auto-rewarded) trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is replicated across all time bins in the trial.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. The outcome is static per-trial but repeated across all time bins to maintain uniform output shape `(n_output, n_timepoints)`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- Sessions missing eye tracking (3 sessions) are excluded entirely.
- Sessions with fewer than 2 valid trials are excluded.
- Sessions with insufficient valid pupil samples are excluded.
- Trials where outcome flags don't sum to exactly 1 raise a ValueError.
- Running speed interpolation uses edge-value extension (left=first, right=last) rather than NaN for out-of-range values.
- Pupil values during blinks are set to NaN before area-to-diameter conversion.
- The AI raises an error if non-finite running or pupil values appear after interpolation.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
...
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
```

iii. The AI takes a strict approach, raising errors for unexpected data issues rather than silently handling them.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the conversion pass where each NWB file is opened and read with h5py. The preview pass takes ~65s and the conversion pass takes ~686s for 281 sessions (total ~758s). Per-session conversion averages ~2-3 seconds.

ii. N/A (timing from conversion_full_out.txt).

iii. The two-pass design (preview then conversion) means each NWB file is opened twice, which doubles I/O cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural data summation loop iterates over each time bin sequentially:
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```
This could be vectorized using cumulative sums.

ii. See above.

iii. The AI's CONVERSION_NOTES.md acknowledges this: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once during the preview pass (to collect metadata, trial specs, running/pupil values for global bin edge computation) and once during the conversion pass (to load neural data and build trial arrays).

ii.
```python
# Preview pass
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        ...

# Conversion pass
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        ...
```

iii. The AI noted this is intentional to avoid storing large neural matrices before global bin edges are known.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preview pass reads and processes running speed and pupil data for all sessions to compute global bin edges. This data is then re-read and re-processed during the conversion pass. Additionally, the `make_image_series` function iterates over all presentation rows for each trial even when most don't overlap the trial window (though this is mitigated by the pre-filtering via `find_presentation_rows`).

ii. N/A

iii. The two-pass architecture trades efficiency for lower peak memory usage.
