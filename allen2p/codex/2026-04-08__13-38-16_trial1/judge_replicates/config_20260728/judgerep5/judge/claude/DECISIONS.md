# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, bypassing the AllenSDK Python API. It discovers all experiment files by globbing `behavior_ophys_experiment_*.nwb` under the data directory. A two-pass workflow is used: first a lightweight "preview" pass to collect metadata and compute global bin edges, then a "conversion" pass that reads full neural data and builds trial arrays.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

# In session_preview:
with h5py.File(path, "r") as f:
    experiment_id = int(decode_scalar(f["/identifier"][()]))
    ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
    ...
    trials = read_trials(f)
    specs = build_trial_specs(trials)
```

iii. The AI noted that `pynwb` was incompatible in the runtime environment, so it read NWB/HDF5 files directly with `h5py`. The AI's CONVERSION_NOTES.md states: "Uses direct HDF5 reads from local NWB files rather than pynwb because the installed NWB stack is incompatible with these files in this environment." It also used a `DATALIMIT_SUBSET.csv` check mechanism (not present in the AI code but present in the reference).

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject_id` field in each NWB file's `/general/subject/subject_id`. Unique subject IDs are collected as sessions are processed; each session is tagged with a subject index.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The AI used the NWB metadata field directly to identify subjects, consistent with the SDK's `mouse_id` field.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (one imaging plane) is treated as a separate "session" in the output. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session. Each experiment file becomes one entry in the session lists.

ii.
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The AI's CONVERSION_NOTES.md Step 5 describes brain regions: "One region index repeated for every neuron in a session" and the code treats each experiment file as a session unit. This is a key structural difference from the reference, which groups experiments by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the `/intervals/trials` table in each NWB file. Aborted and auto-rewarded trials are excluded. The remaining go and catch trials are kept. Trial windows span from `start_time` to `stop_time`, then rebinned into 100ms bins.

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

# Bin construction per trial:
def build_bin_centers(start_time, stop_time, bin_size_sec):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...
```

iii. The AI followed the instruction to exclude aborted and auto-rewarded trials. It additionally requires exactly one valid outcome flag per trial, raising an error otherwise. The trial window uses the full `start_time` to `stop_time` range as defined in the trials table.

## 1-e. How are trials filtered based on quality controls?

i. Three filtering criteria: (1) aborted and auto-rewarded trials are excluded, (2) trials must have exactly one valid outcome flag (hit/miss/false_alarm/correct_reject), and (3) sessions without eye tracking are excluded entirely. Sessions with fewer than 2 eligible trials after preview are also excluded.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)

# Session-level exclusion:
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. The AI's CONVERSION_NOTES.md explains: "Exclude sessions with missing eye tracking: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data." The requirement for exactly one outcome prevents ambiguous trial states.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the **event detection** output (`/processing/ophys/event_detection/data`), NOT from dF/F traces. Only neurons with `valid_roi == True` are included.

ii.
```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
    return event_data, event_rois[valid_mask]
```

iii. The AI justified this in CONVERSION_NOTES.md Step 5: "Neural signal = event-detection output, not dF/F: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events."

## 2-b. How is the `neural` data processed?

i. Event detection data is rebinned from the native ophys frame rate into 100ms bins by summing event magnitudes within each bin. Only valid ROIs are included.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI chose 100ms bins as a compromise: "coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` flag from the cell specimen table. Only valid ROIs are retained.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI notes in CONVERSION_NOTES.md Step 4: "SDK filters to valid_roi by default" and "Keep SDK valid_roi filtering logic even if many local files already appear fully valid."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. Bins are constructed from `start_time` to `stop_time` in 100ms steps. For each bin, ophys frames falling within that bin's time range are summed.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The AI aligned to trial start with `off_start = 0.0` and variable `off_end`. The metadata records `temporal_alignment_event: "trial start"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to **100ms** bins. This is different from the native ophys frame rate (~11 Hz for multiplane, ~31 Hz for single-plane). Event magnitudes are summed within each bin.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,  # 100.0 ms
```

iii. The AI justified the 100ms bin size in CONVERSION_NOTES.md Step 5: "Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions. [...] Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** (`/intervals/*_presentations`), using the `image_name`, `start_time`, `stop_time`, and `omitted` fields. It is NOT derived from the trials table `initial_image_name`/`change_image_name`.

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

iii. The AI chose the presentations table because it provides per-flash timing, allowing precise tracking of when images are on screen vs. gray inter-stimulus intervals. The CONVERSION_NOTES.md Step 5 maps: "Stimulus presentation image_name + presentation timing + omission state -> output[0] (image_identity)."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping. The mapping includes a `gray` class (index 0) for inter-stimulus intervals and omitted flashes. During each trial, bins are initialized to `gray`, then each non-omitted stimulus presentation assigns its image code to the bins it overlaps.

ii.
```python
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
for idx in row_idx:
    ...
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI included `gray` as an explicit class to handle ISI periods. This results in 17 total image classes (gray + 16 task images in the full dataset).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers as the neural data. Each bin center is checked against stimulus presentation windows.

ii.
```python
in_window = (centers >= start) & (centers < stop)
```

iii. Using the same bin centers ensures alignment with the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, combined with presentation `start_time` and `stop_time`.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI used the presentations-level `is_change` flag rather than the trial-level `go` flag combined with `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary series initialized to 0. For each presentation in the trial window, if `is_change` is True and not omitted, the bins overlapping that presentation are set to 1.

ii. Same as 4-a code snippet.

iii. The change indicator is 1 only during the changed-image flash, not during subsequent flashes.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100ms bin centers as neural data. Change is marked during the presentation window of the changed stimulus.

ii. Same as 4-a.

iii. Same bin-center alignment as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/data` and `/processing/running/speed/timestamps` in the NWB files.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The running speed data path is consistent with what the AllenSDK exposes.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 100ms bin centers, then discretized into 5 percentile-based bins. Bin edges are computed globally across all eligible sessions. Interpolation uses `np.interp` with edge-value fill (no NaN).

ii.
```python
def interpolate_series(times, values, query_times):
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
...
running_edges = compute_bin_edges(running_pool, 5)
...
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The AI computed global percentile edges from all eligible sessions' trial windows during the preview pass, then applied them during conversion. The interpolation uses `np.interp` which fills extrapolated values with the first/last known value rather than NaN.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins using 4 inner quantile edges. `np.digitize` with `right=False` is used, then clipped to [0, 4].

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

iii. The 4 inner quantile edges split data into 5 roughly equal bins by construction.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers used for neural data.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. Same bin-center alignment as all other data streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area_raw` (`/acquisition/EyeTracking/pupil_tracking/area_raw`) and the `likely_blink` flag (`/acquisition/EyeTracking/likely_blink/data`). Eye tracking timestamps come from `/acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The AI used `area_raw` and converted to diameter via `2*sqrt(area/pi)`. This differs from the reference which uses `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then pupil area is converted to diameter using `2*sqrt(area/pi)`. The resulting diameter is interpolated to 100ms bin centers using `np.interp`, then discretized into 5 global percentile bins.

ii.
```python
def pupil_area_to_diameter(area):
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The AI noted in CONVERSION_NOTES.md Step 3: "whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter." The AI reversed this by converting area back to diameter.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile-based bins using global quantile edges.

ii. Same discretization functions as running speed.

iii. Same approach ensures uniform bin populations.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers used for neural data.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same alignment strategy as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

iii. The AI requires exactly one outcome to be True, raising an error otherwise. The outcome index maps to `["hit", "miss", "false_alarm", "correct_reject"]`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome index (0-3) is repeated across all time bins in the trial to create a time-varying series with constant value.

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

iii. The AI justified repeating the value: "Trial outcome will be repeated across time bins: Although static per-trial, repeating it across the trial keeps every output array in (n_output, n_timepoints) form."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: 3 sessions excluded entirely.
- **Ambiguous outcome**: Trials without exactly one valid outcome flag raise an error.
- **Empty bins at trial edges**: The bin construction ensures at least one bin per trial.
- **Blink frames**: Pupil area set to NaN, then interpolated over.
- **Extrapolation**: `np.interp` fills with edge values; the AI raises a ValueError if non-finite values remain after interpolation of running or pupil data.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
pupil_area[likely_blink] = np.nan
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The AI chose strict error handling for non-finite values and excluded sessions without eye tracking rather than filling with defaults.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are (1) the preview pass which opens every NWB file to collect metadata (~65s), and (2) the conversion pass which reopens each file and reads neural data (~686s). Both are I/O bound from reading large HDF5 files.

ii. N/A (timing data from conversion_full_out.txt)

iii. The two-pass architecture means each NWB file is opened twice: once for preview and once for conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural rebinning loop iterates over each time bin individually to sum event magnitudes:

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. This could be vectorized using cumulative sums. The AI acknowledged this in CONVERSION_NOTES.md: "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once during the preview pass (to collect metadata, trial specs, and running/pupil values for global bin edge computation) and once during the conversion pass (to read neural data and build trial arrays).

ii.
```python
# Preview pass:
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        ...

# Conversion pass:
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        ...
```

iii. The AI noted this: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preview pass computes trial specs and collects running/pupil values for global bin edge computation. These are computed from raw timestamps and values, then the conversion pass recomputes interpolated values from scratch rather than reusing the preview data. Additionally, the two-pass architecture reads trial tables, running data, and pupil data twice.

ii. N/A

iii. The AI prioritized memory efficiency over avoiding redundant computation.
