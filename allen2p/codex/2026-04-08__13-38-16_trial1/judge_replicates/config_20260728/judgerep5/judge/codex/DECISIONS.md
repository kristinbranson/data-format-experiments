# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It glob-loads every local NWB file under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, runs a lightweight preview pass over each file with `h5py`, and then reopens each eligible file during the conversion pass. Trials, stimulus presentations, running, pupil, and neural arrays are all read directly from NWB/HDF5 groups.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

...

eligible, excluded = collect_previews(
    files=files,
    bin_size_sec=BIN_SIZE_SEC,
    sample_mode=args.sample,
    required_eligible=2,
)

...

with h5py.File(preview.path, "r") as f:
    ophys_timestamps = np.asarray(
        f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
    )
    event_data, _ = load_neural_events(f)
    running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. In `CONVERSION_NOTES.md`, the AI says it used direct HDF5 reads because the installed NWB stack was incompatible in this environment, and that these reads still target the same underlying NWB fields that the Allen SDK would expose.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the NWB subject identifier. During conversion, each distinct `subject_id` becomes one entry in `subjects`, and each converted session stores the corresponding integer index in `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])

...

if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)

...

subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes describe this as the session-order subject mapping for the local NWB files; the agent treats the NWB subject metadata as the authoritative mouse identifier.

## 1-c. How are the data split into sessions?

i. Each eligible NWB experiment file is treated as one output session. The script reads `ophys_session_id` into the preview metadata, but it does not use that field to merge multiple experiments from the same behavioral session.

ii.
```python
experiment_id = int(decode_scalar(f["/identifier"][()]))
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])

...

return SessionPreview(
    path=path,
    experiment_id=experiment_id,
    ophys_session_id=ophys_session_id,
    ...
)

...

for i, preview in enumerate(eligible, start=1):
    ...
    neural_sessions.append(neural_trials)
```

iii. The notes repeatedly refer to “eligible local experiment files” as the processed session unit. The trajectory also states that passive sessions are usable and that the code processes 281 eligible experiment files, confirming that experiments were not grouped by shared `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `/intervals/trials` table. For each retained trial, the code uses the SDK-style `start_time` and `stop_time` boundaries and then rebins the trial into contiguous 100 ms bins.

ii.
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    names = [
        "id",
        "start_time",
        "stop_time",
        ...
    ]
    return read_interval_group(group, names)

...

for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. In the notes, the AI says trial logic is taken directly from the SDK/NWB trial table semantics and that the trial itself is the natural unit for the decoder, with alignment anchored to trial start.

## 1-e. How are trials filtered based on quality controls?

i. At the trial level, the AI excludes `aborted` and `auto_rewarded` trials and also requires exactly one valid outcome label among `hit`, `miss`, `false_alarm`, and `correct_reject`. At the session level, it excludes files with missing eye tracking, fewer than two valid trials, or too few finite pupil samples.

ii.
```python
for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
    outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
    if sum(int(x) for x in outcome_flags) != 1:
        raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
    specs.append(...)

...

if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The notes justify excluding no-eye-tracking sessions because pupil diameter is a required output, and they justify the aborted/auto-rewarded filtering by explicitly following the SDK trial definitions and the user instruction to keep only go and catch trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is taken from the NWB event-detection matrix, plus the event ROI index array and the ROI-validity table used for filtering.

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

iii. The notes say this was a deliberate choice: the paper’s analyses use discrete calcium events, so the agent preferred `/processing/ophys/event_detection` over dF/F while keeping SDK-style ROI filtering.

## 2-b. How is the `neural` data processed?

i. After filtering to valid ROIs, the event matrix is rebinned within each trial into 100 ms bins. Each output bin stores the sum of event-detection magnitudes from all ophys frames whose timestamps fall inside that bin.

ii.
```python
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

iii. The notes describe this as “sum event magnitudes within each 100 ms trial bin.” The agent says the rebinning was added to force one common time step across the converted dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is ROI filtering through the `valid_roi` flag. Trials with all-zero event activity are retained; they were investigated later and explicitly kept.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes justify this as mirroring the Allen SDK’s default ROI-quality filtering. They also state that all-zero event trials were checked against raw NWB data and kept because they reflected genuine sparse event detections rather than conversion errors.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start. For each trial, the code builds 100 ms bins from `start_time` to `stop_time`, then uses ophys timestamps to assign event frames into those bins.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")

...

"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. The notes explicitly say “Alignment event = trial start” and that all streams are synchronized but are projected onto a trial-relative, ophys-anchored binning scheme.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins, and yes, the native signals are temporally rebinned.

ii.
```python
BIN_SIZE_SEC = 0.1

...

"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The notes justify 100 ms as a common bin size across sessions that still resolves the stimulus flashes reasonably well, while avoiding direct dependence on the native frame rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation tables, specifically per-presentation `image_name`, `start_time`, `stop_time`, and `omitted`.

ii.
```python
keep_names = [
    "start_time",
    "stop_time",
    "image_name",
    "omitted",
    "is_change",
    "is_sham_change",
    "trials_id",
    "active",
]

...

presentations = read_task_presentations(f)
```

iii. The notes say the AI wanted stimulus-presentation-level timing rather than relying only on trial-level `initial_image_name` and `change_image_name`, so that gray periods and omissions could be represented explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code creates a global image vocabulary with an added `gray` class, initializes each trial’s image series to `gray`, and then overwrites bins whose centers fall inside non-omitted stimulus presentations with the presented image label.

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

iii. The notes justify the explicit `gray` class by saying the trial contains gray ISI periods and omissions, and that the time-varying output should stay defined at every bin.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is projected onto the exact same 100 ms trial bins used for neural activity. The same `centers` array determines both neural binning and image labels.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The notes describe this as projecting stimulus presentations onto the rebinned ophys-aligned trial axis so that image labels and neural data share a common time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table, using `is_change` together with presentation timing and omission status, rather than from the trial table’s `change_time`.

ii.
```python
keep_names = [
    "start_time",
    "stop_time",
    "image_name",
    "omitted",
    "is_change",
    "is_sham_change",
    ...
]

...

is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes justify this by saying the output should mark the changed-image presentation itself, using the presentation table as the temporally precise source of stimulus changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each trial, the code starts from an all-zero binary series and sets bins to 1 when the bin center falls inside a non-omitted presentation marked `is_change`.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
in_window = (centers >= start) & (centers < stop)
...
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes say this was intentional: the target should mark the changed-image flash interval after binning, rather than only a single instant.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary and is encoded directly as `0 = no_change` and `1 = change`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The notes treat image change as an intrinsically binary output, so no extra thresholding step beyond the binary series construction is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned exactly like image identity: it is evaluated on the same trial bin centers used for the neural data.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
        ...
    ]
)
```

iii. The notes explicitly describe image identity and image change as being projected onto the same rebinned trial axis as the neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `/processing/running/speed/data` and its associated timestamp array.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes say this directly mirrors the running signal serialized into the NWB files.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code first pools in-trial running samples across eligible files to compute global quintile cut points. During conversion, it linearly interpolates running speed to each 100 ms trial-bin center and discretizes those interpolated values with the precomputed global edges.

ii.
```python
running_concat = collect_time_window_values(running_times, running_values, specs)

...

running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)

...

running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The notes justify global percentile bins as one consistent categorical definition across the dataset, and interpolation to trial-bin centers as the way to align running to the common ophys-based time axis.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five global equal-quantile bins.

ii.
```python
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]

def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. The notes state that five equal-percentile bins were required by the task and that global edges keep the category meaning stable across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100 ms trial-bin centers used for neural activity, so every running-speed label corresponds to one neural bin.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes describe this as reindexing running onto the rebinned ophys-aligned trial axis before categorization.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking timestamps, raw pupil area, and likely-blink flags: `/acquisition/EyeTracking/eye_tracking/timestamps`, `/acquisition/EyeTracking/pupil_tracking/area_raw`, and `/acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes say the agent chose raw pupil area plus blink masking, then converted area to an equivalent diameter because pupil diameter was the requested output.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked samples are set to `NaN`, pupil area is converted to diameter with `2 * sqrt(area / pi)`, global quantile edges are estimated from pooled in-trial pupil samples, and then per-trial pupil values are linearly interpolated to 100 ms bin centers and digitized.

ii.
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=np.float64)
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)

...

pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)

...

pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)

...

pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes justify excluding blink frames and using global percentile bins, and they cite the whitepaper discussion of area and diameter as the rationale for converting area into a diameter-like quantity.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global equal-quantile bins.

ii.
```python
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]

...

pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes say the same global percentile strategy was used as for running speed so that bin meanings remain consistent across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100 ms trial-bin centers used for the neural data.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes describe this as putting pupil and neural activity on the same rebinned ophys-based trial axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)

...

outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
```

iii. The notes describe these as the SDK-defined mutually exclusive outcomes for the retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code requires exactly one of the four outcome flags to be true, converts that flag to an integer index, and then repeats that single trial outcome across every time bin in the trial.

ii.
```python
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)

...

outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. The notes say trial outcome is static per trial, but the agent repeated it across time bins to keep every output array in `(n_output, n_timepoints)` form.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing data mainly by exclusion or masking rather than by permissive fallback. Missing eye tracking excludes a whole experiment. Blinked pupil samples are set to `NaN` and then interpolated across finite neighbors. Experiments with too few finite pupil samples or fewer than two valid trials are excluded. If interpolated running or pupil values are still non-finite at conversion time, the code raises an error instead of assigning a default label. The agent also fixed one discovered metadata bug by removing an unused `omitted` class from the image vocabulary.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"

...

pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan

...

if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")

...

if str(x) not in {"", "nan", "None", "omitted"}
```

iii. The notes justify whole-session exclusion for missing eye tracking because pupil diameter is a required decoder output, and justify keeping all-zero event trials after later raw-data sanity checks confirmed they were real.

## 9-a. What are the most time-consuming steps of the code?

i. The main costs are opening and scanning every NWB file in the preview pass, reopening each eligible file in the conversion pass, and the per-trial/per-bin event rebinning loop that sums raw event frames into 100 ms bins.

ii.
```python
eligible, excluded = collect_previews(...)

...

for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)

...

for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes explicitly identify the two-pass scan and the per-bin event summation as the main runtime costs, and they contrast that with the intentional choice not to keep large neural matrices in memory across the whole dataset.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the inner per-bin loop that repeatedly slices `event_data[lo:hi]` and sums over time; the notes say a cumulative-sum approach could replace it. Additional obvious loop-heavy areas are the per-trial preview pooling and the per-presentation image projection.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)

...

for spec in trial_specs:
    lo = np.searchsorted(times, spec.start_time, side="left")
    hi = np.searchsorted(times, spec.stop_time, side="left")
    if hi > lo:
        segments.append(values[lo:hi])

...

for idx in row_idx:
    ...
    image_series[in_window] = ...
```

iii. The notes explicitly call out the per-bin event summation as a CPU-for-memory tradeoff and say cumulative sums would be faster.

## 9-c. What processing does the code repeat multiple times?

i. The script repeats a full-file pass. It opens each file once in `session_preview` and again in `convert_session`. It rereads the trial table and presentation table in both passes. It also pools raw running/pupil samples in the preview pass to compute global bin edges, then later interpolates and discretizes those streams again during actual conversion.

ii.
```python
preview = session_preview(path, bin_size_sec)

...

with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)

...

running_concat = collect_time_window_values(running_times, running_values, specs)
...
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. The notes acknowledge this directly: they describe a deliberate two-pass workflow in which preview computes eligibility, image vocabulary, and global percentile edges before the second pass builds the final arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work are not used by downstream decoding. The preview pass computes summary counts, `ophys_session_id`, `session_type`, `is_go`, `is_catch`, and pooled statistics that are mostly only used for screening or logging. The optional `plot_processing_summary` path generates diagnostic figures that do not enter the pickle. The script also reads dF/F timestamps only as a time base even though dF/F values themselves are never used, because neural activity comes from event-detection data.

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
session_type = decode_scalar(f["/general/metadata"].attrs["session_type"])

...

if show_processing:
    plot_processing_summary(...)

...

ophys_timestamps = np.asarray(
    f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
)
event_data, _ = load_neural_events(f)
```

iii. The notes frame the diagnostic plotting and preview metadata as audit/validation work rather than part of the final representation, and they explicitly say the two-pass design was chosen for memory reasons even though it repeats some discarded preprocessing.
