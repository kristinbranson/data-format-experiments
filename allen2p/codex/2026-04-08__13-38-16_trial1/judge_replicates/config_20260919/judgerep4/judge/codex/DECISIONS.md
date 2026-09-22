# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every local `behavior_ophys_experiment_*.nwb` file and reads NWB datasets directly with `h5py`. It makes a lightweight preview pass, then reopens every eligible file for conversion. Thus “all” means the 284 locally downloaded experiment files, not every experiment listed by the Allen project metadata/cache.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
...
eligible, excluded = collect_previews(files=files, ...)
for i, preview in enumerate(eligible, start=1):
    ... = convert_session(preview=preview, ...)
```

iii. The agent says direct HDF5 was used because the installed NWB stack was incompatible, and that local files were authoritative for conversion scope. It describes a two-pass design for eligibility/global categories followed by full conversion.

## 1-b. How are the data split into subjects?

i. Each NWB's `/general/subject/subject_id` is decoded as a string. A first-seen ordered mapping assigns each converted experiment/session a `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes identify NWB subject metadata (or `mouse_id`) as the unique animal identifier and state that session indices follow conversion order.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB ophys **experiment** (one imaging plane) as an independent output session, despite also reading `ophys_session_id`. It does not group simultaneous experiments/planes sharing that ID.

ii.
```python
experiment_id = int(decode_scalar(f["/identifier"][()]))
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    ... = convert_session(preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The agent reports 281 included “sessions” from 284 experiment files. Its notes focus on actual local NWB files and do not justify failing to merge planes with a common `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from `/intervals/trials`. Each retained row uses its `start_time` through `stop_time` and is divided into variable-count 100 ms bins.

ii.
```python
trials = read_trials(f)
specs = build_trial_specs(trials)
...
starts, ends, centers = build_bin_centers(
    spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The agent chose the SDK-serialized trial table because it contains canonical task trial bounds and flags. It chose trial start as the alignment event and preserves the full, variable-duration trial.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are removed. Every retained trial must have exactly one of hit/miss/false-alarm/correct-reject; otherwise conversion raises an error. Sessions need at least two retained trials. The code does not explicitly require `go` or `catch`, or a finite `change_time`.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
...
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. The notes say this follows the instruction to retain go/catch and exclude aborted/auto-rewarded trials. Exact-one-outcome is used as an integrity check; missing-eye sessions are also excluded because pupil is required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is the NWB event-detection matrix plus its ROI index and the image-segmentation `valid_roi` mask, not dF/F.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
```

iii. The agent preferred events because the analysis paper says it analyzes discrete calcium events, while acknowledging that both events and dF/F exist.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed. Within each 100 ms bin, all raw event magnitudes whose ophys timestamps fall in the bin are summed, producing neurons × bins arrays.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
...
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes argue that events best match the paper and that summing preserves event magnitude under a common 100 ms time grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs for which `valid_roi[event_rois]` is true are retained. Released-session QC is otherwise accepted; no activity threshold or trial-level neural filter is applied.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
return event_data, event_rois[valid_mask]
```

iii. The agent cites SDK default valid-ROI filtering and release-level motion/sync/z-drift/task QC. It notes that inspected local files often already had all ROIs valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial bins begin at each trial's `start_time`; absolute bin boundaries are mapped to ophys frame indices with `searchsorted`. Metadata declares “trial start,” offset 0.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
...
"temporal_alignment_event": "trial start",
```

iii. The agent calls the trial the natural requested unit and uses synchronized absolute clocks before converting to a trial-relative grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 100 ms. Native 11/31 Hz ophys frames are rebinned by summing events within each bin; final partial bins are allowed.

ii.
```python
BIN_SIZE_SEC = 0.1
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
ends = np.minimum(starts + bin_size_sec, stop_time)
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
```

iii. The agent chose 100 ms to give all sessions one nominal bin width without pathologically upsampling 11 Hz data, while still resolving 250 ms flashes.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from all NWB stimulus-presentation interval groups containing `image_name`, using presentation start/stop times and `omitted`; it is not derived from the trial table's initial/change image fields.

ii.
```python
presentations = read_task_presentations(f)
...
start = float(presentations["start_time"][idx])
stop = float(presentations["stop_time"][idx])
omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
```

iii. The notes say presentation timing is needed to produce a genuinely time-varying image/gray series and to handle omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is prefixed with `gray`. Every 100 ms bin starts as gray; bins whose centers fall inside a non-omitted presentation receive that image's global integer code.

ii.
```python
image_values = ["gray"] + image_names
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The agent explicitly chose a gray class so ISIs, omissions, and no-image periods remain defined, and global codes stay consistent across experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are projected onto the same trial-bin centers used to delimit neural bins. A bin is labeled by whether its center falls in the presentation interval.

ii.
```python
starts, ends, centers = build_bin_centers(...)
...
in_window = (centers >= start) & (centers < stop)
image_series[in_window] = ...
```

iii. The shared synchronized absolute timestamps and common bin grid are the stated alignment rationale.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from stimulus-presentation `is_change`, `omitted`, and each presentation's start/stop times.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The agent chose presentation flags because they directly distinguish actual changed-image flashes from sham/catch events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created, overlapping presentation rows are visited, and bins during a non-omitted true-change presentation are set to one.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
for idx in row_idx:
    ...
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. The notes describe this as marking the post-change image flash interval, a robust target after binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: zero means no true changed-image presentation at that bin and one means the bin center is within one. No numeric threshold is applied.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
change_series[in_window] = 1
...
["no_change", "change"]
```

iii. The source `is_change` is boolean, so direct binary encoding is sufficient.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The same absolute presentation intervals and 100 ms trial-bin centers used for image identity are used alongside the neural bin boundaries.

ii.
```python
image_series, change_series = make_image_series(
    presentations, row_idx, centers, image_to_idx)
```

iii. The agent relies on hardware-synchronized streams and a shared converted grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB `/processing/running/speed/data` and its timestamps.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The agent identifies this as the SDK-processed running-speed signal derived from the wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Raw finite speed samples inside all retained trial windows are pooled to set global quintile edges. Speed is linearly interpolated at each 100 ms bin center, then digitized.

ii.
```python
running_edges = compute_bin_edges(running_pool, 5)
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The agent says interpolation respects synchronized time bases and global percentile bins give consistent, approximately balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global 20/40/60/80-percentile edges define five bins (`q1`–`q5`) using `np.digitize(..., right=False)`.

ii.
```python
quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, quantiles)
bins = np.digitize(values, edges, right=False)
```

iii. Global quintiles were chosen to keep category definitions consistent and class counts roughly equal.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated at the centers of the exact 100 ms bins used for neural event sums.

ii.
```python
starts, ends, centers = build_bin_centers(...)
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. The shared hardware clock makes interpolation onto those bin centers valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking timestamps, raw pupil area, and likely-blink flags from NWB.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes say pupil size comes from ellipse-derived area, with blink frames excluded.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples become NaN; nonnegative finite area is converted to an equivalent circular diameter `2*sqrt(area/pi)`. Finite trial-window samples set global quintiles, and diameter is interpolated at trial-bin centers before digitization.

ii.
```python
pupil_area[likely_blink] = np.nan
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The agent intended to remove blink artifacts and convert area into the requested diameter before global balanced discretization.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global finite-value percentile edges (20/40/60/80) define five `q1`–`q5` categories via `np.digitize`.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. As for running, global quintiles provide consistent category meanings and approximately balanced output classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Converted diameter is linearly interpolated at the same 100 ms trial-bin centers used for neural activity. Endpoint extrapolation uses the first/last valid value.

ii.
```python
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The agent cites synchronized clocks; excluding no-eye or insufficient-pupil sessions avoids undefined targets.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from trial-table `hit`, `miss`, `false_alarm`, and `correct_reject` flags.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. These are the SDK's canonical, mutually exclusive go/catch outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag must be true. Its fixed list position becomes code 0–3 and is repeated over every time bin in that trial.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Repetition keeps all five outputs in one `(n_output, n_timepoints)` matrix even though outcome is static.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Boolean NaNs become false; decoding handles bytes/strings; interpolation ignores nonfinite points and handles zero/one valid sample. Missing-eye, too-few-trial, or insufficient-pupil sessions are excluded. Blink/invalid areas become NaN. Retained-session nonfinite interpolants and malformed outcomes raise errors rather than being silently imputed. Interpolation outside coverage uses endpoint values.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
...
if finite.sum() == 0:
    return np.full(query_times.shape, np.nan, dtype=np.float32)
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The notes prefer excluding sessions over fabricating a required pupil output and use explicit sanity failures for malformed retained data.

## 9-a. What are the most time-consuming steps of the code?

i. The code reads every NWB twice (preview and conversion), loads large event matrices in the second pass, and performs nested trial × time-bin neural summation. Optional plotting adds work but is limited to two sessions.

ii.
```python
for idx, path in enumerate(files, start=1):
    preview = session_preview(path, bin_size_sec)
...
for spec in preview.trial_specs:
    ...
    for b in range(T):
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The agent explicitly designed a “lightweight” preview plus conversion pass and reports per-pass timings, but does not provide a formal profiler breakdown.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidate is the inner loop over 100 ms neural bins; reductions could be grouped using bin assignments/reduce-at. Trial-spec construction, trial-window pooling, presentation projection, preview iteration, and per-session conversion also use Python loops, though variable trial lengths make some reasonable.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes emphasize correctness and direct raw-vs-converted checks rather than claiming these loops are optimal.

## 9-c. What processing does the code repeat multiple times?

i. Each file's trials, presentations, running, and pupil data are read during preview and again during conversion. Pupil area-to-diameter conversion is also done in both passes. Trial interval searches are repeated for collecting quantile pools and final conversion.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The agent justifies the first pass as a lighter eligibility/global-statistics pass that avoids retaining full neural arrays in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Preview-only decoded presentation details and session metadata are reread later; `change_time`, `is_go`, `is_catch`, `trial_idx`, and `ophys_session_id` are stored but mostly not used in conversion. `starts` is returned by bin construction but only used to find neural boundaries, while some preview arguments (including `bin_size_sec` in `session_preview`) do not affect preview logic. Optional plots are diagnostic and not consumed by decoding.

ii.
```python
TrialSpec(... change_time=..., is_go=..., is_catch=...)
...
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
if show_processing:
    plot_processing_summary(...)
```

iii. These fields support auditing, future checks, or diagnostics, but the final decoder dataset does not use them directly.
