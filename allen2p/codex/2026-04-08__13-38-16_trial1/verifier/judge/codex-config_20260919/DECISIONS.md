# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every locally downloaded `behavior_ophys_experiment_*.nwb`, reads each directly with `h5py`, performs a metadata/behavior preview pass, then reopens every eligible file for full conversion. Thus “all” means the 284 local NWB experiment files, not every experiment listed by the Allen SDK manifest.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
...
eligible, excluded = collect_previews(files=files, ...)
for i, preview in enumerate(eligible, start=1):
    ... = convert_session(preview=preview, ...)
```

iii. The notes say direct HDF5 was used because the installed NWB stack was incompatible, and the local files were treated as authoritative for conversion scope.

## 1-b. How are the data split into subjects?

i. A subject is the string `/general/subject/subject_id`; unique IDs are registered in first-eligible-file order and each converted experiment gets its subject index.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes identify NWB subject metadata as the authoritative mouse identifier.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is emitted as one target session, even though `ophys_session_id` is read. Experiments/planes sharing an `ophys_session_id` are not merged.

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    ... = convert_session(preview=preview, ...)
    neural_sessions.append(neural_trials)
```

iii. The notes describe processing one “session” per NWB and assigning its single imaging-plane location. They do not justify the failure to group planes by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from `/intervals/trials`; each retained trial spans its `start_time` through `stop_time` and is represented by variable-count 100 ms bins.

ii.
```python
TrialSpec(... start_time=float(trials["start_time"][idx]),
          stop_time=float(trials["stop_time"][idx]), ...)
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The notes state that SDK-style trial definitions are used directly and that trial start is the natural alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are removed. Every retained trial must have exactly one of the four outcome flags. Sessions are excluded for fewer than two valid trials, missing eye tracking, or insufficient pupil samples. The code does not explicitly require go/catch or finite `change_time`.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
...
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. The notes cite the user’s explicit exclusion rule, require pupil-complete sessions, and say malformed outcomes should not be silently accepted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `/processing/ophys/event_detection/data`; dF/F timestamps provide its time axis, and event ROI IDs are related to the cell table’s `valid_roi` flags.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
ophys_timestamps = np.asarray(f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64)
```

iii. The agent chose raw event-detection magnitudes because the analysis paper used discrete calcium events, rather than the reference converter’s dF/F choice.

## 2-b. How is the `neural` data processed?

i. Event traces are transposed conceptually from frame-by-neuron into neuron-by-time output by summing all event magnitudes whose ophys timestamps fall in each 100 ms bin. No smoothing is applied.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo, hi = int(frame_starts[b]), int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes say raw rather than visualization-filtered events best match the paper, and sums preserve event mass during common-bin rebinning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose cell-table `valid_roi` flag is true are retained.

ii.
```python
valid_roi = np.asarray(f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The agent states this mirrors the AllenSDK’s default invalid-ROI exclusion even though inspected local files were already largely/all valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins begin at each trial’s `start_time`; absolute ophys timestamps select the frames in every trial-relative bin. Metadata declares `trial start`, offset 0.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
...
"temporal_alignment_event": "trial start",
```

iii. The notes justify trial start because the requested unit is a full experimental trial and outputs vary within it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at 100 ms. Native event samples are summed within bins, so explicit temporal rebinning is applied.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The agent argues 100 ms accommodates both 11 Hz and 31 Hz recordings without pathological upsampling while retaining 250 ms flash resolution and satisfying the shared-bin-size requirement.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation interval groups’ `start_time`, `stop_time`, `image_name`, and `omitted`, rather than trial `initial_image_name`/`change_image_name`.

ii.
```python
keep_names = ["start_time", "stop_time", "image_name", "omitted", ...]
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The notes say presentation records capture the actual flash/gray/omission sequence more faithfully.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted mapping is built with `gray` first. Every bin defaults to gray; a non-omitted presentation assigns its image to bins whose centers lie inside that presentation. Omitted flashes remain gray.

ii.
```python
image_values = ["gray"] + image_names
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The agent explicitly wanted a defined category during gray ISIs and omissions, unlike carrying the image identity through the entire post-change period.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The same 100 ms bin centers used to define neural bins are tested against absolute presentation intervals.

ii.
```python
starts, ends, centers = build_bin_centers(...)
in_window = (centers >= start) & (centers < stop)
```

iii. The notes rely on the hardware-synchronized timestamps and common trial bin grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change uses each stimulus presentation’s `is_change`, `omitted`, `start_time`, and `stop_time` fields.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes prefer actual presentation flags to inferential trial-level reconstruction.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and bins centered within a non-omitted true-change presentation are set to one. Sham changes and gray periods remain zero.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
change_series[in_window] = 1
```

iii. The agent says marking the changed-image flash is robust after binning and consistent with the task’s transient “right after” wording.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: `is_change and not omitted` maps to category 1 for the presentation interval; everything else maps to 0.

ii.
```python
change_series = np.zeros(...)
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. No numeric threshold is needed; the NWB boolean supplies the category.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change labels use the same bin centers and trial window as neural event sums.

ii.
```python
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The shared synchronized absolute clock and common centers are the stated alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `/processing/running/speed/data` and its timestamps.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes identify this as the SDK-processed wheel-speed series.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated at 100 ms bin centers, then digitized with global percentile edges. Preview-pass edges are computed from native samples lying inside included trial windows.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
running_edges = compute_bin_edges(running_pool, 5)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Global quantiles provide consistent, approximately balanced categories across experiments.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles of pooled finite values define five integer categories 0–4 via `np.digitize`.

ii.
```python
quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, quantiles)
bins = np.digitize(values, edges, right=False)
```

iii. The requested five equal-percentile bins motivate this global thresholding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated at the centers of the very same bins over which neural events are summed.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. The notes cite synchronized stream clocks and the common 100 ms time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses raw pupil ellipse area, eye-tracking timestamps, and `likely_blink` from `/acquisition/EyeTracking`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], ...)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], ...).astype(bool)
```

iii. The notes say the whitepaper treats pupil area/ellipse fits as the source of pupil size.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are made NaN; area is converted to equivalent circular diameter `2*sqrt(area/pi)`, linearly interpolated at bin centers, and globally percentile-binned.

ii.
```python
pupil_area[likely_blink] = np.nan
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The notes justify blink filtering and a diameter derived from area, then use the same balancing logic as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global finite-value quantiles (20/40/60/80%) yield five categories 0–4.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. This directly implements the requested five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated at the common 100 ms neural-bin centers.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The agent relies on synchronized eye/ophys clocks and excludes files without eye tracking rather than fabricating values.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The notes describe these as the canonical mutually exclusive Allen trial outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag is required; its fixed-list position (0–3) becomes the class and is repeated across every time bin of the trial.

ii.
```python
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Repetition keeps every output in a uniform `(n_output, n_timepoints)` array despite outcome being static.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN flags are treated as false; interpolation ignores nonfinite source pairs and endpoint-fills outside range; blink samples become NaN before interpolation. Sessions missing eye tracking, adequate pupil data, or two trials are excluded. Nonfinite converted behavior or malformed outcomes raise errors. Unknown/omitted images fall back to gray.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The agent preferred excluding pupil-incomplete sessions over inventing a required target and used fail-fast checks for malformed retained data.

## 9-a. What are the most time-consuming steps of the code?

i. Full conversion rereads each NWB and loads the large event matrix; neural rebinning then loops over every trial/bin and sums frame slices. The preview pass adds a second file-open/read pass.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
for spec in preview.trial_specs:
    for b in range(T):
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes explicitly identify per-bin sums as a CPU/memory tradeoff and the two-pass read as intentional to cap peak memory.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested trial/bin event-sum loop could use cumulative sums or reduce-at operations. Presentation rows are also applied one at a time, and trial-window pooling loops over trials.

ii.
```python
for b in range(T):
    ... event_data[lo:hi].sum(axis=0)
for idx in row_idx:
    ... image_series[in_window] = ...
```

iii. The notes specifically acknowledge cumulative sums as a speedup but choose slice sums to reduce peak memory.

## 9-c. What processing does the code repeat multiple times?

i. Every eligible NWB is opened in preview and conversion passes. Trials, presentations, running, and pupil are read/processed twice; plotting, if enabled, recomputes representative-trial binning/interpolation already done during conversion.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The agent documents the two-pass repetition as intentional so global categories can be learned without retaining large neural matrices.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preview accepts `bin_size_sec` but never uses it; reads `change_time`, go/catch, session IDs/types, and other presentation fields that do not affect final labels (some only diagnostics/metadata). When plots are disabled, `trials` is reread in `convert_session` but unused. Optional plotting computes and saves diagnostics not consumed by the decoder.

ii.
```python
def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    ...
trials = read_trials(f)
...
if show_processing:
    plot_processing_summary(...)
```

iii. The notes frame diagnostics as validation aids and the lightweight preview as necessary for eligibility/global thresholds, but do not identify these smaller discarded reads explicitly.
