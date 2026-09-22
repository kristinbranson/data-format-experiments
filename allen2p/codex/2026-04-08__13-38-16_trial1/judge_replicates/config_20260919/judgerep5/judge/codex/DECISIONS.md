# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-loads every locally available `behavior_ophys_experiment_*.nwb` file (284 in the run), using `h5py` rather than the AllenSDK cache. It makes a lightweight preview pass over all files and a second conversion pass over the 281 eligible files. Thus it covers the local subset, but treats individual experiment files as independent sessions rather than combining all planes belonging to an ophys session.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

eligible, excluded = collect_previews(files=files, ...)
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. The agent said direct HDF5 was necessary because the installed NWB stack was incompatible, and that the two-pass design avoided retaining large neural matrices while allowing global category/bin definitions. It considered local NWBs authoritative for conversion scope.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB's `/general/subject/subject_id`; a first-seen ordered mapping supplies `subjects` and the per-file `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes identify NWB subject metadata as the authoritative mouse identifier and state that the session-level mapping follows converted order.

## 1-c. How are the data split into sessions?

i. Each NWB `ophys_experiment` file is made one output session, even though its `ophys_session_id` is read. Experiments sharing an ophys session are not grouped and their planes are not merged.

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
for i, preview in enumerate(eligible, start=1):
    ...
    neural_sessions.append(neural_trials)
```

iii. Documentation calls the 281 included experiment files “sessions.” The rationale focused on processing local NWBs one at a time to cap memory, but did not justify departing from the SDK session grouping used by the reference.

## 1-d. How are the data split into trials?

i. Trials come from `/intervals/trials`. For every retained row, bins span `start_time` to `stop_time` in 100 ms steps, producing variable-duration trials on a regular trial-relative grid.

ii.
```python
trials = read_trials(f)
specs = build_trial_specs(trials)
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The agent chose the SDK-style trials table because it already contains task-defined bounds and outcomes, and chose trial start as the natural alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are removed. A retained trial must have exactly one of the four outcomes, or conversion raises an error. Unlike the reference, finite `change_time` is not required. Sessions are also removed if they lack eye tracking, have fewer than two valid trials, or have fewer than two finite pupil samples.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. Excluding aborted and auto-rewarded trials follows the prompt. The agent excluded missing-eye sessions rather than fabricating a required pupil target and required two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB `/processing/ophys/event_detection/data`, with ROI indices from `event_detection/rois` and validity from the cell specimen table. It does not use dF/F.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
```

iii. The agent preferred discrete calcium events because the analysis paper explicitly says its analyses used them, despite noting that both events and dF/F were available.

## 2-b. How is the `neural` data processed?

i. Valid-ROI event magnitudes are selected, then summed over native ophys frames falling in each 100 ms bin. Each experiment/plane remains separate.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
...
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Summation was chosen as an event-count/magnitude-preserving rebinning rule. The notes say raw events, rather than visualization-filtered events, best matched the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose indexed `valid_roi` flag is true are retained; there is no additional activity-based filtering.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The agent retained SDK ROI-validity filtering even though inspected local files generally had all ROIs marked valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial-relative bins begin at `start_time`; bin boundaries are located in the ophys timestamps with `searchsorted`, and neural frames in each interval are summed. Metadata declares `trial start`, `off_start=0`, and variable end.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
...
"temporal_alignment_event": "trial start",
```

iii. The agent viewed the task-defined trial as the natural unit and anchored all modalities to ophys timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 100 ms. Native event samples are rebinned by summation; behavioral outputs are sampled/interpolated at bin centers.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The notes argue 100 ms accommodates both roughly 11 Hz and 31 Hz recordings without pathological upsampling and still resolves 250 ms flashes.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation interval groups: `start_time`, `stop_time`, `image_name`, and `omitted`, rather than trial `initial_image_name`, `change_image_name`, and `change_time`.

ii.
```python
keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]
presentations = read_task_presentations(f)
```

iii. The agent wanted a fully time-varying, omission-aware signal reflecting actual flashes and gray inter-stimulus periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is prefixed with `gray`. Every bin center defaults to gray; centers inside a non-omitted presentation interval receive that image's code. Omissions, ISIs, and unknown names remain gray.

ii.
```python
image_values = ["gray"] + image_names
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The agent reasoned that gray/blank is needed to define identity at every bin and that omissions should not form a distinct image class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation occupancy is evaluated at the same 100 ms bin centers whose intervals define neural sums, yielding the same length as neural data.

ii.
```python
in_window = (centers >= start) & (centers < stop)
image_series[in_window] = ...
```

iii. The rationale is that all raw clocks are synchronized and all converted modalities share the trial-relative ophys-anchored grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from each stimulus presentation's `is_change`, together with its interval and `omitted` flag. Trial `go` and `change_time` are not used to construct it.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The agent regarded presentation metadata as the direct record of a true changed-image flash.

## 4-b. What processing is involved in computing `output` *Image change*?

i. It initializes a zero vector and marks all bin centers within a non-omitted change presentation as one.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
change_series[in_window] = 1
```

iii. The notes say a flash interval is more robust than a single instantaneous marker after temporal binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: presentation `is_change` becomes category 1 during that presentation and 0 otherwise; output labels are `no_change` and `change`.

ii.
```python
output_values = [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. No numerical threshold is needed because the source flag is boolean.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change presentation is projected onto the same 100 ms bin centers used for all trial outputs and corresponding neural bins.

ii.
```python
in_window = (centers >= start) & (centers < stop)
change_series[in_window] = 1
```

iii. Shared centers and synchronized absolute timestamps provide alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `/processing/running/speed/data` and its timestamps.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This is the NWB equivalent of the SDK's standard running-speed interface.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Raw values within retained trial windows are pooled globally to compute four quintile edges. For each trial, speed is linearly interpolated at 100 ms centers and digitized with those edges.

ii.
```python
running_edges = compute_bin_edges(running_pool, 5)
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Global percentiles give consistent categories and approximately balanced decoder classes; linear interpolation aligns synchronized streams.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th global percentiles define five integer bins 0–4 (`q1`–`q5`).

ii.
```python
quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, quantiles)
bins = np.digitize(values, edges, right=False)
```

iii. Equal-percentile bins directly implement the requested five equal percentile categories.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated at each trial's common 100 ms bin centers, while neural activity is summed over the corresponding bin intervals.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. The agent relies on hardware-synchronized time bases and the common trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses raw pupil area, eye-tracking timestamps, and the likely-blink vector, rather than the reference's SDK `pupil_width` field.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes describe this as processed pupil area after blink filtering converted to an equivalent circular diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN; nonnegative area is converted by `2*sqrt(area/pi)`. Finite trial-window values define global quintile edges, and diameter is interpolated at bin centers then digitized.

ii.
```python
pupil_area[likely_blink] = np.nan
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Blink removal avoids artifacts; equivalent diameter makes area interpretable as a diameter; global quantiles balance classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global quantiles at 20% increments define integer categories 0–4 (`q1`–`q5`).

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. This implements five equal-percentile bins consistently across included data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Diameter is linearly interpolated at the same 100 ms trial bin centers corresponding to neural bin intervals.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The agent cites synchronized acquisition clocks and the shared converted grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the trial table's `hit`, `miss`, `false_alarm`, and `correct_reject` flags.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. These are the SDK-defined mutually exclusive outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag must be true. Its fixed-list index (0–3) is repeated over all time bins of the trial.

ii.
```python
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Repetition keeps every output array uniformly shaped `(n_output, n_timepoints)` while preserving a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Boolean NaNs become false. Interpolation ignores nonfinite samples, extends endpoint values outside the measured span, and handles zero/one finite sample specially. Blinks become NaN. Sessions with missing/insufficient pupil data or fewer than two trials are excluded. Invalid outcome cardinality and any nonfinite converted behavior cause hard errors rather than silent repair.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
...
if finite.sum() == 0:
    return np.full(query_times.shape, np.nan, dtype=np.float32)
if finite.sum() == 1:
    return np.full(query_times.shape, float(values[finite][0]), dtype=np.float32)
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
```

iii. The agent preferred excluding sessions over inventing a required pupil target and added strict checks to expose malformed trials or failed interpolation.

## 9-a. What are the most time-consuming steps of the code?

i. The conversion pass reads full event matrices and performs trial/bin neural summation. Every eligible NWB is also opened in both preview and conversion passes. Observed conversion was roughly six seconds per large sample session.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
for spec in preview.trial_specs:
    for b in range(T):
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes identify two-pass file access and per-bin slice sums as costs, while explaining that preview avoids neural loading and session-wise conversion caps memory.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested trial/bin neural aggregation loop could use cumulative sums or a reduce-at/binning operation. The loops over presentation intervals, trial-window value collection, and per-file previews could also be partially vectorized.

ii.
```python
for spec in preview.trial_specs:
    ...
    for b in range(T):
        lo = int(frame_starts[b]); hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The agent explicitly notes that cumulative sums would accelerate event rebinning, but chose slice sums to reduce peak memory.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened and trial/running/pupil data are read once in preview and again during conversion. When plotting, trial/presentation projection and behavior interpolation are recomputed for the representative trial after conversion already performed them.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The two-pass repetition is intentional: it enables global bins without holding neural arrays in memory. Diagnostic recomputation is limited to two sessions.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preview computes/stores `ophys_session_id`, go/catch flags, `change_time`, and several presentation columns that do not affect final conversion. `build_bin_centers` returns `starts`, `ends`, and `centers`, although some callers need only centers. Optional plots recompute and render diagnostics that are not used by the decoder.

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
keep_names = [..., "is_sham_change", "trials_id", "active"]
...
starts, ends, centers = build_bin_centers(...)
```

iii. Most extra fields support inspection or future sanity checks. Plotting is explicitly optional and capped at two sessions.
