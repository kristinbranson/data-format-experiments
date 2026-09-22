# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every locally downloaded `behavior_ophys_experiment_*.nwb` file and reads the required HDF5 datasets directly with `h5py`. It makes a preview pass over all files, then reopens every eligible file for conversion. Thus it loads all *local experiment files*, not the full manifest release, and treats each file/imaging plane independently.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

eligible, excluded = collect_previews(files=files, ...)
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. The notes say direct HDF5 was used because the installed NWB stack was incompatible, and that the serialized NWB tables mirror AllenSDK semantics. The two-pass design avoids retaining large neural arrays while global vocabularies and quantile edges are determined.

## 1-b. How are the data split into subjects?

i. A subject is the NWB `/general/subject/subject_id`. Unique strings are registered in first-eligible-file order, and each converted experiment receives the corresponding `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The agent regarded the NWB subject identifier as the authoritative mouse identifier and checked that the full conversion contained 38 locally represented mice.

## 1-c. How are the data split into sessions?

i. Each NWB `ophys_experiment_id` file—one imaging plane—is emitted as one target session. Although `ophys_session_id` is read, it is not used to group experiments from the same simultaneous behavioral session.

ii.
```python
experiment_id = int(decode_scalar(f["/identifier"][()]))
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_sessions.append(neural_trials)
```

iii. The notes explicitly use “experiment files” as the conversion/session unit and report 281 eligible sessions from 284 NWBs. They justify direct per-file processing as memory-efficient, but do not justify failing to combine planes with the same `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from `/intervals/trials`; every retained row is segmented from its `start_time` to `stop_time`. Each interval is divided into variable-count 100 ms bins.

ii.
```python
specs.append(TrialSpec(...,
    start_time=float(trials["start_time"][idx]),
    stop_time=float(trials["stop_time"][idx]), ...))
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The agent says SDK trial definitions serialized in the NWB are authoritative, and trial start is the natural alignment event for full, variable-duration trials.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded rows are excluded. Every retained row must have exactly one of hit, miss, false alarm, or correct reject, or conversion raises an error. Trials are not filtered for a finite `change_time`, nor explicitly for `go`/`catch`. Experiments are excluded if fewer than two retained trials exist; all experiments here passed that test.

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

iii. The exclusions follow the task instruction. The notes call these “go/catch, non-aborted, non-auto-rewarded” trials, but the code relies on outcome exclusivity rather than testing `go | catch`, and does not apply the reference's finite-`change_time` condition.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `/processing/ophys/event_detection/data`, with ROI indices from `event_detection/rois` and validity from the cell-specimen table. DFF timestamps supply the ophys time axis.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
ophys_timestamps = np.asarray(f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64)
```

iii. The AI preferred detected calcium events because the analysis paper says its analyses used discrete events, despite also noting that the SDK and NWBs provide DFF.

## 2-b. How is the `neural` data processed?

i. Event magnitudes are filtered to valid ROIs, then summed over native ophys frames falling in each 100 ms trial bin. Planes are not stacked into their shared session.

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

iii. The agent says summation is appropriate for event magnitudes and selected 100 ms to provide a common rate without badly upsampling 11 Hz planes while retaining 250 ms flashes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose indexed `valid_roi` flag is true are retained. No trials with zero event activity and no additional neurons are removed.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes identify this as matching the AllenSDK default ROI-quality filter. The AI investigated 3,947 all-zero trial warnings and retained them because raw event matrices were genuinely zero.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start: absolute 100 ms bins begin at `start_time` and end at `stop_time`; ophys frame boundaries are located with `searchsorted`.

ii.
```python
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The agent chose trial start because the requested segmentation unit is a complete trial and records `off_start = 0.0`, `off_end = None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 100 ms bins. Native event samples are summed within bins; running and pupil are interpolated at bin centers; stimulus labels are evaluated at bin centers.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The stated reason is the target requirement for a common bin size across 31 Hz single-plane and roughly 11 Hz multiplane recordings. The AI considered 100 ms a compromise that avoids pathological upsampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `image_name`, `start_time`, `stop_time`, and `omitted` in every interval group resembling a stimulus-presentation table.

ii.
```python
keep_names = ["start_time", "stop_time", "image_name", "omitted", "is_change", ...]
...
presentations = read_task_presentations(f)
```

iii. The agent chose presentation rows instead of trial-level initial/change names to preserve actual flash and gray/omission timing.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is built with `gray` at index 0. Every bin defaults to gray; a non-omitted presentation assigns its image code when the bin center falls inside the presentation. Omitted flashes remain gray.

ii.
```python
image_values = ["gray"] + image_names
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI says the explicit gray class makes identity defined during ISIs and omissions. It later fixed an unused `omitted` vocabulary entry so the final vocabulary is gray plus 16 images.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are projected onto the same trial-bin centers used by the neural matrix; overlap is decided using each center.

ii.
```python
in_window = (centers >= start) & (centers < stop)
image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes state stimulus and ophys clocks are hardware-synchronized and raw-NWB spot checks reproduced converted labels exactly.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change`, `omitted`, `start_time`, and `stop_time`, rather than trial `change_time` and `go`.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The agent viewed `is_change` as the direct SDK/NWB indicator of a true changed-image presentation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero series is created and bins whose centers lie inside a non-omitted `is_change` presentation are set to one.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The stated choice marks the changed-image flash rather than an instantaneous sample, which the AI considered robust under temporal binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 means no changed-image presentation at the bin center and 1 means the center is within a true change presentation. No numerical threshold is applied.

ii.
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. The source `is_change` flag supplies the categorical threshold directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is evaluated on the identical bin-center array as all other time-varying outputs and the 100 ms neural bins.

ii.
```python
starts, ends, centers = build_bin_centers(...)
image_series, change_series = make_image_series(..., centers, ...)
```

iii. The AI cites synchronized timestamps and exact raw reconstruction checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `/processing/running/speed/timestamps` and `/processing/running/speed/data` from each experiment NWB.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes identify this as the SDK-processed wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Native running samples within retained trial windows are pooled to calculate four global quantile cut points. During conversion, speed is linearly interpolated directly at each 100 ms bin center, with constant endpoint extrapolation, then digitized.

ii.
```python
running_edges = compute_bin_edges(running_pool, 5)
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The AI chose global percentiles for consistent, approximately balanced decoder classes, and interpolation because time streams share synchronized clocks.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global 20th/40th/60th/80th percentile edges create five integer categories (`q1`–`q5`) via `np.digitize`.

ii.
```python
quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, quantiles)
bins = np.digitize(values, edges, right=False)
```

iii. Equal-percentile bins were required and provide roughly equal sample counts; global edges keep labels comparable across experiments.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated at the centers of the same 100 ms absolute-time bins used to sum neural events.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. The agent relies on shared hardware synchronization and validated selected raw/converted values.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses raw pupil ellipse area, eye-tracking timestamps, and likely-blink flags: `/acquisition/EyeTracking/pupil_tracking/area_raw`, `/eye_tracking/timestamps`, and `/likely_blink/data`.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The AI interpreted area as a processed pupil-size signal and used blink flags to reject artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN. Area is converted to an equivalent circular diameter `2*sqrt(area/pi)`. Finite values are linearly interpolated (including across blinks) at 100 ms bin centers and discretized with global quantiles. Experiments without eye tracking or enough finite pupil values are excluded.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The agent says blink masking prevents artifacts, equivalent diameter converts ellipse area into diameter units, and exclusion is preferable to fabricating a required output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global percentile edges over finite, blink-masked diameter samples form five `q1`–`q5` categories.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The same global equal-percentile rationale as running speed is used.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated at the same 100 ms bin centers as the other outputs and neural event sums.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The agent cites hardware-synchronized eye and ophys clocks and raw-data spot checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the mutually exclusive trial-table flags `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The AI calls these the SDK's canonical outcomes and enforces exactly one outcome on every retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The flag position becomes category 0–3 in the fixed listed order and is repeated across every time bin in the trial.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Repetition keeps all five outputs in a uniform `(n_output, n_timepoints)` matrix while representing a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN boolean flags become false. Interpolation ignores nonfinite samples and uses the nearest endpoint outside the measured time span; one finite value becomes constant. Missing eye tracking and insufficient pupil data exclude the experiment. Invalid outcome multiplicity and any remaining nonfinite interpolated output raise errors. Omitted images map to gray. The last bin may be shorter than 100 ms. No broad per-session exception handler is used.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The notes justify dropping three eye-less experiments because pupil is required, and prefer strict checks to silently fabricating outputs. They explicitly retained genuine zero-event trials after verification.

## 9-a. What are the most time-consuming steps of the code?

i. The conversion pass is dominant (686 s versus 65 s preview), especially loading full event matrices and performing per-trial/per-bin event summation. Pickling a 4.3 GB result is also material.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
for spec in preview.trial_specs:
    for b in range(T):
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes estimated conversion from sample timing and identified per-bin slice sums as extra CPU work; the full logs confirmed conversion was the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested trial/bin loop for neural sums could use cumulative sums or grouped reductions. Trial-window pooling and presentation-to-bin projection also loop over trials/presentations, and output construction loops over experiments.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The AI explicitly notes cumulative sums as an alternative but chose slice sums to reduce peak memory. It considered session-wise processing important for memory control.

## 9-c. What processing does the code repeat multiple times?

i. Every eligible NWB is opened twice. Trials, running, pupil, blink data, and presentation tables are read in preview and again during conversion; pupil conversion and trial parsing are repeated. Trial windows are traversed once to pool quantile data and again to build outputs.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The agent acknowledges this intentional two-pass repetition as a memory tradeoff needed to determine global categories and thresholds before constructing the large final arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Preview reads presentation tables and computes unique images, raw trial-window running/pupil pools, then discards those tables/arrays after retaining summaries. Conversion rereads trial fields such as `change_time`, `go`, and `catch` that are stored in `TrialSpec` but never used to build outputs. It reads several presentation fields (`is_sham_change`, `trials_id`, `active`) that are never used. `starts` and `widths` are partly bookkeeping, and `session_stats` is accumulated but never saved.

ii.
```python
keep_names = [..., "is_sham_change", "trials_id", "active"]
...
session_stats.append(stats)
```

iii. The notes mainly frame preview work as intentional for eligibility and global definitions. They do not justify the unused fields or unsaved `session_stats`; optional plotting work is limited to two sessions and only occurs when requested.
