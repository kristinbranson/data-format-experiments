# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every locally downloaded `behavior_ophys_experiment_*.nwb`, reads each directly with `h5py`, and makes a preview pass followed by a conversion pass. Thus “all” means the 284 local experiment files, not every experiment listed by the SDK manifest.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
...
for idx, path in enumerate(files, start=1):
    preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
```

iii. The notes say direct HDF5 was used because the installed NWB stack was incompatible, and call the local downloaded files authoritative for conversion scope. Two passes avoid retaining large neural matrices while global categories and quantiles are computed.

## 1-b. How are the data split into subjects?

i. The NWB subject ID is read per experiment; unique IDs are accumulated in encounter order and each converted experiment gets the corresponding `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes identify NWB subject metadata/mouse ID as the animal identifier and state that indices follow converted-session order.

## 1-c. How are the data split into sessions?

i. Each NWB ophys experiment file is treated as one output session. Although `ophys_session_id` is read, it is metadata only; experiments/imaging planes sharing that ID are not combined.

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
```

iii. The README explicitly reports “281 ophys experiment files” as sessions. The notes focus on processing local files independently and never justify not grouping simultaneous planes by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from `/intervals/trials`; each retained table row is a trial spanning its `start_time` to `stop_time`, represented on a trial-relative 100 ms grid.

ii.
```python
trials = read_trials(f)
specs = build_trial_specs(trials)
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The agent says this mirrors AllenSDK trial semantics and that trial start is the natural alignment event for the requested trial-based decoder.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded rows are removed. Every retained row must have exactly one of hit/miss/false-alarm/correct-reject. Sessions are excluded if they lack eye tracking, have fewer than two retained trials, or have fewer than two finite pupil samples.

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

iii. Excluding aborted/auto-rewarded trials is directly required. Missing-eye sessions are dropped rather than fabricating a required pupil target; the notes report three such files.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB event-detection magnitudes and their ROI IDs, plus the image-segmentation `valid_roi` mask—not from dF/F.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
```

iii. The agent preferred events because the analysis paper says its analyses used discrete calcium events, while preserving SDK ROI filtering.

## 2-b. How is the `neural` data processed?

i. Event columns are restricted to valid ROIs. For each 100 ms trial bin, event magnitudes at ophys timestamps within the bin are summed, yielding neurons by bins.

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

iii. The notes argue event data best matches the paper and that sums preserve event magnitude under rebinning. A 100 ms common grid was chosen to accommodate 11/31 Hz recordings while resolving 250 ms flashes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose indexed `valid_roi` flag is true are retained; no further neuron-level threshold is applied.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes say this mirrors the AllenSDK load path and dataset QC; they also observed that many local files already had all ROI rows marked valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bins begin at each trial’s `start_time`; ophys timestamps locate all native event frames in each trial-relative bin. Metadata names “trial start” as the alignment event.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
...
"temporal_alignment_event": "trial start",
```

iii. The agent considered the complete trial the requested unit and used the synchronized ophys clock as the common absolute time base.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 100 ms. Native event frames are rebinned by summing within bins; behavioral signals are interpolated at bin centers.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The notes justify 100 ms as common across different native frame rates, not pathological upsampling of 11 Hz data, and still finer than the 250 ms image flash.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation interval tables: `start_time`, `stop_time`, `image_name`, and `omitted`.

ii.
```python
keep_names = ["start_time", "stop_time", "image_name", "omitted", ...]
...
presentations = read_task_presentations(f)
```

iii. The notes say presentation timing is preferable for a genuinely time-varying output and permits explicit handling of omissions and gray inter-stimulus periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is prefixed with `gray`. Every bin starts as gray, then bins whose centers fall inside a non-omitted presentation receive that image’s integer code; omissions remain gray.

ii.
```python
image_values = ["gray"] + image_names
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The agent states that gray/blank is needed to define the category during ISIs and omissions and that one global mapping maintains consistent labels.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels and neural activity share the same 100 ms bin centers/edges. Presentation membership is evaluated at each center.

ii.
```python
in_window = (centers >= start) & (centers < stop)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The notes emphasize that stimulus and ophys timestamps are hardware-synchronized and projected onto the common trial grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from stimulus-presentation `is_change`, `omitted`, `start_time`, and `stop_time`.

ii.
```python
keep_names = [..., "omitted", "is_change", "is_sham_change", ...]
...
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
```

iii. The notes say presentation flags directly identify a true changed-image flash and allow sham/catch events to remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and bins inside a non-omitted presentation flagged `is_change` are set to one.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The agent chose the duration of the changed-image presentation rather than a single instant because it is robust to temporal binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is inherently binary: 0=`no_change`, 1=`change`; there is no numeric threshold beyond boolean conversion of `is_change`.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
...
["no_change", "change"],
```

iii. The source flag already supplies the requested binary categorization.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is calculated on the identical 100 ms bin centers used for neural data, turning on where centers overlap the changed presentation.

ii.
```python
in_window = (centers >= start) & (centers < stop)
change_series[in_window] = 1
```

iii. The common synchronized trial grid is the stated alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `/processing/running/speed/data` and its timestamps.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The agent identifies this as the SDK-processed wheel speed stream rather than recomputing speed from encoder voltage.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Native speed is linearly interpolated at 100 ms bin centers. Four global quantile edges, calculated from finite native speed samples inside retained trial windows, convert values to five bins.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
...
running_edges = compute_bin_edges(running_pool, 5)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Global percentiles give consistent labels and approximately balanced classes. Interpolation is justified by synchronized clocks.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles form global boundaries; `np.digitize` produces labels 0–4 (`q1`–`q5`).

ii.
```python
quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, quantiles)
...
bins = np.digitize(values, edges, right=False)
```

iii. The requested “five equal percentile bins” is interpreted globally across included data.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated at the same bin centers that define the neural sum bins.

ii.
```python
starts, ends, centers = build_bin_centers(...)
...
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. Hardware-synchronized timestamps and the shared trial grid are the alignment rationale.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses raw tracked pupil area, eye timestamps, and `likely_blink`; diameter is not read directly.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes cite the paper/whitepaper’s ellipse-based pupil size and choose an equivalent diameter from area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples become NaN; nonnegative finite area becomes equivalent-circle diameter `2*sqrt(area/pi)`. This is interpolated at bin centers and discretized by global percentile edges computed from finite native trial-window values.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Blink removal prevents artifacts; global percentile categories parallel running speed. Sessions without enough pupil data are excluded.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global 20/40/60/80th percentile edges and `np.digitize` yield labels 0–4 (`q1`–`q5`).

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. This implements five approximately equal-frequency categories consistently across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Blink-cleaned derived diameter is linearly interpolated at the shared 100 ms neural-bin centers.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The agent relies on hardware synchronization of eye and ophys timestamp streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. These are the SDK’s mutually exclusive canonical outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag is required. Its fixed list position gives code 0–3, which is repeated across all time bins in that trial.

ii.
```python
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Repetition keeps all five outputs in a uniform `(n_output, n_timepoints)` array while preserving a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Boolean NaNs become false; blinks become NaN before pupil interpolation; missing-eye, too-few-trial, and insufficient-pupil sessions are excluded. Interpolation sorts finite samples and extends endpoint values outside their range. Nonfinite converted behavioral values cause an error; malformed outcome rows also raise, rather than being repaired.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
...
finite = np.isfinite(times) & np.isfinite(values)
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The agent chose exclusion over fabricating a required pupil target and treated structural outcome inconsistencies as hard errors. Endpoint filling is intended to keep interpolated target arrays finite.

## 9-a. What are the most time-consuming steps of the code?

i. The conversion pass dominates: full event matrices are read and every trial’s events are summed bin by bin. The measured full preview took 65 s and conversion took 686 s.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
...
for spec in preview.trial_specs:
    ...
    for b in range(T):
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes explicitly identify per-bin slice sums as a CPU cost and report conversion as the dominant timed stage.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested per-trial/per-bin neural summation could use cumulative sums or a grouped reduction. Trial-window value collection and presentation projection also loop over trials/presentations, though they are smaller costs.

ii.
```python
for spec in preview.trial_specs:
    ...
    for b in range(T):
        ... event_data[lo:hi].sum(axis=0)
...
for idx in row_idx:
```

iii. The agent acknowledges that cumulative sums would accelerate neural rebinning but chose lower peak memory.

## 9-c. What processing does the code repeat multiple times?

i. Every eligible file is opened in the preview pass and reopened in conversion. Trials, presentations, running, pupil area/blinks, and derived pupil diameter are read/constructed in both passes.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The notes call this intentional: the first pass establishes inclusion and global edges without retaining large neural arrays, capping peak memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preview reads and concatenates native running/pupil samples solely to calculate global edges, then rereads those streams. It also reads `ophys_session_id`, `is_sham_change`, `trials_id`, `active`, trial `change_time`, `is_go`, and `is_catch` without using most of them in final conversion logic. Optional plotting recomputes one representative trial’s outputs.

ii.
```python
running_concat = collect_time_window_values(...)
pupil_values_all = collect_time_window_values(...)
...
keep_names = [..., "is_sham_change", "trials_id", "active"]
```

iii. The agent frames the duplicated preview work as a memory/runtime tradeoff; diagnostic fields and plotting support validation, but their results are not part of downstream decoder data.
