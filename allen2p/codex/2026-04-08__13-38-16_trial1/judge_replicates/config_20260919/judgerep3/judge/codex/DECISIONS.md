# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every local `behavior_ophys_experiment_*.nwb` file, sorts the paths, and opens each twice with `h5py`: a preview pass reads metadata/trials/behavior, and a conversion pass reads the full selected streams. Full mode uses all locally present files; sample mode stops after two eligible files.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
...
eligible, excluded = collect_previews(files=files, ...)
...
with h5py.File(preview.path, "r") as f:
```

iii. The notes say direct HDF5 was used because the installed NWB stack was incompatible, while preserving SDK semantics serialized in NWB. The two-pass design permits global category/bin calculation without retaining large neural matrices. The local NWBs, rather than the larger manifest, were treated as the available conversion scope.

## 1-b. How are the data split into subjects?

i. Subject IDs come from each NWB's `/general/subject/subject_id`; unique strings are accumulated in first-encounter order and each converted file receives the corresponding `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes identify NWB subject metadata (equivalent to mouse ID) as the authoritative unique animal identifier.

## 1-c. How are the data split into sessions?

i. Each individual NWB ophys experiment is emitted as one target session. Although `ophys_session_id` is read, files sharing it are not grouped, so simultaneous imaging planes remain separate sessions.

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
```

iii. The agent describes the local units as “NWB experiment files” and assigns one region per file. It does not justify why the behavioral-session identifier is not used to combine planes.

## 1-d. How are the data split into trials?

i. Trials are read from `/intervals/trials`; each retained row defines a variable-length window from `start_time` to `stop_time`. A uniform trial-relative grid is constructed for that interval.

ii.
```python
trials = read_trials(f)
specs = build_trial_specs(trials)
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The notes state that the SDK trial table is used directly because it contains the canonical go/catch timing, flags, and outcomes. Trial start was chosen as the natural alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded rows are excluded. Every retained row must have exactly one of hit, miss, false alarm, or correct reject, otherwise conversion raises an error. Sessions/files are excluded if they lack eye tracking, have fewer than two retained trials, or have fewer than two finite pupil samples. The code does not explicitly require `go` or `catch`, nor a finite `change_time`.

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

iii. Excluding aborted and auto-rewarded trials follows the task. Missing-eye sessions are dropped rather than inventing a required pupil output, and the two-trial minimum follows validator requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from `/processing/ophys/event_detection/data`; its ROI IDs are matched to `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
```

iii. The paper explicitly says its analyses used discrete calcium events, so the agent preferred events over the also-available dF/F traces and retained the SDK-style ROI validity filter.

## 2-b. How is the `neural` data processed?

i. Event arrays are transposed conceptually from time-by-neuron into neuron-by-time trial matrices. For every 100 ms interval, all native-frame event magnitudes are summed. No normalization is applied.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Summing is appropriate for event magnitudes and the common grid handles differing native frame rates. Notes acknowledge cumulative sums could make this faster.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose `valid_roi` entry is true are retained; no further cell/session signal-quality threshold is applied in the converter.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes say this mirrors AllenSDK loading and dataset-release QC, even though inspected files often already mark every ROI valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each grid is anchored at the trial's `start_time`; ophys timestamps select frames in each trial-relative bin through `searchsorted`. Metadata names “trial start” as the event.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
...
"temporal_alignment_event": "trial start",
```

iii. The agent considered the trial start the natural event for a trial-segmented dataset and used synchronized ophys timestamps as required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 100 ms. Native event frames are rebinned by summation, and behavioral/stimulus values are evaluated on bin centers.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
```

iii. The notes argue 100 ms avoids pathological upsampling across 11/31 Hz recordings while retaining resolution for 250 ms flashes and ensuring one bin size across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `image_name`, `start_time`, `stop_time`, and `omitted`, read from presentation interval groups.

ii.
```python
keep_names = ["start_time", "stop_time", "image_name", "omitted", ...]
...
omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
```

iii. The agent chose actual presentation intervals rather than trial-level initial/change names to represent the time-varying flashed stimulus and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is prefixed with `gray`. Every bin begins as gray, then bins whose centers lie inside a non-omitted presentation receive that image's integer code. Omitted flashes remain gray.

ii.
```python
image_values = ["gray"] + image_names
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes say an explicit gray/blank class keeps identity defined during inter-stimulus and omission periods and global codes consistent.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image presentation intervals are projected onto the same 100 ms bin centers used for neural bins.

ii.
```python
in_window = (centers >= start) & (centers < stop)
image_series[in_window] = ...
```

iii. All streams share absolute synchronized clocks, and the common centers produce equal-length neural/output arrays.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses presentation-level `is_change`, `omitted`, `start_time`, and `stop_time`.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The presentation flag directly distinguishes a true changed image from repeats and sham/catch changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and set to one for grid centers inside a non-omitted presentation marked `is_change`.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
change_series[in_window] = 1
```

iii. The changed-image flash interval was judged more robust after temporal binning than a one-bin impulse.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: false/no interval is category 0 and a true changed-image presentation is category 1; no numeric threshold is estimated.

ii.
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. This follows the requested binary target and raw boolean change flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The same trial grid centers define membership in changed-presentation intervals.

ii.
```python
in_window = (centers >= start) & (centers < stop)
change_series[in_window] = 1
```

iii. Shared synchronized absolute times and identical grid length align the output to neural bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `/processing/running/speed/data` and its timestamps.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. This is the SDK-processed wheel-speed stream described by the whitepaper.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Raw samples within retained trial windows are pooled globally to determine quantiles. During conversion, speed is linearly interpolated at each 100 ms center, then discretized.

ii.
```python
running_edges = compute_bin_edges(running_pool, 5)
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Global percentile definitions keep labels consistent and approximately balanced; interpolation uses hardware-synchronized timestamps.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global 20/40/60/80 percentile edges create five classes (`q1`–`q5`) via `np.digitize`.

ii.
```python
quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, quantiles)
bins = np.digitize(values, edges, right=False)
```

iii. Five equal-percentile bins are explicitly required, and global edges make categories comparable across files.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Linear interpolation samples running speed at the same grid centers corresponding to neural event-sum bins.

ii.
```python
running_interp = interpolate_series(running_times, running_values, centers)
```

iii. The notes rely on the shared synchronized clock and validate that interpolation is finite.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses raw pupil area, eye-tracking timestamps, and the `likely_blink` stream.

ii.
```python
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The agent interpreted area as convertible to an equivalent circular diameter and used blink flags to exclude artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink areas become NaN; finite nonnegative area is converted to equivalent diameter `2*sqrt(area/pi)`. Trial-window diameters determine global quantiles; diameter is linearly interpolated at bin centers and discretized. Files without adequate eye tracking are excluded.

ii.
```python
pupil_area[likely_blink] = np.nan
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. The notes cite blink filtering and seek a true diameter-like measure, preferring session exclusion to fabricated pupil values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global 20/40/60/80 percentile edges produce five `q1`–`q5` categories with `np.digitize`.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. This implements five equal-percentile bins consistently across included data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Blink-filtered equivalent diameter is linearly interpolated at the same 100 ms centers as neural bins.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Eye and ophys streams are synchronized; shared query times enforce equal trial lengths.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the four trial boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. These are the SDK's canonical mutually exclusive go/catch outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag is required. Its fixed-list index (0–3) is repeated across every time bin in that trial.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Repetition keeps all five outputs in a uniform `(n_output, n_timepoints)` matrix while preserving a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Byte strings are decoded; NaN booleans become false; pupil blink/missing samples become NaN; interpolation handles zero/one finite sample generically but eligible sessions must have adequate pupil data; interpolation must be entirely finite or conversion raises. Missing-eye, too-few-trial, and insufficient-pupil sessions are recorded and excluded. Invalid outcome rows raise rather than being silently repaired.

ii.
```python
go = np.nan_to_num(trials["go"], nan=0.0).astype(bool)
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The notes favor explicit exclusion/failure over fabricating required outputs, and record excluded experiment IDs and reasons in metadata.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large neural event matrices and converting 281 files dominates. Files are also opened/read in both preview and conversion passes, and neural rebinning runs a Python loop over every trial bin.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
...
for b in range(T):
    neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Notes identify repeated file scans and per-bin event sums as intentional memory/CPU tradeoffs; preview avoids neural matrices and processing remains one file at a time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural `for b in range(T)` slice-sum loop could use cumulative sums or grouped reductions. Presentation-row loops and trial-window collection loops could also be vectorized, though they are smaller.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes explicitly identify cumulative sums as a speedup but retain slice sums to reduce peak memory.

## 9-c. What processing does the code repeat multiple times?

i. Every eligible NWB is opened in preview and again in conversion. Trials, stimulus presentations, running, pupil area/blinks, and derived pupil diameter are read/computed in both passes. Optional plotting additionally recomputes grid projections/interpolations for a representative trial.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. This repetition is documented as a deliberate two-pass strategy for global bin edges while limiting memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads/stores `change_time`, go/catch flags, session ID/type, presentation sham/trial/active fields, event ROI IDs, and bin start arrays that are not all used in final output calculation. `starts` from `build_bin_centers` is needed for neural bounds, but returned `ends`/some preview metadata have limited downstream use; optional plots create diagnostics not used by the decoder.

ii.
```python
keep_names = ["start_time", "stop_time", "image_name", "omitted",
              "is_change", "is_sham_change", "trials_id", "active"]
...
event_data, _ = load_neural_events(f)
```

iii. Most extra fields support validation, diagnostics, or semantic reconstruction rather than the saved decoder arrays; the notes intentionally preserve auditability and optional sanity plots.
