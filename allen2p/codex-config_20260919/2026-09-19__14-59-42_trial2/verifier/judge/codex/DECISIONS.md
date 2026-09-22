# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the release metadata CSV and locally available NWB files directly with pandas, h5py, and a filename-to-experiment-ID map. It selects published `VisualBehavior` experiments that are active, non-passive, locally present, and contain pupil-area data, then processes each selected file.

ii.
```python
table = pd.read_csv(EXPERIMENT_TABLE)
files = _file_map()
selected = table[
    (table["project_code"] == PROJECT_CODE)
    & (table["behavior_type"] == "active_behavior")
    & (~table["passive"].astype(bool))
    & (table["ophys_experiment_id"].isin(files))
].copy()
...
with h5py.File(path, "r") as nwb:
    event_ds = nwb[f"{EVENT_ROOT}/data"]
```

iii. The notes justify direct HDF5 access as following documented SDK paths while avoiding irrelevant templates and masks. Active sessions are chosen because passive replay is not the behavioral task; sessions without pupil data cannot provide a required output.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among selected experiments, numerically sorted; every output session receives an index into that list.

ii.
```python
subjects = sorted(selected["mouse_id"].astype(str).unique(), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
[subject_lookup[str(mouse)] for mouse in selected["mouse_id"]]
```

iii. The agent treats the release's `mouse_id` as the animal identifier and reports that all 37 mice remain after curation.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id`/NWB file is one output session. This is equivalent to one ophys session for this single-plane `VisualBehavior` project, whose experiment-to-session mapping is one-to-one.

ii.
```python
for session, (_, row) in enumerate(selected.iterrows()):
    converted = convert_session(row=row, image_to_id=image_to_id, ...)
    neural_sessions.append(neural)
```

iii. Exploration found the requested project is single-plane VISp and its 239 experiments map one-to-one to 239 ophys sessions, so no cross-plane merge is required.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`. Retained trials use the SDK half-open interval `[start_time, stop_time)`, translated to the first ophys sample at or after each boundary with `searchsorted`; trial lengths therefore vary.

ii.
```python
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
...
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. The notes say full SDK trial boundaries preserve the experimental trial and allow all requested time-varying outputs; trial start is the alignment origin.

## 1-e. How are trials filtered based on quality controls?

i. Only explicit go or catch trials are retained; aborted and auto-rewarded trials are excluded. Each must have exactly one canonical outcome, at least two ophys frames, and valid half-open boundaries. Sessions need at least two retained trials.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
...
if outcome_flags.sum() != 1: raise RuntimeError(...)
if hi - lo < 2: raise RuntimeError(...)
```

iii. This directly follows the requested go/catch inclusion and aborted/auto-rewarded exclusion. Assertions prevent malformed source trials from being silently accepted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the released event-detection `data` array and its timestamps/ROI references, i.e. raw FastLZero L0 calcium-event magnitudes, not dF/F.

ii.
```python
EVENT_ROOT = "processing/ophys/event_detection"
event_ds = nwb[f"{EVENT_ROOT}/data"]
ophys_t = np.asarray(nwb[f"{EVENT_ROOT}/timestamps"][:], dtype=np.float64)
event_rois = np.asarray(nwb[f"{EVENT_ROOT}/rois"][:], dtype=np.int64)
```

iii. The final notes say L0 events are used by the supplied paper and avoid prolonged calcium-decay contamination. The released stream is used without recomputation.

## 2-b. How is the `neural` data processed?

i. The event matrix is read as float32, checked for finite values, sliced by trial, transposed from time-by-neuron to neuron-by-time, and made contiguous. No temporal filtering, normalization, or resampling is applied.

ii.
```python
events = event_ds.astype(np.float32)[:]
if not np.isfinite(events).all():
    raise RuntimeError("Neural event data contain NaN/Inf")
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. The notes emphasize exact use of released events, float32 memory efficiency, and retention of measured native frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code requires all stored ROIs to be marked valid, checks ROI count/order against the event channels, and rejects nonfinite event data. It applies no activity-based neuron or trial filtering and retains genuine all-zero event trials.

ii.
```python
valid_rois = np.asarray(nwb[f"{CELL_ROOT}/valid_roi"][:], dtype=bool)
if not valid_rois.all() or len(event_rois) != n_neurons:
    raise RuntimeError("Unexpected invalid/missing event ROI in published NWB")
```

iii. The notes state published files already exclude invalid ROIs. Removing silent trials would be an unsupported post-publication activity filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to SDK trial start on the synchronized ophys clock. Column zero is the first ophys frame at or after `start_time`, and data extend to but exclude `stop_time`.

ii.
```python
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. The agent chose full experimental trials rather than a fixed change-centered window so pre-change context and post-change behavior remain available.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native single-plane ophys frames are retained at about 31 Hz (global/session median about 32.31 ms). No rebinning or neural resampling is applied; the dataset metadata stores the median of session median intervals.

ii.
```python
"median_ophys_interval_ms": float(np.median(np.diff(ophys_t)) * 1000.0),
...
median_bin_ms = float(np.median([x["median_ophys_interval_ms"] for x in session_info]))
```

iii. Tiny acquisition-clock variation is retained because resampling would move neural events away from measured frames.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from active stimulus-presentation interval groups: `image_name`, `start_time`, `stop_time`, and `active`. Omitted presentations are also inspected.

ii.
```python
names = _decode_strings(group["image_name"][:])
starts = np.asarray(group["start_time"][:], dtype=float)
stops = np.asarray(group["stop_time"][:], dtype=float)
active = np.asarray(group["active"][:], dtype=bool)
```

iii. The notes argue presentation intervals are necessary to represent the 250-ms image/500-ms gray cadence and omissions, which trial-level initial/change names cannot reconstruct.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Sixteen globally sorted natural-image names receive IDs 1-16; zero means gray. For every active, non-omitted presentation, its ID fills the ophys samples in its half-open interval. Omitted and inter-stimulus periods remain gray.

ii.
```python
image_to_id = {name: idx + 1 for idx, name in enumerate(image_names)}
identity = np.zeros(len(ophys_t), dtype=np.int16)
...
if hi <= lo or name == "omitted": continue
identity[lo:hi] = image_to_id[name]
```

iii. Gray is required because no image is displayed for much of the trial; omissions are treated as gray rather than a new image identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation start/stop times are mapped onto the same `ophys_t` grid, and identity is sliced with the same trial `lo:hi` indices as neural data.

ii.
```python
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
identity[lo:hi] = image_to_id[name]
...
identity[spec["lo"]:spec["hi"]]
```

iii. A shared synchronized timestamp grid guarantees column-wise alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from each active stimulus presentation's `is_change`, together with its start and stop times and image name.

ii.
```python
is_change = np.asarray(group["is_change"][:], dtype=float)
for name, start, stop, is_active, changed in zip(...):
```

iii. The presentation table distinguishes actual changed-image presentations from catch sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created on the ophys grid. Samples throughout an active, non-omitted presentation whose `is_change` is finite and positive are set to one, producing an approximately 250-ms changed-image interval.

ii.
```python
change = np.zeros(len(ophys_t), dtype=np.int16)
...
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. The agent interprets “right after” as the changed-image display interval, noting that a single-frame onset pulse would precede much of the cortical response.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The source `is_change` value is considered change when finite and greater than 0.5; output categories are integer 0 (`no_change`) and 1 (`change`).

ii.
```python
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. This converts the source indicator robustly to the requested binary output and leaves catch sham changes at zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change presentation intervals are placed on the same ophys timestamp grid and sliced with the neural trial's identical `lo:hi` bounds.

ii.
```python
change[lo:hi] = 1
...
image_change[spec["lo"]:spec["hi"]]
```

iii. The synchronized presentation and ophys times provide direct frame alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from the released filtered running-speed `data` and `timestamps` under `processing/running/speed`.

ii.
```python
run_t = np.asarray(nwb[f"{RUN_ROOT}/timestamps"][:], dtype=np.float64)
run_source = np.asarray(nwb[f"{RUN_ROOT}/data"][:], dtype=np.float64)
```

iii. The notes identify this as the SDK's standard processed speed in cm/s, so no encoder filtering is repeated.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite source samples are stably sorted, duplicate timestamps removed, and linearly interpolated onto all ophys timestamps. The result is then discretized with session-specific quintiles calculated only from retained trial samples.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
running_bin, running_thresholds = _quantile_bins(running, used_indices)
```

iii. Interpolation aligns clocks; per-session quintiles create comparable within-session states and avoid pooling across animals/rigs.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles of retained samples in each session define five categories, assigned with `np.digitize(..., right=False)`. Non-distinct thresholds cause an error.

ii.
```python
thresholds = np.percentile(retained, [20, 40, 60, 80])
if np.any(np.diff(thresholds) <= 0): raise RuntimeError(...)
bins = np.digitize(values, thresholds, right=False).astype(np.int16)
```

iii. The task requests five equal percentile bins; thresholds and achieved distributions are retained for audit.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto `ophys_t` before discretization, and its bin vector is sliced with the same trial bounds as neural data.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
...
running_bin[lo:hi]
```

iii. Shared ophys indices ensure one running label per neural column.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking timestamps and the processed pupil `area` field, whose blink/outlier samples are already NaN.

ii.
```python
eye_t = np.asarray(nwb[f"{EYE_ROOT}/eye_tracking/timestamps"][:], dtype=np.float64)
pupil_area = np.asarray(nwb[f"{EYE_ROOT}/pupil_tracking/area"][:], dtype=np.float64)
```

iii. The notes choose processed area because the SDK has already applied blink/outlier masking.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to circular-equivalent diameter `2*sqrt(area/pi)`. Finite values are linearly interpolated over the ophys grid, then categorized with retained-sample, per-session quintiles.

ii.
```python
pupil_diameter_source = 2.0 * np.sqrt(pupil_area / np.pi)
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
pupil_bin, pupil_thresholds = _quantile_bins(pupil, used_indices)
```

iii. This implements a literal diameter from processed area. The agent records missingness and gap duration but avoids arbitrary gap-based trial rejection unsupported by the paper/SDK.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Per session, the 20/40/60/80 percentiles over retained trial samples define bins 0-4, with equal-to-threshold values placed in the upper bin.

ii.
```python
thresholds = np.percentile(retained, [20, 40, 60, 80])
bins = np.digitize(values, thresholds, right=False).astype(np.int16)
```

iii. Per-session percentiles avoid combining different pupil pixel scales across rigs and animals while satisfying five equal percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Finite pupil diameter is interpolated to the neural event timestamps and then sliced with identical trial indices.

ii.
```python
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
...
pupil_bin[lo:hi]
```

iii. Hardware-synchronized timestamps and a common ophys grid give column-wise alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the four trial-table flags `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
outcome_flags = np.array(
    [bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool
)
```

iii. These are the canonical mutually exclusive outcomes for eligible go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The sole true flag's position maps outcomes to integers 0-3 in the declared order. The static code is repeated across all trial timepoints to share the homogeneous `(5,T)` output representation.

ii.
```python
if outcome_flags.sum() != 1: raise RuntimeError(...)
"outcome": int(np.flatnonzero(outcome_flags)[0]),
...
outcome = np.full(hi - lo, spec["outcome"], dtype=np.int16)
```

iii. Broadcasting is explicitly described as a storage convention; trial-level counts are separately retained so duration weighting remains visible.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter fails loudly on malformed required streams, invalid ROI mappings, nonfinite neural values, ambiguous outcomes, degenerate trials, or non-distinct quintiles. Sessions missing required eye data are excluded and recorded. For behavioral streams, nonfinite samples are removed before interpolation; pupil gaps are bridged and quantified rather than dropping trials. Duplicate/unsorted timestamps are normalized before interpolation.

ii.
```python
valid = np.isfinite(source_t) & np.isfinite(source_x)
if valid.sum() < 2: raise RuntimeError(...)
order = np.argsort(source_t, kind="stable")
unique = np.r_[True, np.diff(source_t) > 0]
return np.interp(target_t, source_t[unique], source_x[unique])
```

iii. The agent favors explicit assertions and audit metadata. It rejected arbitrary maximum pupil-gap rules because they would silently alter experimental trial curation without a reference-backed cutoff.

## 9-a. What are the most time-consuming steps of the code?

i. Reading each full event matrix and copying neural data into per-trial arrays dominates conversion; serializing the roughly 7.8-GiB pickle is another material cost. Optional plotting adds sample-mode overhead.

ii.
```python
events = event_ds.astype(np.float32)[:]
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Notes report an 85.86-s full conversion, including 8.37 s serialization, and explain that direct HDF5 reads avoid much larger SDK object-loading overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Python loops remain over sessions, trials, presentation groups/presentations, and files when collecting image names. Trial slicing is inherently ragged but metadata/boundary derivation and presentation assignment could partly be vectorized. Major sample-wise work already uses `searchsorted`, `np.interp`, percentiles, and `digitize`.

ii.
```python
for name, start, stop, is_active, changed in zip(...):
    ...
for spec in specs:
    lo, hi = spec["lo"], spec["hi"]
```

iii. The notes specifically avoid per-sample Python loops; remaining loops express ragged trial and interval construction and were fast enough for the full run.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened once during selection to check for pupil area, again while collecting global image names, and again for conversion. Active presentation groups are scanned during both vocabulary collection and output construction. Trial slices also copy portions of a full session event array.

ii.
```python
with h5py.File(path, "r") as nwb:  # select_experiments
...
with h5py.File(path, "r") as nwb:  # collect_image_names
...
with h5py.File(path, "r") as nwb:  # convert_session
```

iii. This repetition separates curation, global vocabulary validation, and conversion while keeping only one large session resident at a time.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive audit metadata (cell IDs, trial IDs, presentation counts, missing-gap statistics, thresholds, timings) and, when requested, plots eight-panel diagnostics; the decoder does not consume these. It interpolates full-session running/pupil and constructs full-session stimulus vectors although only retained trial slices are saved.

ii.
```python
pupil_missing, pupil_max_gap = _nan_gap_stats(...)
identity, image_change, n_presentations = _stimulus_on_ophys_grid(...)
if show_processing:
    _plot_processing(...)
```

iii. The agent intentionally retains these checks for scientific traceability and validation; full-grid construction simplifies consistent alignment and remains inexpensive relative to neural I/O.
