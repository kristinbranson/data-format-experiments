# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB/HDF5 files directly using `h5py`, bypassing the AllenSDK's high-level `BehaviorOphysExperiment` interface. It first reads the experiment table CSV from the project metadata directory, filters to `VisualBehavior` project code with `active_behavior` type and non-passive sessions, then verifies each NWB file has an eye-tracking stream. Selected experiments are processed one at a time.

ii.
```python
EXPERIMENT_TABLE = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"
table = pd.read_csv(EXPERIMENT_TABLE)
selected = table[
    (table["project_code"] == PROJECT_CODE)
    & (table["behavior_type"] == "active_behavior")
    & (~table["passive"].astype(bool))
    & (table["ophys_experiment_id"].isin(files))
]
# Then checks eye tracking availability per NWB file
with h5py.File(path, "r") as nwb:
    has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
```

iii. The AI chose direct HDF5 reads to avoid loading unneeded data (image templates, ROI masks) that the full SDK materializes. This yields the same arrays but with lower memory footprint and faster I/O. The active-behavior and eye-tracking filters were justified because passive sessions have invalid trial outcomes (lick spout retracted) and pupil diameter is a required decoder output.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values in the filtered experiment table, sorted numerically.

ii.
```python
subjects = sorted(selected["mouse_id"].astype(str).unique(), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. `mouse_id` is the SDK's canonical unique identifier for each animal. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each row in the filtered experiment table corresponds to one session. For the `VisualBehavior` project (single-plane), each experiment maps one-to-one to an ophys session. The AI does not group experiments by `ophys_session_id` because this is unnecessary for single-plane recordings.

ii.
```python
for session, (_, row) in enumerate(selected.iterrows()):
    converted = convert_session(row=row, image_to_id=image_to_id, ...)
```

iii. The CONVERSION_NOTES document that VisualBehavior is single-plane and experiment/session IDs are one-to-one, so treating each experiment as one session is equivalent to grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. Each trial is segmented using the SDK-defined `start_time` and `stop_time` with half-open bounds `[start_time, stop_time)`. This gives variable-length trials. Go and catch trials are kept; aborted and auto-rewarded are excluded.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
# ...
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
```

iii. The AI uses explicit `go|catch` filtering rather than excluding aborted/auto-rewarded alone. The half-open boundary convention prevents double-counting ophys frames at trial boundaries. Trials with fewer than 2 frames raise an error.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: aborted and auto-rewarded trials are excluded; only go and catch trials are retained. Outcome flags must sum to exactly 1 (exactly one of hit/miss/false_alarm/correct_reject). Trials with fewer than 2 ophys frames raise an error. Session-level: passive sessions are excluded (invalid trial outcomes); sessions without eye-tracking data are excluded (pupil diameter is a required output); sessions with fewer than 2 retained trials raise an error.

ii.
```python
# Trial filtering
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
# Outcome validation
outcome_flags = np.array(
    [bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool
)
if outcome_flags.sum() != 1:
    raise RuntimeError(...)
# Session filtering
selected = table[
    (table["project_code"] == PROJECT_CODE)
    & (table["behavior_type"] == "active_behavior")
    & (~table["passive"].astype(bool))
    & (table["ophys_experiment_id"].isin(files))
]
# Eye tracking check
has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
```

iii. The AI justifies passive-session exclusion because passive replay has the lick spout retracted, making trial outcomes meaningless. Eye-tracking exclusion is justified because pupil diameter is a required decoder output and 3 active sessions lack this stream. The strict outcome validation catches any malformed trial data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the released raw L0 calcium-event magnitudes (`processing/ophys/event_detection/data`), not dF/F traces.

ii.
```python
EVENT_ROOT = "processing/ophys/event_detection"
event_ds = nwb[f"{EVENT_ROOT}/data"]
ophys_t = np.asarray(nwb[f"{EVENT_ROOT}/timestamps"][:], dtype=np.float64)
events = event_ds.astype(np.float32)[:]
```

iii. The AI chose L0 events because "the supplied paper used detected calcium events for all neural analyses" (CONVERSION_NOTES Step 4). FastLZero event inference removes slow GCaMP decay and is the neural signal used in the reference paper's analyses.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The released L0 events are read directly from NWB and cast to float32. The data is transposed from (time, neurons) to (neurons, time) per trial slice.

ii.
```python
events = event_ds.astype(np.float32)[:]
# Per trial:
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. The L0 events are already processed by the Allen SDK pipeline. The AI explicitly verifies all values are finite and no further normalization or filtering is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All published valid ROIs (event channels) are retained. The AI verifies that `valid_roi` is all True in the published NWB files.

ii.
```python
valid_rois = np.asarray(nwb[f"{CELL_ROOT}/valid_roi"][:], dtype=bool)
if not valid_rois.all() or len(event_rois) != n_neurons:
    raise RuntimeError("Unexpected invalid/missing event ROI in published NWB")
```

iii. Published NWBs already exclude invalid ROIs. The AI verified this and documented that "no new activity threshold is described for the paper's decoder, so no post-publication neuron filter should be invented."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamp grid. Each trial extracts frames from the first ophys frame at or after `start_time` to the first ophys frame at or after `stop_time` (half-open `[start_time, stop_time)`). This gives variable-length trials aligned to trial start.

ii.
```python
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
# Invariant check:
if not (ophys_t[lo] >= start and ophys_t[hi - 1] < stop):
    raise RuntimeError("Half-open trial boundary invariant failed")
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. The half-open convention ensures no frame appears in two trials. The invariant check validates that every included frame is within the SDK-defined trial window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native single-plane ophys frame rate (~31 Hz, ~32.31 ms). The reported `time_bin_size` is the median of per-session median frame intervals.

ii.
```python
median_bin_ms = float(np.median([x["median_ophys_interval_ms"] for x in session_info]))
# Per session:
"median_ophys_interval_ms": float(np.median(np.diff(ophys_t)) * 1000.0),
```

iii. The instructions say to "temporally align based on ophys timestamp," so native ophys frames are the master grid. The paper interpolated to 30 Hz for event-triggered analyses, but the AI retains native samples because the task requires ophys alignment.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation interval tables in the NWB file (`intervals/*_presentations`). The `image_name`, `start_time`, `stop_time`, `active`, and `is_change` fields are used. Active, non-omitted presentations are mapped onto the ophys grid.

ii.
```python
def _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id):
    identity = np.zeros(len(ophys_t), dtype=np.int16)  # 0 is gray
    for group in _image_presentation_groups(nwb):
        names = _decode_strings(group["image_name"][:])
        starts = np.asarray(group["start_time"][:], dtype=float)
        stops = np.asarray(group["stop_time"][:], dtype=float)
        active = np.asarray(group["active"][:], dtype=bool)
        for name, start, stop, is_active, changed in zip(...):
            if not is_active:
                continue
            lo = int(np.searchsorted(ophys_t, start, side="left"))
            hi = int(np.searchsorted(ophys_t, stop, side="left"))
            if hi <= lo or name == "omitted":
                continue
            identity[lo:hi] = image_to_id[name]
```

iii. Using the presentation table gives frame-by-frame image identity, including gray inter-stimulus intervals (500ms between 250ms flashes) and omitted stimulus periods. This is more accurate than using trial-level `initial_image_name`/`change_image_name` because it reflects what was actually displayed at each moment.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes 1-16 (sorted), with 0 reserved for gray (inter-stimulus intervals, omissions). The global vocabulary of 16 images is collected from all selected NWBs. The identity array defaults to 0 (gray) and is set to the image code only during active, non-omitted presentation intervals.

ii.
```python
image_to_id = {name: idx + 1 for idx, name in enumerate(image_names)}
# Output values include gray:
"output_values": [["gray", *image_names], ...]
```

iii. Including gray as a separate category captures the actual visual stimulus at each moment. The AI notes: "the output is required at every neural sample and no image is displayed for two-thirds of the cadence."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the full ophys grid using `np.searchsorted` on presentation start/stop times, then sliced per trial using the same `[lo:hi]` indices as the neural data.

ii.
```python
identity, image_change, n_presentations = _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id)
# Per trial:
decoder_output = np.vstack((identity[lo:hi], ...))
```

iii. Computing image identity on the full ophys grid before trial slicing ensures frame-by-frame alignment with the neural data. Both share the same ophys timestamp indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentation intervals. A presentation is marked as changed if `is_change > 0.5` and is finite. The image change indicator is 1 during the changed-image presentation interval, 0 otherwise. Catch sham changes (where `is_change` is not set) remain 0.

ii.
```python
is_change = np.asarray(group["is_change"][:], dtype=float)
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. Using the presentation-level `is_change` field directly distinguishes real changes from catch sham changes without needing to check trial-level `go`/`catch` flags.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The binary indicator is set to 1 only during the 250ms changed-image presentation interval (`[start_time, stop_time)` of the changed presentation). No additional processing is needed.

ii.
```python
change = np.zeros(len(ophys_t), dtype=np.int16)
# For changed presentations:
change[lo:hi] = 1
```

iii. The AI marks only the 250ms image presentation window (not the following 500ms gray). The justification is that the change flag represents when the changed image is actually on screen.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = changed-image presentation). No thresholding is needed.

ii.
```python
image_change_value_names = ['no_change', 'change']  # reference code equivalent
# AI code:
["no_change", "change"]  # in output_values
```

iii. The binary nature follows directly from the stimulus design: at each moment, either a changed image is being displayed or it is not.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the full ophys grid using presentation `start_time`/`stop_time` and `np.searchsorted`, then sliced per trial using the same `[lo:hi]` indices as neural data.

ii.
```python
identity, image_change, n_presentations = _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id)
decoder_output = np.vstack((identity[lo:hi], image_change[lo:hi], ...))
```

iii. Same frame-level alignment as image identity and neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides the released filtered speed (cm/s) and timestamps.

ii.
```python
RUN_ROOT = "processing/running/speed"
run_t = np.asarray(nwb[f"{RUN_ROOT}/timestamps"][:], dtype=np.float64)
run_source = np.asarray(nwb[f"{RUN_ROOT}/data"][:], dtype=np.float64)
```

iii. The released filtered speed stream has already been processed by the SDK pipeline (encoder unwrapping, artifact rejection, Butterworth filtering).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase using only finite values. It is then discretized into 5 equal percentile bins **per session**, using retained trial samples only. Thresholds are the 20th, 40th, 60th, and 80th percentiles.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
# Per session:
running_bin, running_thresholds = _quantile_bins(running, used_indices)

def _quantile_bins(values, used_indices):
    retained = np.asarray(values[used_indices], dtype=np.float64)
    thresholds = np.percentile(retained, [20, 40, 60, 80])
    bins = np.digitize(values, thresholds, right=False).astype(np.int16)
    return bins, thresholds
```

iii. Per-session percentile discretization "preserves low-to-high state despite between-session behavioral scale differences" (CONVERSION_NOTES). The AI argues this avoids pooling pupil pixel scales across rigs/animals.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using per-session quintile thresholds (20th, 40th, 60th, 80th percentiles) computed from retained trial samples within each session.

ii.
```python
thresholds = np.percentile(retained, [20, 40, 60, 80])
bins = np.digitize(values, thresholds, right=False).astype(np.int16)
```

iii. The AI verifies that thresholds are strictly increasing (`np.any(np.diff(thresholds) <= 0)` raises an error). Each bin contains approximately 20% of the retained samples per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full ophys timebase before trial segmentation, then sliced per trial using the same `[lo:hi]` indices as neural data.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
# Per trial:
running_bin[lo:hi]  # used in decoder_output
```

iii. Interpolation to the ophys grid ensures alignment. The `_interp_finite` function handles NaN/non-finite source values.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (processed, blink-masked pupil area). Diameter is computed as `2 * sqrt(area / pi)`.

ii.
```python
pupil_area = np.asarray(nwb[f"{EYE_ROOT}/pupil_tracking/area"][:], dtype=np.float64)
with np.errstate(invalid="ignore"):
    pupil_diameter_source = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The AI justifies this formula based on the whitepaper definition: "diameter as the longest pupil ellipse axis and area from that circularized axis." The factor of 2 doesn't affect percentile bins (monotonic transform), but matches the whitepaper's definition.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area (already blink-masked by SDK processing) is converted to diameter, then linearly interpolated from finite values to the ophys timebase. Discretization into 5 bins uses **per-session** percentile thresholds computed from retained trial samples.

ii.
```python
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
pupil_bin, pupil_thresholds = _quantile_bins(pupil, used_indices)
```

iii. The processed area already has blinks/outliers set to NaN by the SDK (z-score > 3, dilated by 2 frames). The `_interp_finite` function interpolates only from non-NaN values, avoiding blink artifact propagation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: per-session quintile thresholds (20th, 40th, 60th, 80th percentiles) from retained trial samples, producing bins 0-4.

ii.
```python
pupil_bin, pupil_thresholds = _quantile_bins(pupil, used_indices)
```

iii. Same justification as running speed. The AI notes sessions without eye tracking are excluded entirely rather than filling with a default bin.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to the full ophys timebase, then sliced per trial using the same indices.

ii.
```python
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
# Per trial in decoder_output:
pupil_bin[lo:hi]
```

iii. Finite-only interpolation to the ophys grid guarantees temporal alignment with neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
outcome_flags = np.array(
    [bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool
)
if outcome_flags.sum() != 1:
    raise RuntimeError(...)
specs.append({..., "outcome": int(np.flatnonzero(outcome_flags)[0]), ...})
```

iii. These four columns are mutually exclusive for non-aborted, non-auto-rewarded trials. The AI validates this with a strict one-hot check.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single active outcome flag is mapped to an integer code 0-3 (`hit=0, miss=1, false_alarm=2, correct_reject=3`). This static value is broadcast across all time bins within the trial.

ii.
```python
outcome = np.full(hi - lo, spec["outcome"], dtype=np.int16)
decoder_output = np.vstack((identity[lo:hi], image_change[lo:hi],
                            running_bin[lo:hi], pupil_bin[lo:hi], outcome))
```

iii. Outcome is semantically static per trial. Broadcasting is required to create a homogeneous `(5, T)` output array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses strict pre-filtering and validation:
- **Missing eye tracking**: 3 active sessions entirely lack eye data and are excluded during session selection.
- **Blink-masked pupil**: SDK-processed NaN values are interpolated over using only finite samples.
- **Non-finite running/neural values**: `_interp_finite` rejects non-finite samples; neural data is verified all-finite.
- **Malformed trials**: Outcome flag validation requires exactly one active flag; trials with fewer than 2 frames raise an error.
- **Silent neural trials**: Trials with all-zero L0 events are retained (1,729 of 42,470 trials) since they represent genuine sparse events in low-cell-count sessions.

ii.
```python
# Pre-filtering for eye data:
has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
# Finite-only interpolation:
valid = np.isfinite(source_t) & np.isfinite(source_x)
if valid.sum() < 2:
    raise RuntimeError("Fewer than two finite samples in behavioral stream")
# Neural validation:
if not np.isfinite(events).all():
    raise RuntimeError("Neural event data contain NaN/Inf")
```

iii. The AI favors failing loudly over silently handling edge cases. Missing eye-tracking sessions are excluded at selection time rather than handled with try/except fallbacks. Silent neural trials are documented but retained because "the reference specifies no activity-based trial rejection."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading NWB/HDF5 files from disk. Each file contains full-session neural and behavioral data arrays. The AI reports conversion of 165 sessions completed in ~86 seconds total.

ii. N/A (timing is reported per session in stdout)

iii. Direct HDF5 reads avoid the SDK's full object materialization (image templates, ROI masks), reducing both I/O and memory. Processing is I/O-bound.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-presentation loop in `_stimulus_on_ophys_grid` iterates over each stimulus presentation to set image identity/change arrays. The per-trial loop in `convert_session` iterates to create trial slices. Both could theoretically be vectorized using advanced indexing.

ii.
```python
for name, start, stop, is_active, changed in zip(names, starts, stops, active, is_change):
    if not is_active:
        continue
    lo = int(np.searchsorted(ophys_t, start, side="left"))
    hi = int(np.searchsorted(ophys_t, stop, side="left"))
    identity[lo:hi] = image_to_id[name]
```

iii. These loops are not bottlenecks compared to file I/O. The presentation loop runs ~300-400 times per session, and the trial loop ~250 times.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each NWB file is opened once, and all data streams (events, running, pupil, trials, presentations) are read in a single pass. The image vocabulary requires a separate pass over all files to collect names before conversion.

ii.
```python
# First pass: collect image names
image_names = collect_image_names(selected)
# Second pass: convert sessions
for session, (_, row) in enumerate(selected.iterrows()):
    converted = convert_session(row=row, ...)
```

iii. The two-pass approach (vocabulary collection then conversion) requires opening each NWB twice but only reads different datasets each time.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI opens each NWB file twice: once during `collect_image_names` (to build the global image vocabulary) and once during `convert_session`. The first pass only reads presentation names but still opens the full NWB file. Additionally, the full-session event, running, and pupil arrays are read even though only trial-windowed slices are retained.

ii.
```python
# Full session read, but only trial slices retained:
events = event_ds.astype(np.float32)[:]
# ...
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. Reading full session arrays is necessary because trials can span arbitrary portions of the recording. The double NWB open for image vocabulary is a minor overhead.
