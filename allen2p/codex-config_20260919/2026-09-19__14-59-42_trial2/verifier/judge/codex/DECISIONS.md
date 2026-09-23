# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK cache. It reads the project metadata CSV directly, maps available local NWB files, filters to `VisualBehavior` + `active_behavior` + non-passive experiments, drops experiments that lack pupil data, and then opens each NWB file with `h5py` during conversion.

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
    has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
...
with h5py.File(path, "r") as nwb:
    event_ds = nwb[f"{EVENT_ROOT}/data"]
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as matching the supplied local V1.1 release directly, avoiding unnecessary SDK object materialization, and restricting to active, eye-equipped sessions because passive sessions lack meaningful trial outcomes and pupil is a required decoder target.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the selected experiment table rows, sorted numerically and stored as strings.

ii.
```python
subjects = sorted(selected["mouse_id"].astype(str).unique(), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[str(mouse)] for mouse in selected["mouse_id"]], dtype=np.int64
),
```

iii. The notes say subjects should track the release metadata’s mouse identities and that all 37 mice remain after excluding the three sessions without eye tracking.

## 1-c. How are the data split into sessions?

i. Each retained `ophys_experiment_id` is treated as one session. The AI relies on the fact that in the supplied single-plane `VisualBehavior` project, experiment and session are one-to-one.

ii.
```python
for session, (_, row) in enumerate(selected.iterrows()):
    converted = convert_session(
        row=row,
        image_to_id=image_to_id,
        show_processing=show_processing and session < 2,
    )
```

iii. The notes explicitly state that `VisualBehavior` is single-plane and that “experiment/session IDs are one-to-one,” so one NWB experiment file is used as one output session.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI keeps eligible go/catch trials, converts each to a half-open ophys-frame interval `[start_time, stop_time)`, and stores variable-length trials.

ii.
```python
def _trial_table(nwb: h5py.File) -> dict[str, np.ndarray]:
    group = nwb[TRIAL_ROOT]
    needed = (
        "id", "start_time", "stop_time", "change_time",
        "go", "catch", "aborted", "auto_rewarded", *OUTCOME_COLUMNS,
    )
    return {name: np.asarray(group[name][:]) for name in needed}

lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
...
specs.append({"trial_id": int(trials["id"][raw_idx]), "lo": lo, "hi": hi, ...})
```

iii. The notes justify full SDK-defined trial windows rather than fixed change-centered windows because trial duration is inherently variable and the decoder outputs are meant to stay time-varying inside the native trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, requires exactly one valid outcome flag, requires at least 2 ophys frames per trial, and requires at least 2 retained trials per session.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
...
if outcome_flags.sum() != 1:
    raise RuntimeError(...)
...
if hi - lo < 2:
    raise RuntimeError(...)
...
if len(specs) < 2:
    raise RuntimeError(...)
```

iii. The notes say this follows the task’s explicit go/catch vs aborted/auto-rewarded curation and avoids inventing extra behavioral filters beyond validity/invariant checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the released raw FastLZero calcium-event array at `processing/ophys/event_detection/data`, with timestamps from the paired event timestamp dataset.

ii.
```python
event_ds = nwb[f"{EVENT_ROOT}/data"]
ophys_t = np.asarray(nwb[f"{EVENT_ROOT}/timestamps"][:], dtype=np.float64)
```

iii. The notes state that the supplied paper analyzed released detected calcium events rather than dF/F, so the AI chose raw L0 events as the closest reference-matched neural signal.

## 2-b. How is the `neural` data processed?

i. The AI does minimal processing: it validates event/timestamp length and ROI mapping, reads the event matrix as `float32`, slices each trial in time, and transposes to `(neurons, time)`. It does not smooth, normalize, or recompute events/dF/F.

ii.
```python
if event_ds.shape[0] != len(ophys_t):
    raise RuntimeError("Event/timestamp length mismatch")
...
events = event_ds.astype(np.float32)[:]
...
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. The notes say extra smoothing or normalization would move away from the released event signal used in the reference paper, so the AI intentionally preserves the raw published event magnitudes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no extra post-release neuron filter. The AI only checks that all ROIs marked in the published file are valid and that the event ROI mapping is complete.

ii.
```python
cell_ids = np.asarray(nwb[f"{CELL_ROOT}/cell_specimen_id"][:], dtype=np.int64)
valid_rois = np.asarray(nwb[f"{CELL_ROOT}/valid_roi"][:], dtype=bool)
event_rois = np.asarray(nwb[f"{EVENT_ROOT}/rois"][:], dtype=np.int64)
if not valid_rois.all() or len(event_rois) != n_neurons:
    raise RuntimeError("Unexpected invalid/missing event ROI in published NWB")
```

iii. The notes say published NWB files already reflect Allen QC and that inventing a new activity-based neuron filter would not be reference grounded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the synchronized ophys timebase and segmented from the first ophys frame at or after `start_time` up to the first frame at or after `stop_time`. The metadata describe the alignment event as trial start on the ophys grid.

ii.
```python
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
...
neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
...
"temporal_alignment_event": (
    "SDK trial start on the synchronized ophys timestamp grid "
    "(first ophys frame at or after start_time)"
),
```

iii. The notes justify using native ophys timestamps as the master clock because the task explicitly asked for ophys-time alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native single-plane ophys/event frame rate, about 31 Hz, with no temporal rebinning or resampling of the neural signal.

ii.
```python
"median_ophys_interval_ms": float(np.median(np.diff(ophys_t)) * 1000.0),
...
median_bin_ms = float(np.median([x["median_ophys_interval_ms"] for x in session_info]))
...
"time_bin_size": median_bin_ms,
"neural_sampling": "Native single-plane ophys frames, nominally 31 Hz",
```

iii. The notes say the paper sometimes interpolated event-triggered analyses to 30 Hz, but the decoder instructions explicitly prioritize native ophys timestamps instead.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from stimulus-presentation interval tables, using `image_name`, `start_time`, `stop_time`, and `active`, while excluding `omitted` presentations.

ii.
```python
for group in _image_presentation_groups(nwb):
    names = _decode_strings(group["image_name"][:])
    starts = np.asarray(group["start_time"][:], dtype=float)
    stops = np.asarray(group["stop_time"][:], dtype=float)
    active = np.asarray(group["active"][:], dtype=bool)
```

iii. The notes say image identity should be built from the actual presentation table, not just trial-level initial/change image names, because the stimulus has explicit image and gray intervals and explicit omission records.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a global 16-image vocabulary, reserves code `0` for `gray`, writes image IDs only during non-omitted active presentation intervals, and leaves gray elsewhere (including the inter-stimulus gray screen and omissions).

ii.
```python
image_names = collect_image_names(selected)
image_to_id = {name: idx + 1 for idx, name in enumerate(image_names)}
...
identity = np.zeros(len(ophys_t), dtype=np.int16)  # 0 is gray
...
if hi <= lo or name == "omitted":
    continue
...
identity[lo:hi] = image_to_id[name]
```

iii. The notes justify the explicit `gray` class by arguing that no image is displayed for most of the 750 ms cadence, so the per-timepoint output should distinguish image frames from gray periods rather than carry the image label through the whole interval.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first rasterized onto the full ophys timestamp grid, then each trial reuses the same `[lo:hi)` indices as the neural array.

ii.
```python
identity, image_change, n_presentations = _stimulus_on_ophys_grid(
    nwb, ophys_t, image_to_id
)
...
decoder_output = np.vstack(
    (
        identity[lo:hi],
        image_change[lo:hi],
        ...
    )
)
```

iii. The notes say this preserves exact synchronized presentation timing on the same clock as the neural data, including the gray gaps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derives image change from the stimulus-presentation table’s `is_change` annotation rather than from trial-table `change_time` alone.

ii.
```python
is_change = np.asarray(group["is_change"][:], dtype=float)
...
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. The notes say this makes image change track the actual changed-image presentation interval and ensures catch sham changes stay zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI initializes a zero vector on the ophys grid and sets it to `1` for the duration of an active changed-image presentation. It does not create a one-frame pulse or a full 750 ms post-change window.

ii.
```python
change = np.zeros(len(ophys_t), dtype=np.int16)
...
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. The notes explicitly justify choosing the 250 ms changed-image interval as the “right after” period, instead of the whole flash-plus-gray 750 ms cycle.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary with categories `0 = no_change` and `1 = change`.

ii.
```python
"output_values": [
    ["gray", *image_names],
    ["no_change", "change"],
    ...
]
```

iii. The notes treat image change as a categorical indicator of changed-image presentations and keep catch sham changes in the `0` class.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is rasterized on the full ophys grid and then sliced with the same per-trial indices used for neural data.

ii.
```python
identity, image_change, n_presentations = _stimulus_on_ophys_grid(
    nwb, ophys_t, image_to_id
)
...
decoder_output = np.vstack(
    (
        identity[lo:hi],
        image_change[lo:hi],
        ...
    )
)
```

iii. The notes justify this as exact clock-level alignment to synchronized presentation intervals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the released running-speed stream at `processing/running/speed`, using both `timestamps` and `data`.

ii.
```python
run_t = np.asarray(nwb[f"{RUN_ROOT}/timestamps"][:], dtype=np.float64)
run_source = np.asarray(nwb[f"{RUN_ROOT}/data"][:], dtype=np.float64)
```

iii. The notes say this is the released filtered running signal and should be used directly rather than recomputed from raw encoder channels.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates finite running samples to the ophys timestamps, then bins the interpolated values into quintiles computed separately within each session using only samples that are retained inside go/catch trials.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
...
used_indices = np.concatenate(
    [np.arange(spec["lo"], spec["hi"], dtype=np.int64) for spec in specs]
)
running_bin, running_thresholds = _quantile_bins(running, used_indices)
```

iii. The notes justify per-session percentile thresholds as a way to preserve within-session behavioral state structure and avoid pooling different animals/rig scales.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five bins using the 20th, 40th, 60th, and 80th percentiles of retained running samples from that session. `np.digitize(..., right=False)` maps values to bins `0` through `4`.

ii.
```python
thresholds = np.percentile(retained, [20, 40, 60, 80])
...
bins = np.digitize(values, thresholds, right=False).astype(np.int16)
...
list(QUINTILE_NAMES)
```

iii. The notes explicitly describe this as per-session quintile discretization over the converted samples only.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full ophys timebase before any trial slicing, then each trial uses the same `[lo:hi)` indices as the neural matrix.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
...
decoder_output = np.vstack(
    (
        identity[lo:hi],
        image_change[lo:hi],
        running_bin[lo:hi],
        ...
    )
)
```

iii. The notes say the ophys grid is the master clock, so interpolation is the appropriate way to synchronize running with neural activity.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking timestamps plus the processed pupil `area` dataset in the NWB eye-tracking hierarchy.

ii.
```python
eye_t = np.asarray(nwb[f"{EYE_ROOT}/eye_tracking/timestamps"][:], dtype=np.float64)
pupil_area = np.asarray(nwb[f"{EYE_ROOT}/pupil_tracking/area"][:], dtype=np.float64)
```

iii. The notes say the processed area stream already reflects the reference blink/outlier masking and that area-based diameter was chosen to match the whitepaper’s pupil-diameter definition better than `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts processed pupil area to diameter with `2*sqrt(area/pi)`, linearly interpolates finite values to the ophys timestamps, tracks missing-gap statistics, and then bins the interpolated signal into per-session quintiles over retained trial samples.

ii.
```python
with np.errstate(invalid="ignore"):
    pupil_diameter_source = 2.0 * np.sqrt(pupil_area / np.pi)
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
pupil_missing, pupil_max_gap = _nan_gap_stats(eye_t, pupil_diameter_source)
...
pupil_bin, pupil_thresholds = _quantile_bins(pupil, used_indices)
```

iii. The notes justify preserving the release’s blink masking, interpolating only the processed signal, and avoiding arbitrary trial rejection for long masked gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five per-session percentile bins using the 20th, 40th, 60th, and 80th percentiles of retained pupil samples, again with `np.digitize(..., right=False)` to produce bins `0` through `4`.

ii.
```python
thresholds = np.percentile(retained, [20, 40, 60, 80])
...
bins = np.digitize(values, thresholds, right=False).astype(np.int16)
...
"pupil_quintile_thresholds_pixels": pupil_thresholds.tolist(),
```

iii. The notes say per-session thresholding avoids mixing across sessions with different pupil scales.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the ophys timestamp grid before trial extraction, then trial slices use the same `[lo:hi)` indices as the neural data.

ii.
```python
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
...
decoder_output = np.vstack(
    (
        identity[lo:hi],
        image_change[lo:hi],
        running_bin[lo:hi],
        pupil_bin[lo:hi],
        outcome,
    )
)
```

iii. The notes say all streams should be synchronized on the ophys grid for the decoder.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the mutually exclusive trial flags `hit`, `miss`, `false_alarm`, and `correct_reject` in the trial table.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcome_flags = np.array(
    [bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool
)
```

iii. The notes say these four categories are the valid go/catch outcomes after aborted and auto-rewarded trials are removed.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the one-hot outcome flags to integers `0..3`, stores that integer in `spec["outcome"]`, and repeats the value across all time bins in the trial so every output row shares shape `(5, T)`.

ii.
```python
"outcome": int(np.flatnonzero(outcome_flags)[0]),
...
outcome = np.full(hi - lo, spec["outcome"], dtype=np.int16)
...
"outcome_storage": "Static value repeated across time to share a (5,T) output array",
```

iii. The notes justify time-broadcasting as a storage-format choice only; semantically the variable is still static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases by hard curation or interpolation: it excludes entire sessions with no pupil stream, rejects malformed trials/outcomes with runtime errors, interpolates finite running and pupil values to the ophys grid, records pupil missing-gap statistics, and does not discard zero-event trials.

ii.
```python
with h5py.File(path, "r") as nwb:
    has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
...
if outcome_flags.sum() != 1:
    raise RuntimeError(...)
if hi - lo < 2:
    raise RuntimeError(...)
...
return np.interp(target_t, source_t[unique], source_x[unique]).astype(np.float32)
...
pupil_missing, pupil_max_gap = _nan_gap_stats(eye_t, pupil_diameter_source)
```

iii. The notes say missing-eye sessions had to be excluded because pupil is required, blink-masked pupil gaps should be interpolated rather than restored from raw outliers, and sparse all-zero event trials should be retained because adding an activity filter would be unjustified.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify large NWB/HDF5 reads, especially session-scale event arrays, as the main cost. Writing the multi-gigabyte pickle is also a substantial end-of-pipeline cost.

ii.
```python
with h5py.File(path, "r") as nwb:
    event_ds = nwb[f"{EVENT_ROOT}/data"]
...
events = event_ds.astype(np.float32)[:]
...
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes contrast this with heavier AllenSDK/PyNWB object loading and say the converter was structured to read only the needed datasets one session at a time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes the heavy numeric work, but the remaining Python loops that could be further optimized are the per-presentation loop in `_stimulus_on_ophys_grid`, the per-trial loop in `_trial_specs` / `convert_session`, and the per-file scans in `select_experiments` and `collect_image_names`.

ii.
```python
for name, start, stop, is_active, changed in zip(...):
    ...
for raw_idx in np.flatnonzero(keep):
    ...
for spec in specs:
    ...
for path in selected["nwb_path"]:
    ...
```

iii. The notes say the converter intentionally avoided per-sample Python loops and used `np.searchsorted`, `np.interp`, `np.percentile`, and `np.digitize` for the bulk work, because those dominate the runtime profile more than the remaining control-flow loops.

## 9-c. What processing does the code repeat multiple times?

i. The code rescans all selected NWB files once to collect the global image vocabulary and then opens them again for actual conversion. It also recomputes some small per-session summary statistics such as `np.diff(ophys_t)` in multiple places for validation/metadata.

ii.
```python
def collect_image_names(selected: pd.DataFrame) -> list[str]:
    for path in selected["nwb_path"]:
        with h5py.File(path, "r") as nwb:
            ...
...
for session, (_, row) in enumerate(selected.iterrows()):
    converted = convert_session(...)
...
"median_ophys_interval_ms": float(np.median(np.diff(ophys_t)) * 1000.0),
...
ophys_dt=np.diff(ophys_t),
```

iii. The notes frame the first extra pass as a deliberate way to build a consistent global image mapping before conversion; the repeated summary calculations support validation and metadata rather than core signal extraction.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and stores a number of audit-only quantities that are not needed by downstream decoding, such as missing-gap statistics, conversion timing, per-session thresholds, per-session metadata, optional processing plots, and internal structural validation. These checks are helpful for debugging but not used as decoder inputs/outputs.

ii.
```python
pupil_missing, pupil_max_gap = _nan_gap_stats(...)
...
if show_processing:
    _plot_processing(...)
...
info = {
    ...
    "running_quintile_thresholds_cm_per_s": running_thresholds.tolist(),
    "pupil_quintile_thresholds_pixels": pupil_thresholds.tolist(),
    "pupil_source_missing_fraction": pupil_missing,
    "pupil_source_longest_missing_gap_s": pupil_max_gap,
    "conversion_seconds": elapsed,
}
...
validate_converted(data)
```

iii. `CONVERSION_NOTES.md` makes clear these were added as sanity checks and documentation aids rather than as part of the decoder-facing representation.
