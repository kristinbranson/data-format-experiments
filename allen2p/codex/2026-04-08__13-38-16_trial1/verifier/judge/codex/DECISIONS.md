# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the AllenSDK `VisualBehaviorOphysProjectCache` reference path. Instead, it enumerated local NWB experiment files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, opened them directly with `h5py`, ran a preview pass over every file, and then reopened eligible files in a conversion pass. Trial tables, stimulus-presentation tables, running, pupil, and neural event arrays were all read from NWB/HDF5 groups.

ii. ```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

def main() -> None:
    files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
    eligible, excluded = collect_previews(
        files=files,
        bin_size_sec=BIN_SIZE_SEC,
        sample_mode=args.sample,
        required_eligible=2,
    )
```

```python
def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        trials = read_trials(f)
        presentations = read_task_presentations(f)
        running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the agent said it used direct HDF5 reads from local NWB files rather than `pynwb`, and in Step 10 it justified this as using the “same underlying NWB tables/fields” while bypassing the AllenSDK object model. The trajectory also states that it chose a “two-pass workflow” over the local NWB files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as unique NWB `/general/subject/subject_id` values encountered among eligible experiment files. The script builds `subjects` in encounter order and maps each converted experiment-file session to a `subject_idx`.

ii. ```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
...
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes describe subjects as “38 local mice represented in eligible experiments,” and the trajectory says the local NWB payload was treated as authoritative for conversion scope. There is no sign that the agent reconstructed subjects from the project experiment table.

## 1-c. How are the data split into sessions?

i. The AI treated each NWB `behavior_ophys_experiment_*.nwb` file as one session. Although it read `ophys_session_id`, it did not group multiple experiments by shared `ophys_session_id`; each eligible experiment file became one output session.

ii. ```python
experiment_id = int(decode_scalar(f["/identifier"][()]))
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview,
        ...
    )
    neural_sessions.append(neural_trials)
```

iii. In Step 9 of `CONVERSION_NOTES.md`, the agent explicitly summarized the dataset as “281 eligible local experiment files,” and in the README it described “Included sessions: 281 ophys experiment files.” The trajectory likewise says it would process “eligible sessions” discovered from experiment files rather than reconstructed SDK sessions.

## 1-d. How are the data split into trials?

i. Trials are taken from each NWB file’s `/intervals/trials` table, converted into `TrialSpec` records, and then each trial window is rebinned onto fixed 100 ms bins from `start_time` to `stop_time`. Trials are therefore fixed-width in bin size but variable in number of bins.

ii. ```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    ...
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        ...
        specs.append(
            TrialSpec(
                trial_idx=idx,
                start_time=float(trials["start_time"][idx]),
                stop_time=float(trials["stop_time"][idx]),
```

```python
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The Step 5 notes say “Alignment event = trial start,” and Step 6 says the code uses “a common 100 ms trial-relative time base across all sessions.” The trajectory also says the agent chose to “rebin all sessions to one common trial-relative time base.”

## 1-e. How are trials filtered based on quality controls?

i. The AI excluded aborted and auto-rewarded trials, required exactly one of `hit/miss/false_alarm/correct_reject` to be true, and excluded experiment-file sessions with fewer than 2 such trials. It did not explicitly exclude trials with missing `change_time`; those remain in `TrialSpec` with `math.nan`.

ii. ```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
```

```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. The Step 5 notes say “Use SDK-valid trials only” and “Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed.” Step 6 adds an extra session-level exclusion for missing eye tracking because pupil was considered required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from the NWB event-detection arrays, specifically `/processing/ophys/event_detection/data` and `/processing/ophys/event_detection/rois`, with ROI validity taken from `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`.

ii. ```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
    )
```

iii. The Step 5 notes state “Neural signal = event-detection output, not dF/F,” justified because the paper “explicitly uses discrete calcium events.” The trajectory repeats that the main design choice was to use event-detection outputs as neural activity.

## 2-b. How is the `neural` data processed?

i. Neural event traces are filtered to valid ROIs and then rebinned into 100 ms trial bins by summing event magnitudes whose ophys timestamps fall within each bin. The result is a `(n_neurons, n_bins)` matrix per trial.

ii. ```python
event_data, _ = load_neural_events(f)
...
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
    frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
    frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
    ...
    neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
    for b in range(T):
        lo = int(frame_starts[b])
        hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 6 says the code uses a common 100 ms time base and that “session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums.” The notes frame this as an intentional common-bin representation for the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applied ROI quality control by masking event-detection traces with the NWB `valid_roi` flag. No further neural activity thresholding or session-level neural filtering was applied beyond excluding sessions with missing eye tracking.

ii. ```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The Step 4 and Step 5 notes say the agent wanted to “keep SDK `valid_roi` filtering logic” even when reading raw NWBs directly. In Step 10 it described this as matching the AllenSDK default ROI loading semantics.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start and then represented on ophys-timestamp-derived 100 ms bins spanning each trial’s `start_time` to `stop_time`. Ophys timestamps determine which event samples are summed into each bin.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "trial start",
    "off_start": 0.0,
    "off_end": None,
}
```

iii. Step 5 says “Alignment event = trial start,” and Step 12 says the agent rechecked that neural, stimulus, running, and pupil traces were “synchronized on the same ophys-aligned trial window.” The justification was that trial start is the natural per-trial anchor while still aligning streams by ophys time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms time bin for all sessions and trials. Yes, temporal rebinning is applied: event data are summed into 100 ms bins, while running and pupil are interpolated to 100 ms bin centers.

ii. ```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
...
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. Step 5 calls the “tentative common bin size = 100 ms,” arguing that it was coarse enough for 11 Hz data while still resolving 250 ms flashes. Step 10 then defends the added rebinning as a decoder-format requirement rather than reference processing.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation tables under `/intervals/*_presentations`, using at least `image_name`, `start_time`, `stop_time`, and `omitted`.

ii. ```python
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
```

```python
def make_image_series(...):
    ...
    omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. Step 5 maps “Stimulus presentation `image_name` + presentation timing + omission state” to `image_identity`, and the notes justify adding `gray` so that identity is defined during inter-stimulus intervals and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI assembled a global image vocabulary consisting of `gray` plus all nonempty, non-`omitted` image names. Within each trial, it projected presentation intervals onto 100 ms trial bins, filled the default value with `gray`, and overwrote bins covered by non-omitted presentations with the presented image code.

ii. ```python
def unique_nonempty_images(presentations: Dict[str, np.ndarray]) -> List[str]:
    ...
    names = [
        str(x)
        for x in presentations["image_name"]
        if str(x) not in {"", "nan", "None", "omitted"}
    ]
    return sorted(set(names))
```

```python
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
```

iii. The Step 5 notes explicitly state “Image identity will include a `gray` class” because trials contain gray ISI and omission periods. Step 10 documents a bug fix where `omitted` was removed from the image vocabulary so only `gray + 16 task images` remained.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by projecting presentation windows onto the same 100 ms trial bins used for neural data. A bin gets the image code if its center falls within a presentation interval; otherwise it remains `gray`.

ii. ```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

```python
in_window = (centers >= start) & (centers < stop)
...
image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The Step 10 comparison claims this is “same source variables and semantics” with only added common-bin rebinning. The trajectory says the agent chose trial-relative ophys-aligned bins so all outputs would share the neural time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation tables’ `is_change`, `start_time`, `stop_time`, and `omitted` fields, not from the trial table’s `change_time`/`go` fields.

ii. ```python
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
```

```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 maps “Stimulus presentation `is_change` + presentation timing” to `image_change`, and Step 10 says output construction comes from presentation-table variables rather than trial-table `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI created a binary series over 100 ms bins, initialized to 0, and set bins to 1 whenever their centers overlapped a non-omitted presentation marked `is_change`. This marks the changed presentation interval itself, not a 750 ms post-change window.

ii. ```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The Step 5 notes say the binary series is “1 during the changed-image presentation immediately after a true image-identity change, else 0.” The justification is tied to the presentation table rather than trial `change_time`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous threshold is used. The AI directly encoded `image_change` as a binary categorical series with values 0 (`no_change`) and 1 (`change`).

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
]
```

iii. The agent treated image change as an already discrete task variable. That is consistent with the task framing in both the notes and the final README.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned to the neural data by projecting `is_change` presentation intervals onto the same 100 ms trial bins used for neural event sums.

ii. ```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
```

iii. The Step 10 notes say the output matrices reconstructed from NWB “matched the converted pickle exactly,” and they present the binwise projection as the agent’s intended alignment method.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running processing group, specifically `/processing/running/speed/timestamps` and `/processing/running/speed/data`.

ii. ```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 6 says the code mirrors SDK semantics and takes running from `/processing/running/speed`. The notes justify this as using the NWB serialization of the Allen running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first pooled across valid trial windows during the preview pass to compute global quintile edges. During conversion, it is linearly interpolated to 100 ms trial-bin centers with `np.interp`, required to be finite, and then discretized with those global edges.

ii. ```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
```

```python
running_interp = interpolate_series(running_times, running_values, centers)
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 5 planned “interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins,” and Step 6 says the two-pass workflow exists partly to compute global running/pupil percentile edges before final conversion.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-quantile bins using global bin edges computed from all finite running samples collected across eligible trial windows.

ii. ```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. The Step 5 mapping explicitly chose “5 global percentile bins,” and Step 9 confirms the resulting running-bin distribution was near-uniform “by construction.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolating it to the same 100 ms trial-bin centers used for neural data. The aligned, discretized running bins share the same number of time bins `T` as the neural trial matrix.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
        running_bins,
```

iii. Step 12 says temporal alignment was checked using processing plots and that running interpolation and neural event traces were synchronized on the same ophys-aligned trial window.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/eye_tracking/timestamps`, `/acquisition/EyeTracking/pupil_tracking/area_raw`, and `/acquisition/EyeTracking/likely_blink/data`. The agent did not use the SDK-style `pupil_width` field.

ii. ```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(
    f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64
)
blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. Step 5 says pupil should use “processed pupil area after blink filtering” and then be converted to an equivalent diameter. The trajectory also notes the agent excluded sessions with missing eye tracking because pupil was a required decoder target.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to `NaN`, pupil area is converted to an equivalent diameter via `2*sqrt(area/pi)`, values are pooled across valid trial windows to compute global quintile edges, and then per-trial pupil is interpolated to 100 ms trial-bin centers and discretized. Sessions lacking eye tracking, or with too few valid pupil samples, are excluded.

ii. ```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=np.float64)
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

```python
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Step 5 explicitly documents the area-to-diameter conversion and the exclusion of sessions with missing eye tracking. Step 6 repeats that missing-eye-tracking sessions are excluded because pupil diameter is a required output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-quantile bins using global bin edges computed from all finite pooled pupil-diameter samples across eligible trial windows.

ii. ```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The Step 5 mapping uses the same “5 global percentile bins” strategy as running speed, and Step 9 confirms the pooled pupil-bin distribution was approximately quintile-like.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating the processed pupil signal to the same 100 ms trial-bin centers used for the neural data, then discretizing those aligned values.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
        running_bins,
        pupil_bins,
```

iii. Step 12 says the processing plots showed “pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window.”

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
```

iii. Step 5 maps “Trial outcome flags `hit`, `miss`, `false_alarm`, `correct_reject`” directly to `trial_outcome`, consistent with the SDK trial semantics the agent extracted in Step 1.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI required exactly one outcome flag to be true, encoded the chosen category as an integer index 0 to 3, and repeated that code across all time bins in the trial so every `output_trial` remained time-varying with shape `(5, T)`.

ii. ```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
        running_bins,
        pupil_bins,
        outcome_series,
    ]
)
```

iii. Step 5 says trial outcome is a “single categorical value per trial, repeated across all time bins in that trial to keep output arrays uniformly time-varying,” and Step 5 Key Decision 11 reiterates that this repetition was intentional for shape uniformity.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled missing or problematic data by excluding whole experiment files with missing eye tracking, excluding sessions with too few valid pupil samples or fewer than two valid trials, masking blink-contaminated pupil samples to `NaN`, using endpoint-filled interpolation for running and pupil, and raising an error if interpolated running or pupil still contained non-finite values during conversion. It also treated genuine all-zero event trials as acceptable after post hoc validation.

ii. ```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

```python
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. Step 6 says the script excludes sessions with missing eye tracking because pupil is required. Step 10 then documents that all-zero neural warnings were investigated against raw NWBs and intentionally not fixed because they were judged to be genuine sparse-event trials.

## 9-a. What are the most time-consuming steps of the code?

i. The agent’s own notes identify the expensive parts as the full preview pass over all NWB files and the conversion pass that reopens them, with per-session event-data rebinning adding CPU cost. In other words, file I/O over large NWBs dominates, with some extra cost from per-bin event summation.

ii. ```python
eligible, excluded = collect_previews(...)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. Step 6 says “Full preview scans all NWB files once and the conversion pass opens them again,” and notes that per-bin slice sums trade memory for extra CPU. Step 9 reports measured preview and conversion runtimes for the full run.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the inner per-bin loop that sums event magnitudes for every 100 ms bin in every trial. The agent explicitly noted that cumulative sums or a more vectorized rebinning strategy could replace the current per-bin slice-sum loop.

ii. ```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 6 names this inefficiency directly: “Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU.”

## 9-c. What processing does the code repeat multiple times?

i. The AI intentionally repeats dataset scanning by opening the NWB files in a preview pass and then reopening eligible files in the full conversion pass. It also computes trial-window bin centers and interval overlap logic separately in preview, conversion, raw-check scripts, and optional plotting.

ii. ```python
eligible, excluded = collect_previews(...)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. Step 6 explicitly says the code uses “a two-pass workflow” and that “Full preview scans all NWB files once and the conversion pass opens them again; this is intentional.” The trajectory similarly describes a “light first pass” followed by a “second pass” that builds the final arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and stores several trial-table fields that are not used to generate the final outputs, including `change_time`, `change_frame`, `initial_image_name`, `change_image_name`, `go`, and `catch`; final image/change outputs are built from presentation tables instead. It also supports optional diagnostic plots and tracking fields such as `trial_idx`, `ophys_session_id`, and `session_type` that are useful for validation/logging but not used by downstream decoder analyses.

ii. ```python
names = [
    "id",
    "start_time",
    "stop_time",
    "go",
    "catch",
    "aborted",
    "auto_rewarded",
    "hit",
    "miss",
    "false_alarm",
    "correct_reject",
    "change_time",
    "change_frame",
    "initial_image_name",
    "change_image_name",
]
```

```python
parser.add_argument(
    "--show-processing",
    action="store_true",
    help="Plot processing diagnostics for up to 2 sessions.",
)
```

iii. The notes justify these extras as validation aids: Step 6 says optional processing plots are for diagnostics, and Step 10 describes standalone raw-NWB reconstruction checks used to inspect warnings. The final converted dataset does not consume those extra fields or diagnostic artifacts.
