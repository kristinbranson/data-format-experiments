# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data by globbing all local NWB experiment files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` and reading them directly with `h5py`. It does not use the Allen SDK `VisualBehaviorOphysProjectCache`, and it works from the locally downloaded subset rather than reconstructing the full project via the experiment table.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        ...
```

iii. In Step 6 notes the agent says it used direct HDF5/NWB reads because the installed NWB stack was incompatible in this environment. In trajectory Step 120 it says the script would “use direct HDF5 reads from the NWBs” and mirror the SDK semantics from the serialized NWB tables.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB field `/general/subject/subject_id`. A subject is added the first time an eligible experiment file from that subject is processed.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The notes describe this as the session-level subject identifier available in the NWB metadata, and the agent used it as the mouse identifier when building `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each eligible NWB experiment file is treated as one session. The script stores `ophys_session_id` in preview metadata, but it does not group multiple experiment files sharing the same `ophys_session_id`.

ii.
```python
def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        experiment_id = int(decode_scalar(f["/identifier"][()]))
        ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
        ...
        return SessionPreview(
            path=path,
            experiment_id=experiment_id,
            ophys_session_id=ophys_session_id,
            ...
        )
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_sessions.append(neural_trials)
```

iii. Step 2 notes document that the local raw payload is organized as one NWB per experiment and that the local subset contains mostly one experiment per ophys session. The implementation follows that file organization rather than reconstructing multi-experiment sessions.

## 1-d. How are the data split into trials?

i. Trials are split using the NWB `/intervals/trials` table. For each retained trial, the script uses the trial `start_time` and `stop_time` to define the window, then creates a rebinned time grid inside that window.

ii.
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    ...

def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    ...
    specs.append(
        TrialSpec(
            trial_idx=idx,
            start_time=float(trials["start_time"][idx]),
            stop_time=float(trials["stop_time"][idx]),
            change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
            ...
        )
    )
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. Step 1 notes say the SDK defines trials from the behavior `trial_log`, and Step 5 says the agent would “use SDK-valid trials only” and “alignment event = trial start,” so the code uses the trial table’s explicit start/stop boundaries rather than inferring trials from stimulus presentations.

## 1-e. How are trials filtered based on quality controls?

i. The script excludes aborted and auto-rewarded trials. It also rejects sessions with missing eye tracking, fewer than two valid trials, or too few finite pupil samples. It does not explicitly drop trials with missing `change_time`.

ii.
```python
for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
    ...

if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. Step 5 notes say to keep go/catch trials while excluding aborted and auto-rewarded trials, and to exclude sessions with missing eye tracking because pupil diameter is a required target. Trajectory Step 109 explicitly mentions identifying the few files missing eye tracking.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from `/processing/ophys/event_detection/data`, with ROI selection controlled by `/processing/ophys/event_detection/rois` and `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`.

ii.
```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
    )
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
    return event_data, event_rois[valid_mask]
```

iii. Step 5 notes state the key decision explicitly: “Neural signal = event-detection output, not dF/F,” because the analysis paper used discrete calcium events. Trajectory Step 112 repeats that the main design choice was to use event-detection outputs as neural activity.

## 2-b. How is the `neural` data processed?

i. Neural activity is processed by filtering to valid ROIs, then summing event magnitudes into uniform 100 ms bins within each trial window. The output per trial is a neuron-by-time matrix of rebinned event totals.

ii.
```python
BIN_SIZE_SEC = 0.1
...
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

iii. Step 5 says the agent wanted one common bin size across sessions and chose 100 ms as “coarse enough to avoid pathological upsampling of 11 Hz recordings” while still resolving stimulus flashes. Step 6 notes describe this as rebinned neural “calcium-event activity.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered to ROIs marked `valid_roi == True`. No additional amplitude thresholding or session-level neural filtering is applied during conversion.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. Step 1 notes emphasize that the SDK’s cell-specimen path filters invalid ROIs by default. Step 4 lists “Keep SDK `valid_roi` filtering logic” as the resolution for neuron curation when reading the NWBs directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. For each trial, the script builds 100 ms bins spanning `start_time` to `stop_time`, locates the corresponding ophys frames with `np.searchsorted`, and sums events inside each bin.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 5 explicitly says “Alignment event = trial start,” because the trial is treated as the natural unit requested by the user. The metadata written at the end also stores `temporal_alignment_event: "trial start"` and `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins (`BIN_SIZE_SEC = 0.1`). Yes, temporal rebinning is applied to neural activity, image outputs, running speed, and pupil diameter.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"metadata": {
    ...
    "time_bin_size": BIN_SIZE_SEC * 1000.0,
    "binning_rule": "sum event magnitudes within each 100 ms trial bin",
    ...
}
```

iii. Step 5 notes justify the choice as a common bin size across mixed 11 Hz and 31 Hz recordings, and Step 6 notes say the common 100 ms time base was used to satisfy the shared-bin-size requirement.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation tables under `/intervals/*_presentations`, specifically the `image_name`, `start_time`, `stop_time`, and `omitted` fields, not from the trial table’s `initial_image_name` / `change_image_name`.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
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

iii. Step 5 maps “Stimulus presentation `image_name` + presentation timing + omission state” to `image_identity`, because the agent wanted the output to reflect the actual flashed image over time rather than only the trial table’s initial/change labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script collects the unique non-empty stimulus image names globally, prepends a `gray` class, and then projects stimulus presentations onto each trial’s 100 ms bin centers. Bins default to `gray`; omitted flashes also stay `gray`.

ii.
```python
def unique_nonempty_images(presentations: Dict[str, np.ndarray]) -> List[str]:
    names = [
        str(x)
        for x in presentations["image_name"]
        if str(x) not in {"", "nan", "None", "omitted"}
    ]
    return sorted(set(names))

def make_image_series(...):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    ...
    omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. Step 5 says “Image identity will include a `gray` class” because trials contain gray inter-stimulus intervals and omission periods. Step 10 notes mention a later fix so that `omitted` would not remain as a stray class label in `output_values`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by evaluating which stimulus presentation overlaps each trial’s 100 ms bin center. The resulting `image_series` uses the same per-trial time axis as the neural data.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
        ...
    ]
)
```

iii. Step 10 says the processing plots rechecked that “stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window.”

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag together with `start_time`, `stop_time`, and `omitted`, rather than from the trial table’s `change_time` and `go` fields.

ii.
```python
keep_names = [
    "start_time",
    "stop_time",
    "image_name",
    "omitted",
    "is_change",
    ...
]
...
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 maps “Stimulus presentation `is_change` + presentation timing” directly to the `image_change` output so the change label is tied to the changed-image presentation itself.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script initializes a zero vector per trial, finds the presentation rows that overlap that trial, and sets bins to 1 only when their bin centers fall inside a non-omitted presentation with `is_change == True`.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
for idx in row_idx:
    ...
    in_window = (centers >= start) & (centers < stop)
    ...
    is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. Step 5 notes justify this as marking the changed-image presentation rather than a single instant, because that is “more robust after binning and is consistent with the task structure.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric thresholding is applied. The output is directly converted into the two categories `no_change` and `change` from the boolean `is_change` presentation flag.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
...
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. The agent treated image change as inherently binary in the source data, so the only categorization step was mapping absent/present change to 0/1.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned on the same rebinned trial time base as neural activity, using the same 100 ms bin centers and trial window.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 10’s processing-review notes say the stimulus identity/change and neural event traces were verified on the same ophys-aligned trial window.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB arrays `/processing/running/speed/timestamps` and `/processing/running/speed/data`.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 6 notes say the script mirrors the SDK semantics already serialized into the NWBs and uses `/processing/running/speed` for the locomotion signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is pooled across trial windows in the preview pass to estimate global quintile edges, then linearly interpolated to each trial’s 100 ms bin centers and discretized with `np.digitize`.

ii.
```python
running_concat = collect_time_window_values(running_times, running_values, specs)
...
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 5 says the bin edges should be global rather than per-session so output categories are consistent across the dataset. Step 6 notes describe a two-pass workflow to compute those global percentile edges before final conversion.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global equal-percentile bins.

ii.
```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. Step 5 explicitly says “Running and pupil bin edges will be global, not per-session,” to keep one consistent categorical definition across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same 100 ms trial bin centers used for neural rebinning.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 4 notes resolve temporal alignment by resampling synchronized running and eye streams onto an ophys-aligned common binning grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw`, `/acquisition/EyeTracking/likely_blink/data`, and `/acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. Step 5 says the agent chose processed pupil area after blink filtering and converted it to equivalent diameter because the output target is pupil diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to `NaN`, pupil area is converted to an equivalent diameter `2 * sqrt(area / pi)`, pupil values are pooled across trial windows to compute global quintile edges, then values are linearly interpolated to 100 ms trial bin centers and discretized.

ii.
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
...
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Step 5 notes justify this by saying the whitepaper describes pupil size in geometric terms, and Step 6 says sessions without usable eye tracking are excluded rather than fabricating pupil values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global equal-percentile bins.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
]
```

iii. Step 5 says the global-percentile rule used for running speed should also be used for pupil so the categories are comparable across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same 100 ms trial bin centers used for the neural matrices.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 4 notes say synchronized eye-tracking time bases were resampled onto the ophys-aligned common binning grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. Step 5 notes map these trial outcome flags directly to the output and say outcome will be repeated across bins to keep outputs uniformly time-varying.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four boolean fields are converted to an integer index 0-3 by position in `TRIAL_OUTCOME_VALUES`, and that single code is repeated across all time bins in the trial.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Step 5 note 11 says trial outcome is static per-trial but is repeated across time bins so every output tensor has shape `(n_output, n_timepoints)`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing data by excluding whole sessions without eye tracking, excluding sessions with too few valid trials or too few finite pupil samples, forcing omitted image flashes to `gray`, raising an error if a retained trial does not have exactly one valid outcome, and rejecting sessions if running or pupil interpolation produces non-finite values. Blink pupil samples are converted to `NaN` before interpolation.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
...
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
...
pupil_area[likely_blink] = np.nan
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. Step 5 and Step 6 justify the session exclusions by saying pupil diameter is a required output and should not be fabricated. Step 10 also records a bug fix where omitted flashes were removed from the image vocabulary and mapped only to `gray`.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the two-pass scan over all NWB files and, within conversion, the per-bin summation of neural event data for every trial. The script opens every file once in the preview pass and again in the conversion pass.

ii.
```python
eligible, excluded = collect_previews(
    files=files,
    bin_size_sec=BIN_SIZE_SEC,
    sample_mode=args.sample,
    required_eligible=2,
)
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. Step 6 notes say the preview pass is intentionally lightweight, but the full preview still scans all NWB files and the conversion pass opens them again. The same notes call out per-bin event summation as a CPU cost accepted to keep peak memory lower.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the inner per-bin neural loop in `convert_session`, which repeatedly slices and sums `event_data`. The stimulus-presentation loop in `make_image_series` and the preview-time concatenation loops are also potential vectorization targets.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
...
for idx in row_idx:
    ...
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. Step 6 notes explicitly say session conversion “currently rebins event data with per-bin slice sums instead of using cumulative sums,” identifying this as a speed tradeoff rather than the most optimized implementation.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats a full-file preview pass before the actual conversion pass. That means trials, stimulus presentations, running speed, pupil arrays, and metadata are read once to determine eligibility and global bin edges, then read again to build final outputs.

ii.
```python
eligible, excluded = collect_previews(...)
...
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. Step 6 notes acknowledge this explicitly: “Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known.”

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs preview-only bookkeeping that is not preserved in the final dataset, such as `raw_trial_count`, `valid_trial_count`, `unique_images`, and pooled `running_values` / `pupil_values` in `SessionPreview`. It also optionally generates diagnostic plots and reads several trial-table columns (`change_time`, `change_frame`, `initial_image_name`, `change_image_name`, `go`, `catch`) that are not used in the final output construction.

ii.
```python
@dataclass
class SessionPreview:
    ...
    running_values: np.ndarray
    pupil_values: np.ndarray
    unique_images: List[str]
    raw_trial_count: int
    valid_trial_count: int
...
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

iii. Step 6 notes frame this as an intentional engineering tradeoff: the preview pass and optional plots exist for validation and for computing global discretization settings, even though much of that intermediate work is not kept in `converted_data.pkl`.
