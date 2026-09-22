# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by scanning local NWB files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, previewing every file with `h5py`, and then reopening eligible files in a second conversion pass. It does not use the AllenSDK cache or experiment table; it reads NWB/HDF5 fields directly.

ii.
```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
...
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
eligible, excluded = collect_previews(
    files=files,
    bin_size_sec=BIN_SIZE_SEC,
    sample_mode=args.sample,
    required_eligible=2,
)
...
with h5py.File(preview.path, "r") as f:
    ...
```

iii. In `CONVERSION_NOTES.md`, the AI says it read the same underlying NWB tables/fields directly with `h5py` because the higher-level path was not usable in the environment, and it chose a preview pass so it could decide session eligibility and compute global image/running/pupil statistics before conversion.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB subject identifier stored in `/general/subject/subject_id`. During conversion, each new `subject_id` is added once to `subjects`, and sessions point into that list via `subject_idx`.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
...
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes describe subject metadata in the NWB files as authoritative for the locally available dataset, so the AI used those identifiers directly rather than reconstructing mice from the AllenSDK experiment table.

## 1-c. How are the data split into sessions?

i. The AI treats each eligible NWB experiment file as one output session. It records `ophys_session_id` during preview, but it does not merge multiple experiments/planes that share the same `ophys_session_id`.

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
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
```

iii. The notes frame the local raw payload as “per-experiment NWB files,” and the implementation keeps that unit all the way through conversion instead of reconstructing multi-plane sessions.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `/intervals/trials` table. For every retained trial, the code uses the trial’s `start_time` and `stop_time` and creates 100 ms bins across that interval. Trials are therefore variable-length in duration but represented on a common 100 ms grid.

ii.
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    ...

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
                change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
                ...
            )
        )
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The notes say the AI kept the SDK-style trial table semantics directly from NWB and chose trial start as the natural unit for conversion, with a common 100 ms time base to satisfy the decoder format.

## 1-e. How are trials filtered based on quality controls?

i. At the trial level, the AI excludes `aborted` and `auto_rewarded` trials and requires exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject` to be true. At the session level, it excludes sessions with missing eye tracking, fewer than two valid trials, or too few finite pupil samples.

ii.
```python
for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
    outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
    if sum(int(x) for x in outcome_flags) != 1:
        raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The notes justify these filters by the task requirements: aborted and auto-rewarded trials must be excluded, pupil diameter is a required output so sessions without usable eye tracking are dropped, and the decoder needs at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the NWB event-detection output, specifically `/processing/ophys/event_detection/data` plus the ROI index array `/processing/ophys/event_detection/rois`, with ROI validity taken from `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`.

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

iii. In the notes, the AI explicitly chose event-detection outputs rather than dF/F because the paper’s analysis uses discrete calcium events, and it wanted the neural representation to match that choice.

## 2-b. How is the `neural` data processed?

i. After loading event-detection magnitudes, the AI filters to valid ROIs and rebins the data into 100 ms trial bins by summing all event magnitudes whose ophys timestamps fall inside each bin.

ii.
```python
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

iii. The notes state that a common 100 ms bin size was chosen to force one shared temporal resolution across sessions while preserving ophys-time alignment. They also note that the code uses per-bin slice sums instead of a faster cumulative-sum implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered only by `valid_roi`. The AI does not apply additional event-amplitude, neuron-activity, or session-level neural QC beyond that ROI validity mask and the session exclusions already described.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes say this mirrors the AllenSDK’s default ROI-quality behavior as closely as possible while still reading raw NWB tables directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. For each trial, the code builds bins from `start_time` to `stop_time`, uses ophys timestamps to decide which frames fall in each bin, and stores the result as a per-trial neuron-by-time matrix.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
...
neural_trials.append(neural_trial)
```

iii. The notes explicitly say “Alignment event = trial start” because the trial is the natural unit for the decoder task, while the actual timing inside each trial remains tied to ophys timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins for all trials and sessions. Yes: the AI rebins native ophys events into this common temporal grid.

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

iii. The notes justify 100 ms as a compromise that is coarse enough to work across mixed native frame rates while still resolving the task structure, and they present the rebinning as required by the desired decoder format.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation tables under `/intervals/*_presentations`, using each presentation’s `image_name`, `start_time`, `stop_time`, and `omitted` flag.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    ...
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
...
omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
if not omitted:
    image_name = str(presentations["image_name"][idx])
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes say the AI intentionally used presentation-level stimulus records rather than only the trial table so it could represent flashes, omissions, and gray periods explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI creates one global image vocabulary across eligible sessions, prepends a `gray` class, initializes every trial bin to `gray`, and overwrites bins that overlap non-omitted presentations with that presentation’s image code.

ii.
```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_name = str(presentations["image_name"][idx])
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes justify this as omission-aware and gray-aware output construction. They also document a later bug fix where an unused `omitted` label was removed from `output_values` because omitted periods were already encoded as `gray`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned to the same 100 ms trial bins used for neural data. The code projects each stimulus presentation onto the trial bin centers and writes the resulting category code into the matching bins.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The notes describe this as synchronized output construction on the same ophys-aligned trial window as the neural data, with no separate post hoc realignment step.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table, specifically the `is_change`, `omitted`, `start_time`, and `stop_time` fields of the presentation rows that overlap each trial.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes say the AI preferred presentation-level change annotations because they directly mark the changed-image flash rather than reconstructing it from trial-level metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector for the trial, finds presentation rows overlapping the trial, and sets bins to `1` wherever a non-omitted presentation row has `is_change == True`.

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

iii. The notes characterize this as a direct projection of changed-image presentations onto the rebinned trial axis.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary from the start: `0` means no change and `1` means change. No further thresholding is applied.

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
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The output label names in the final data dictionary show that the intended categories are exactly `["no_change", "change"]`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned to the same trial bins and bin centers as the neural data, because it is created alongside image identity on the rebinned trial axis.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
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

iii. The notes say stimulus identity/change, running, pupil, and neural event traces are all synchronized on the same ophys-aligned trial window.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running stream: `/processing/running/speed/timestamps` and `/processing/running/speed/data`.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes identify the Allen running-speed stream as the authoritative locomotion source and use it directly from the NWB files.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. In preview, the AI pools raw running samples that fall inside valid trial windows to compute global percentile edges. In conversion, it linearly interpolates running speed to the 100 ms trial bin centers and discretizes the interpolated values using those global edges.

ii.
```python
running_concat = collect_time_window_values(running_times, running_values, specs)
...
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The notes say the global-bin strategy keeps one categorical definition across all sessions, and the preview pooling was chosen as a cheaper way to estimate those edges without interpolating every trial twice.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is split into five global percentile bins, labeled `q1` through `q5`.

ii.
```python
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
...
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)
...
def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. The notes explicitly say the bin edges are global percentile edges so that one consistent 5-level output is used across the whole dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same 100 ms trial bin centers used for neural activity.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes frame this as synchronization onto a common ophys-aligned trial axis shared by all outputs and the rebinned neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from raw eye-tracking area and blink flags: `/acquisition/EyeTracking/eye_tracking/timestamps`, `/acquisition/EyeTracking/pupil_tracking/area_raw`, and `/acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes say the AI used blink-filtered eye-tracking data directly from NWB and chose a diameter representation derived from the tracked pupil area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI masks blink frames to `NaN`, converts pupil area to an equivalent diameter using `2*sqrt(area/pi)`, pools within-trial values in preview to compute global percentile edges, interpolates the diameter trace to trial bin centers during conversion, and discretizes with those edges.

ii.
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=np.float64)
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

iii. The notes justify excluding sessions without eye tracking because pupil diameter is a required decoder target, and they describe the blink masking plus diameter conversion as their chosen preprocessing.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global percentile bins, labeled `q1` through `q5`.

ii.
```python
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes say the running and pupil outputs share the same global-percentile discretization design so the output categories are consistent across all sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same 100 ms trial bin centers used for neural data.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes describe this as shared trial-window alignment across all streams after resampling onto the common rebinned axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes say these are the canonical task outcomes from the SDK/NWB trial logic, and the code enforces that exactly one of them must hold for each retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the one-hot outcome flags into an integer index stored in `TrialSpec.outcome_idx`, then repeats that single class across all time bins of the trial when building the output matrix.

ii.
```python
specs.append(
    TrialSpec(
        ...
        outcome_idx=outcome_idx,
        ...
    )
)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. The notes justify repeating the trial outcome across time so every output trial has a uniform `(n_output, n_timepoints)` structure even for static labels.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye-tracking data causes the whole session to be excluded. Blink-contaminated pupil samples are set to `NaN` and then interpolated over when enough samples remain. Sessions with too few valid pupil samples are excluded. Interpolation extrapolates using endpoint values rather than leaving boundary `NaN`s. The notes also document one conversion bug that was fixed later: an unused `omitted` image label was removed from `output_values`.

ii.
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
...
pupil_area[likely_blink] = np.nan
...
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The notes justify these choices pragmatically: pupil is a required output, so sessions without usable pupil traces are dropped rather than imputed wholesale; blink frames are treated as artifacts; and the later `omitted`-label fix came from a raw-vs-converted sanity check.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s code spends most of its time in two places: the preview pass over all NWB files and the per-session conversion pass that loads full event matrices and sums events into 100 ms bins. The notes specifically call out per-bin event summation as extra CPU work.

ii.
```python
eligible, excluded = collect_previews(...)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
...
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the preview pass was designed to avoid loading neural event matrices, but the full conversion still has to read them and aggregate them bin by bin, which is where a large share of runtime goes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the inner per-bin neural loop that sums event slices for each 100 ms bin. Secondary candidates are the presentation loop in `make_image_series` and the preview loop that concatenates running/pupil segments across trial windows.

ii.
```python
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
...
for spec in trial_specs:
    lo = np.searchsorted(times, spec.start_time, side="left")
    hi = np.searchsorted(times, spec.stop_time, side="left")
    if hi > lo:
        segments.append(values[lo:hi])
```

iii. The notes explicitly mention that per-bin slice summation could be replaced by a cumulative-sum style implementation for speed, but the author left the simpler loop in place to keep memory use lower.

## 9-c. What processing does the code repeat multiple times?

i. The code uses a two-pass pipeline, so it reopens each eligible NWB file at least twice: once during preview and again during conversion. It also rereads trial tables and stimulus-presentation tables in both passes.

ii.
```python
eligible, excluded = collect_previews(...)
...
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The notes present this as intentional: preview computes eligibility and global discretization state cheaply, then conversion does the heavy extraction. Even so, it is repeated work relative to a single-pass design.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preview pass computes and stores several fields used only for filtering, logging, or optional plots, not for the saved dataset: `change_time`, `is_go`, `is_catch`, raw/valid trial counts, and optional plotting summaries. The code also does optional visualization work and constructs preview-only summaries that are discarded after conversion.

ii.
```python
@dataclass
class TrialSpec:
    trial_idx: int
    start_time: float
    stop_time: float
    change_time: float
    outcome_idx: int
    is_go: bool
    is_catch: bool
...
@dataclass
class SessionPreview:
    ...
    raw_trial_count: int
    valid_trial_count: int
    excluded_reason: str | None = None
...
if show_processing:
    plot_processing_summary(...)
```

iii. The notes acknowledge that preview work is partly there to support sanity checks and debugging. That is useful during development, but much of it does not survive into `converted_data.pkl`.
