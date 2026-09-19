# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all available local NWB experiment files from `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` with `h5py`, then does a two-pass workflow: a preview pass to collect eligibility and global binning statistics, and a conversion pass to build the final dataset. It does not use the AllenSDK cache or experiment table in the final converter.

ii. ```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
eligible, excluded = collect_previews(
    files=files,
    bin_size_sec=BIN_SIZE_SEC,
    sample_mode=args.sample,
    required_eligible=2,
)
```

iii. In `CONVERSION_NOTES.md`, the AI says it used direct HDF5 reads because the installed NWB stack was incompatible in this environment, and that direct reads still expose the same serialized NWB tables as the SDK.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB field `/general/subject/subject_id`. The final `subjects` list is created in first-seen order across included sessions, with a `subject_to_idx` map used to populate `subject_idx`.

ii. ```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])

if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
...
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes describe subject identifiers as coming from NWB subject metadata and being carried through as session-level subject indices.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one session. The code records `ophys_session_id`, but it does not group multiple experiments that share the same `ophys_session_id`; each eligible preview object becomes one output session.

ii. ```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for idx, path in enumerate(files, start=1):
    preview = session_preview(path, bin_size_sec)
    if preview.excluded_reason is None:
        eligible.append(preview)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
```

iii. The notes frame the local NWB file as the working conversion unit and justify that choice as a direct-file workaround rather than reconstructing AllenSDK multi-experiment sessions.

## 1-d. How are the data split into trials?

i. Trials are split from the NWB `/intervals/trials` table. For each included trial row, the code uses `start_time` and `stop_time` as the trial window and later rebins that interval into 100 ms bins.

ii. ```python
trials = read_trials(f)
specs = build_trial_specs(trials)
...
TrialSpec(
    trial_idx=idx,
    start_time=float(trials["start_time"][idx]),
    stop_time=float(trials["stop_time"][idx]),
    change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
    ...
)
...
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The AI repeatedly states in its notes that it is mirroring SDK trial semantics already serialized in the NWB trial table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `aborted` and `auto_rewarded` rows and by requiring exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject` to be true. Separately, sessions are excluded if eye tracking is missing, if they have fewer than two valid trials, or if they have too few finite pupil samples. The code does not explicitly require non-missing `change_time`.

ii. ```python
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

iii. The notes justify dropping no-eye-tracking sessions because pupil diameter is a required decoder target, and justify the outcome checks as trial-validity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection matrix, not dF/F. The code loads `/processing/ophys/event_detection/data`, uses `/processing/ophys/event_detection/rois` for ROI indexing, filters by `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`, and uses dF/F timestamps as the ophys frame times.

ii. ```python
ophys_timestamps = np.asarray(
    f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
)
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. `CONVERSION_NOTES.md` explicitly says the key neural decision was to use event-detection outputs instead of dF/F because the paper analyzed discrete calcium events.

## 2-b. How is the `neural` data processed?

i. Neural events are rebinned into fixed 100 ms trial bins. For each bin, the code sums all event magnitudes from ophys frames whose timestamps fall inside the bin. There is no additional normalization after that.

ii. ```python
BIN_SIZE_SEC = 0.1
...
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes justify the common 100 ms time base as a way to standardize across single-plane and multi-plane recordings while keeping ophys-timestamp alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC in the converter is ROI filtering with `valid_roi`. Neurons from invalid ROIs are removed before trial extraction. The AI kept all remaining event traces, including trials that are entirely zero after event extraction.

ii. ```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes cite SDK behavior around `valid_roi` and argue that this reproduces the SDK’s ROI curation when reading raw NWB tables directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to the trial start and represented over the whole trial window from `start_time` to `stop_time`. Bin membership is determined against absolute ophys timestamps with `np.searchsorted`.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The notes say the “alignment event” is `trial start`, with `off_start = 0.0`, because the trial is the natural unit requested by the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins, and the native ophys frames are explicitly rebinned into those bins.

ii. ```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The notes explicitly justify 100 ms as a compromise that avoids pathological upsampling of 11 Hz recordings while still resolving 250 ms stimulus flashes.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation tables under `/intervals/*_presentations`, specifically `image_name`, `start_time`, `stop_time`, and `omitted`.

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
...
image_name = str(presentations["image_name"][idx])
```

iii. The notes say the AI preferred projecting the actual presentation table onto trial bins so the output reflects what was on screen, including gray periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds a global categorical vocabulary consisting of `"gray"` plus all non-empty, non-`"omitted"` image names found in included sessions. Each trial starts as `"gray"` everywhere, then bins whose centers fall inside non-omitted stimulus presentations are overwritten with the presented image code.

ii. ```python
names = [
    str(x)
    for x in presentations["image_name"]
    if str(x) not in {"", "nan", "None", "omitted"}
]
...
image_values = ["gray"] + image_names
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
if not omitted:
    image_name = str(presentations["image_name"][idx])
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes explicitly justify adding a gray class so every trial bin has a defined image-identity label, including inter-stimulus gray periods and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is projected onto the same 100 ms trial bins used for neural data. A bin gets a stimulus label if its center lies within a presentation interval; otherwise it stays gray.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The notes describe this as projecting stimulus presentations onto the shared ophys-aligned trial time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, together with `start_time`, `stop_time`, and `omitted`.

ii. ```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes say the code uses presentation-table change flags rather than reconstructing change windows from trial-table `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero series and sets bins to 1 when their centers fall inside a non-omitted changed-image presentation interval.

ii. ```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes describe this as a time-varying binary output marking the changed presentation itself.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is represented as a binary categorical variable with values `0` and `1`, later named `["no_change", "change"]`.

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The AI treated image change as an already discrete signal, so no thresholding beyond binary coding was applied.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned exactly like image identity: it is projected onto the same 100 ms trial bins that index the rebinned neural matrix.

ii. ```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    ...
])
```

iii. The notes repeatedly describe a single shared trial-relative time base for all outputs and neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `/processing/running/speed/timestamps` and `/processing/running/speed/data`.

ii. ```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes identify these datasets as the direct NWB equivalent of the SDK running-speed object.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI first pools running-speed samples from valid trial windows across eligible sessions to compute global quantile edges. During conversion, it linearly interpolates running speed to the 100 ms trial-bin centers with `np.interp` and discretizes the interpolated values using those global edges.

ii. ```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The notes justify global percentile binning for consistent categories across sessions and the shared-bin interpolation as part of the 100 ms common time base.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five quantile bins using global percentile cut points computed over all included trial-window running samples.

ii. ```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

running_edges = compute_bin_edges(running_pool, 5)
```

iii. The notes explicitly say the five bins are global percentile bins, not session-specific bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same 100 ms bin centers that index the rebinned neural activity.

ii. ```python
running_interp = interpolate_series(running_times, running_values, centers)
...
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins,
    ...
])
```

iii. The notes describe all streams as being synchronized and then represented on one common trial-relative time axis.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking timestamps, the raw pupil area signal, and the blink mask: `/acquisition/EyeTracking/eye_tracking/timestamps`, `/acquisition/EyeTracking/pupil_tracking/area_raw`, and `/acquisition/EyeTracking/likely_blink/data`.

ii. ```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes say the AI used processed pupil area after blink masking and converted it to an equivalent diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to `NaN`, pupil area is converted to diameter with `2*sqrt(area/pi)`, trial-window pupil samples are pooled to compute global quantile edges, then the pupil diameter is interpolated to 100 ms trial-bin centers and discretized.

ii. ```python
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
...
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes justify excluding missing-eye-tracking sessions because pupil is mandatory, and justify diameter-from-area as matching the whitepaper discussion of pupil geometry.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins using quantile edges computed from pooled included-session trial-window pupil samples.

ii. ```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes describe this as the pupil analogue of the running-speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same 100 ms trial-bin centers used for neural data.

ii. ```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins,
    pupil_bins,
    outcome_series,
])
```

iii. The notes describe pupil interpolation onto the same ophys-aligned trial window used for all other streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
```

iii. The notes describe these as the SDK-defined outcome fields serialized into the NWB trial table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. For each kept trial, the code requires exactly one outcome flag to be true, converts that outcome to an integer index, and repeats the same index across every time bin in the trial.

ii. ```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. The AI treated trial outcome as a static per-trial label that is broadcast over the trial’s time bins for uniform output shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or messy data by:
- excluding sessions with no eye tracking or too little finite pupil data;
- masking blink-contaminated pupil samples with `NaN`;
- using constant fill when interpolation queries fall outside the observed time range;
- filling a whole query with `NaN` if a series has no finite points, which then causes session exclusion or an error;
- raising an error if a kept trial does not have exactly one outcome;
- accepting all-zero neural trials as real sparse-event trials.

ii. ```python
if finite.sum() == 0:
    return np.full(query_times.shape, np.nan, dtype=np.float32)
if finite.sum() == 1:
    v = float(values[finite][0])
    return np.full(query_times.shape, v, dtype=np.float32)
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
```

iii. The notes justify missing-eye-tracking exclusion as necessary because pupil is required, and explicitly say the remaining all-zero neural warnings were investigated and kept because they matched the raw event matrices.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify the two-pass file scan as a major cost: first previewing all NWB files, then reopening them for conversion. Within conversion, the per-bin event summation loop is the main CPU-heavy step.

ii. ```python
eligible, excluded = collect_previews(...)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

iii. `CONVERSION_NOTES.md` explicitly says the full preview scans all NWB files once and the conversion pass opens them again, and that event rebinning by per-bin slice sums trades memory for extra CPU.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization target is the inner per-bin loop that sums event data one bin at a time. The presentation-projection loop is another candidate. The notes also mention cumulative-sum style alternatives for event rebinning.

ii. ```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)

for idx in row_idx:
    ...
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. The notes explicitly say the current per-bin event summation could be replaced by a more efficient approach, but was left as-is to reduce peak memory.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several operations across the preview and conversion passes: opening every eligible NWB file, reading the trials table, reading presentation tables, and reading running/pupil streams once for preview statistics and again for actual conversion.

ii. ```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The notes explicitly describe the repeated preview-plus-conversion pattern as intentional so global image/bin metadata can be computed without storing full neural matrices.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads or computes several things that are not needed in the final saved dataset: `TrialSpec.trial_idx`, `TrialSpec.change_time`, and `TrialSpec.is_catch`; presentation columns such as `is_sham_change`, `trials_id`, and `active`; optional processing plots; and `session_stats`, which are accumulated but never written into the pickle.

ii. ```python
class TrialSpec:
    trial_idx: int
    start_time: float
    stop_time: float
    change_time: float
    outcome_idx: int
    is_go: bool
    is_catch: bool

keep_names = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "is_sham_change", "trials_id", "active",
]
...
session_stats.append(stats)
```

iii. The notes describe `--show-processing` as a diagnostics-only path and otherwise focus on the saved neural/input/output arrays, confirming that these extra artifacts are not used downstream.
