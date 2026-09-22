# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the AllenSDK cache or experiment table. It discovers data by globbing local NWB files under a hard-coded experiment directory, runs a preview pass over those files, and then reopens each eligible file in a second conversion pass.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
eligible, excluded = collect_previews(...)

with h5py.File(preview.path, "r") as f:
    ...
```

iii. The justification in `CONVERSION_NOTES.md` is that direct HDF5 reads were used because the installed NWB stack was incompatible in that environment, and that a two-pass workflow was needed to identify eligible sessions and compute global discretization edges before full conversion.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB subject metadata field `/general/subject/subject_id`. A unique string subject list is built in first-seen order, and each converted session stores an index into that list.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])

if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)

subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The code does not give a long explicit defense here; the implied justification is that `subject_id` is the canonical per-file mouse identifier available in the NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one session. The code reads `ophys_session_id`, but it never groups multiple experiment files that share the same session id.

ii.
```python
experiment_id = int(decode_scalar(f["/identifier"][()]))
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])

eligible, excluded = collect_previews(files=files, ...)

for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
```

iii. `CONVERSION_NOTES.md` justifies direct per-file processing as a practical local-NWB workflow and as a way to process one session at a time for memory control. It does not justify not merging files with the same `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from `/intervals/trials`. For every non-aborted, non-auto-rewarded row, the code records `start_time`, `stop_time`, and metadata in a `TrialSpec`, then later converts the whole `start_time` to `stop_time` interval into 100 ms bins.

ii.
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    ...

for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
    specs.append(
        TrialSpec(
            trial_idx=idx,
            start_time=float(trials["start_time"][idx]),
            stop_time=float(trials["stop_time"][idx]),
            ...
        )
    )

starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The justification in the notes is that the conversion should mirror the SDK semantics already serialized into the NWB trial table and include the full trial interval rather than a fixed event window.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filtering excludes only `aborted` and `auto_rewarded` rows. The code also requires exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject` to be true. Session-level filtering excludes sessions with missing eye tracking, fewer than two valid trials, or too few finite pupil samples. There is no explicit `change_time.notna()` filter and no explicit empty-window check.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue

outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")

if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The explicit justification in the notes is that pupil diameter is a required decoder target, so sessions without eye tracking are excluded, and at least two valid trials are needed for downstream decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection matrix, plus the event ROI index array and the `valid_roi` mask. The code uses the dF/F timestamps only as the ophys time base for alignment.

ii.
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
ophys_timestamps = np.asarray(
    f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
)
```

iii. The notes explicitly justify this as a paper-driven choice: the paper’s analyses used discrete calcium events, so the agent chose event-detection outputs instead of dF/F traces.

## 2-b. How is the `neural` data processed?

i. The event matrix is filtered to valid ROIs, then each trial is rebinned into 100 ms bins by summing all event magnitudes whose ophys timestamps fall inside each bin. Because each experiment file is treated as a session, no multi-plane session merge is performed.

ii.
```python
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]

for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
    frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
    frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
    neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
    for b in range(T):
        lo = int(frame_starts[b])
        hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. The notes justify this with two points: use event-detection outputs because they are closer to the paper, and use a common 100 ms trial-relative time base so every session shares one bin size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC filter is `valid_roi`. Invalid ROIs are dropped before any trial extraction.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes say this is meant to mirror the AllenSDK behavior, which filters invalid ROIs at load time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to trial start. The binning window begins at `start_time`, ends at `stop_time`, and neural counts are assigned according to ophys timestamps falling in each 100 ms bin.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")

"metadata": {
    "temporal_alignment_event": "trial start",
    "off_start": 0.0,
    "off_end": None,
}
```

iii. The notes explicitly say the trial itself is the natural unit for this decoder task, so the alignment event is `trial start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. Yes, temporal rebinning is applied: neural event magnitudes are summed into 100 ms trial-relative bins.

ii.
```python
BIN_SIZE_SEC = 0.1

def build_bin_centers(start_time: float, stop_time: float, bin_size_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...

"metadata": {
    "time_bin_size": BIN_SIZE_SEC * 1000.0,
    "binning_rule": "sum event magnitudes within each 100 ms trial bin",
}
```

iii. The notes justify 100 ms as a common bin width across sessions that still respects ophys-time alignment and keeps 250 ms stimulus flashes resolvable.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `image_identity` is derived from the stimulus-presentation tables, not from the trial table. The code uses `image_name`, `start_time`, `stop_time`, and `omitted` from `/intervals/*_presentations`.

ii.
```python
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

def make_image_series(...):
    ...
    omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes justify this by saying stimulus presentations should be projected directly onto the converted time bins, with an explicit `gray` state outside image flashes or during omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Every trial bin is initialized to `gray`. For each overlapping presentation interval, non-omitted bins are set to that presentation’s `image_name`. A global label vocabulary is built as `["gray"] + sorted(unique_images)`, excluding empty strings and `omitted`.

ii.
```python
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_name = str(presentations["image_name"][idx])
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])

def unique_nonempty_images(presentations: Dict[str, np.ndarray]) -> List[str]:
    names = [
        str(x)
        for x in presentations["image_name"]
        if str(x) not in {"", "nan", "None", "omitted"}
    ]
    return sorted(set(names))
```

iii. The notes explicitly justify the `gray` class as a way to keep the categorical output defined during inter-stimulus intervals and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The code finds all presentation rows that overlap the trial window, then labels 100 ms trial bins according to whether each bin center lies inside a stimulus presentation interval. Those bins are on the same rebinned trial grid as the neural data.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)

neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
output_trial = np.vstack([
    image_series.astype(np.int16),
    ...
])
```

iii. The notes say all converted streams should share the same ophys-anchored 100 ms trial window, so stimulus identity is projected onto the same bin centers used for neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the stimulus-presentation table fields `is_change`, `start_time`, `stop_time`, and `omitted`. The code reads `is_sham_change` but does not use it.

ii.
```python
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

is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes justify this as marking the changed-image presentation itself from the presentation table rather than reconstructing change timing from trial metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code creates a zero vector for the trial and sets bins to `1` wherever the bin center falls inside a non-omitted presentation interval whose `is_change` flag is true.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
in_window = (centers >= start) & (centers < stop)
...
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes describe this as a direct projection of the changed-image presentation onto the rebinned trial timeline.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous signal. The code emits integer `0/1` labels directly and names those categories `["no_change", "change"]`.

ii.
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
change_series[in_window] = 1

"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
]
```

iii. No additional justification was needed beyond the notes’ statement that image change is meant to be a binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned exactly like `image_identity`: presentation intervals overlapping the trial are projected onto the same 100 ms trial bins used for neural data.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    ...
])
```

iii. The notes justify a single shared rebinned timeline for neural and behavioral/stimulus outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `running_speed` is derived from the NWB running-speed series: `/processing/running/speed/timestamps` and `/processing/running/speed/data`.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes say this mirrors the running-speed signal already stored in the NWB files rather than recomputing wheel speed from lower-level raw signals.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the 100 ms trial-bin centers. Global bin edges are computed from pooled within-trial running samples gathered in the preview pass, and each trial is discretized with those edges.

ii.
```python
def interpolate_series(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)

running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_edges = compute_bin_edges(running_pool, 5)

running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The notes explicitly justify the two-pass design so that global running-speed percentile bins can be defined once and then reused consistently across all converted sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into five global quantile bins using four internal quantile edges; the stored output labels are `q1` through `q5`.

ii.
```python
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]

def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)

def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    bins = np.clip(bins, 0, len(edges))
    return bins.astype(np.int16)
```

iii. The notes describe these as global percentile bins over included sessions and trials.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same 100 ms trial-bin centers used for neural data.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
running_interp = interpolate_series(running_times, running_values, centers)

neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
output_trial = np.vstack([
    ...,
    running_bins,
    ...
])
```

iii. The notes justify this with the shared synchronized time base: everything is projected onto one ophys-anchored trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `pupil_diameter` is derived from eye-tracking timestamps, raw pupil area, and the blink mask: `/acquisition/EyeTracking/eye_tracking/timestamps`, `/acquisition/EyeTracking/pupil_tracking/area_raw`, and `/acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes justify this by saying the conversion should use processed pupil area from the NWB eye-tracking stream and convert it to an equivalent diameter, while excluding sessions with no eye tracking because pupil is a required output.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to `NaN`, pupil area is converted to equivalent diameter `2 * sqrt(area / pi)`, the result is interpolated to the 100 ms trial-bin centers, and then discretized with global five-bin quantile edges.

ii.
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)

pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)

pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes explicitly justify blink filtering, a diameter representation, and global percentile bins computed in the preview pass.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five global quantile bins with labels `q1` through `q5`.

ii.
```python
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. The notes justify the same percentile-bin strategy used for running speed so category definitions are consistent across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same 100 ms trial-bin centers used for neural data.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
output_trial = np.vstack([
    ...,
    pupil_bins,
    outcome_series,
])
```

iii. The notes say all converted streams are synchronized by projecting them onto the same ophys-anchored trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)

outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The notes treat these as the SDK-style canonical trial outcomes already serialized in the NWB trial table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code converts the four outcome booleans into a single integer class index at trial-build time and then repeats that class for every time bin in the trial.

ii.
```python
outcome_idx = outcome_flags.index(True)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)

output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins,
    pupil_bins,
    outcome_series,
])
```

iii. The notes justify making trial outcome time-varying only in the sense that every output array should share one `(n_output, T)` format, so the static trial label is repeated across bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases, but unevenly. It decodes string/byte fields defensively, excludes sessions with missing eye tracking, sets blink-contaminated pupil samples to `NaN`, uses finite-only interpolation, and rejects sessions whose interpolated running or pupil values remain non-finite. It also excludes sessions with too few valid pupil samples. It does not explicitly filter missing `change_time`, and a zero-length trial can still produce one bin because `build_bin_centers()` inserts one start bin when the range is empty.

ii.
```python
if isinstance(value, bytes):
    return value.decode("utf-8")

if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"

pupil_area[likely_blink] = np.nan

if finite.sum() == 0:
    return np.full(query_times.shape, np.nan, dtype=np.float32)
if finite.sum() == 1:
    v = float(values[finite][0])
    return np.full(query_times.shape, v, dtype=np.float32)

if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)

if starts.size == 0:
    starts = np.array([start_time], dtype=np.float64)
```

iii. The notes explicitly justify excluding sessions when pupil is missing because pupil is a required decoder target. Other edge handling is mostly implicit in the code rather than strongly justified in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. The code’s expensive stages are the full preview pass over all NWB files and the per-session conversion pass, especially the neural rebinning loop that sums event data bin by bin for every trial.

ii.
```python
eligible, excluded = collect_previews(...)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)

for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says the code scans all files in preview, then opens them again in conversion, and that per-bin slice sums were kept intentionally because they reduce peak memory even though they cost extra CPU.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the inner per-bin neural summation loop in `convert_session()`. The preview helper `collect_time_window_values()` and the per-presentation loop in `make_image_series()` are also straightforward candidates.

ii.
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)

for spec in trial_specs:
    lo = np.searchsorted(times, spec.start_time, side="left")
    hi = np.searchsorted(times, spec.stop_time, side="left")
    if hi > lo:
        segments.append(values[lo:hi])

for idx in row_idx:
    ...
```

iii. The notes explicitly acknowledge one such opportunity: session conversion “currently rebins event data with per-bin slice sums instead of using cumulative sums.”

## 9-c. What processing does the code repeat multiple times?

i. The code repeats file I/O and metadata extraction across two passes. Preview opens every file to read trial/presentation/running/pupil metadata and compute global pools; conversion opens each eligible file again and rereads trials, presentations, running, and pupil before building outputs.

ii.
```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ophys_timestamps = ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
    running_times = ...
    pupil_times = ...
```

iii. The notes justify this repetition as an intentional two-pass design: first define eligibility and global bin edges without loading neural matrices, then perform the heavier full conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and carries some fields that are not used in the final dataset: `ophys_session_id`, `trial_idx`, `change_time`, `is_catch`, `raw_trial_count`, and `is_sham_change`. It also supports optional processing plots and per-session stats that are not part of the saved decoder dataset. `MANIFEST_PATH` is stored in metadata only as a filename string and is otherwise unused.

ii.
```python
class TrialSpec:
    trial_idx: int
    ...
    change_time: float
    ...
    is_catch: bool

class SessionPreview:
    ...
    ophys_session_id: int
    raw_trial_count: int

keep_names = [
    ...,
    "is_sham_change",
    ...
]

if show_processing:
    plot_processing_summary(...)
```

iii. There is little explicit justification for these extras beyond debugging, auditability, and optional visualization support described in the notes.
