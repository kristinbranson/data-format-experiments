# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data by scanning all local NWB experiment files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, then opening each file directly with `h5py`. It does a first "preview" pass to decide session eligibility and collect global statistics, then reopens eligible files in a second conversion pass to read the full per-trial neural and behavioral streams.

ii. ```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

def collect_previews(
    files: List[Path],
    bin_size_sec: float,
    sample_mode: bool,
    required_eligible: int,
) -> Tuple[List[SessionPreview], List[SessionPreview]]:
    eligible: List[SessionPreview] = []
    excluded: List[SessionPreview] = []
    for idx, path in enumerate(files, start=1):
        preview = session_preview(path, bin_size_sec)
        ...

with h5py.File(preview.path, "r") as f:
    ophys_timestamps = np.asarray(
        f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
    )
    event_data, _ = load_neural_events(f)
    running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. In `CONVERSION_NOTES.md`, the agent says it used direct HDF5 reads because the installed NWB stack was incompatible in this environment, and that the two-pass design was intentional so it could compute global image vocabularies and running/pupil quantile edges before materializing the full converted dataset.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each file’s NWB subject metadata (`/general/subject/subject_id`). The agent builds a unique `subjects` list and a per-session `subject_idx` mapping during the conversion loop.

ii. ```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
subjects: List[str] = []
subject_to_idx: Dict[str, int] = {}
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
...
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. The notes describe `/general/subject/subject_id` as the authoritative mouse identifier and say session order in `subject_idx` follows the order of converted sessions.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB behavior-ophys experiment file as one converted session. It records both `experiment_id` and `ophys_session_id` in previews, but the unit carried into `neural`, `input`, and `output` is the per-file experiment, not a grouped multi-plane ophys session.

ii. ```python
@dataclass
class SessionPreview:
    path: Path
    experiment_id: int
    ophys_session_id: int
    ...

experiment_id = int(decode_scalar(f["/identifier"][()]))
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```

iii. In Step 1 of the notes, the agent explicitly wrote that "One experiment corresponds to one imaging plane in one session" and used that as the reason to keep per-experiment granularity instead of grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`/intervals/trials`). For each row, the agent uses the stored `start_time`, `stop_time`, `change_time`, and outcome flags to create one `TrialSpec`, then converts each kept `TrialSpec` into one trial matrix in the output lists.

ii. ```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    names = [
        "id",
        "start_time",
        "stop_time",
        ...
        "change_time",
        ...
    ]
    return read_interval_group(group, names)

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
```

iii. The notes say the agent intentionally mirrored AllenSDK trial semantics already serialized into the NWB files, rather than reconstructing trial boundaries from presentations or image changes.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes `aborted` and `auto_rewarded` trials, keeps only trials with exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject`, and later requires sessions to have at least two valid trials. It keeps both `go` and `catch` trials.

ii. ```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
auto_rewarded = np.nan_to_num(trials["auto_rewarded"], nan=0.0).astype(bool)
...
for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
    outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
    if sum(int(x) for x in outcome_flags) != 1:
        raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
...
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
```

iii. This matches the task wording in the instructions, and the notes explicitly state "Use SDK-valid trials only" and "Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the event-detection matrix at `/processing/ophys/event_detection/data`, indexed through `/processing/ophys/event_detection/rois`, and filtered with `valid_roi` from the cell specimen table. Timing comes from `/processing/ophys/dff/traces/timestamps`.

ii. ```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
    )
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
    return event_data, event_rois[valid_mask]
...
ophys_timestamps = np.asarray(
    f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
)
```

iii. The notes say the key design choice was to use event-detection outputs rather than dF/F because the paper’s analyses used discrete calcium events, while still preserving SDK-style ROI filtering and ophys timestamps.

## 2-b. How is the `neural` data processed?

i. The agent does not recompute dF/F or smooth the events. It rebins event-detection amplitudes into 100 ms trial bins by summing all event values whose ophys timestamps fall inside each bin, producing an `(n_neurons, n_timepoints)` matrix per trial.

ii. ```python
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

iii. The notes justify this as a decoder-format adaptation: the source processing is event detection from the NWB, and the extra step is only the common binning required to give every trial/session the same time-bin semantics.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered only by ROI validity. Any ROI with `valid_roi == False` is removed before conversion. The agent does not apply extra neuron-level thresholds on activity or event counts.

ii. ```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The notes tie this directly to AllenSDK’s default ROI filtering and explicitly say there was no analogous electrophysiology-style unit QC path because this is calcium imaging, not ephys.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to each trial’s absolute `start_time`, but sampled using the shared ophys timestamp clock. For each trial, the agent creates trial-relative 100 ms bins spanning `start_time` to `stop_time` and sums ophys event samples inside those windows.

ii. ```python
def build_bin_centers(start_time: float, stop_time: float, bin_size_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ...
    centers = starts + 0.5 * widths
    return starts, ends, centers
...
"metadata": {
    ...
    "temporal_alignment_event": "trial start",
    "off_start": 0.0,
    "off_end": None,
    ...
}
```

iii. The notes say the agent chose "trial start" as the alignment event because the requested unit of analysis was the trial, while still keeping all streams aligned on the ophys time base.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins (`BIN_SIZE_SEC = 0.1`). This is an added temporal rebinning step: native ophys, running, and pupil streams are all projected onto this coarser common trial-relative grid.

ii. ```python
BIN_SIZE_SEC = 0.1
...
"metadata": {
    ...
    "time_bin_size": BIN_SIZE_SEC * 1000.0,
    ...
    "binning_rule": "sum event magnitudes within each 100 ms trial bin",
}
```

iii. In the notes the agent says 100 ms was chosen to avoid pathological upsampling of 11 Hz multi-plane data, while still resolving 250 ms image flashes and staying compatible with 30 Hz behavior and eye-tracking streams.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `image_identity` is derived from the stimulus presentation tables under `/intervals/*_presentations`, specifically `image_name`, `start_time`, `stop_time`, and `omitted`.

ii. ```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    ...
    keep_names = [
        "start_time",
        "stop_time",
        "image_name",
        "omitted",
        "is_change",
        ...
    ]
...
image_name = str(presentations["image_name"][idx])
image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes say image identity should come from the per-presentation table rather than trial-level image-name fields, because the decoder target is time-varying and needs to represent gray intervals and omissions inside each trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent finds the presentation rows overlapping each trial, creates a trial-bin image series initialized to `gray`, and overwrites bins covered by non-omitted presentations with the corresponding image category. Omitted flashes remain `gray`.

ii. ```python
def make_image_series(
    presentations: Dict[str, np.ndarray],
    row_idx: np.ndarray,
    centers: np.ndarray,
    image_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    ...
    omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The notes explain the explicit `gray` class as necessary because trials contain inter-stimulus gray screens and omitted flashes, and the agent later fixed a bug so omitted flashes did not become a separate unused class label.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-identity series is projected onto the exact same per-trial bin centers used for neural data, using overlap between each presentation interval and the trial’s 100 ms bin centers.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. The notes describe this as one of the core synchronization choices: use the ophys-aligned trial window for all variables, and then project stimulus presentations into that shared window.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the same stimulus presentation table, using `is_change` together with the presentation `start_time`, `stop_time`, and omission state.

ii. ```python
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

iii. The notes say the agent preferred presentation-level `is_change` over reconstructing changes from successive image identities because the NWB already stores the intended AllenSDK semantics.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent initializes a binary zero vector per trial, then sets bins to `1` for the entire duration of any non-omitted presentation flagged with `is_change`.

ii. ```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The notes explicitly justify this choice: the target marks the changed-image presentation interval rather than a single instant, which the agent considered more robust after temporal binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value; it is coded directly as a binary categorical variable with classes `no_change` and `change`.

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The instructions explicitly asked for a binary image-change variable, so the notes do not discuss any alternative thresholding rule.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is assigned on the same trial-bin centers used for neural event sums, based on overlap between each 100 ms bin and stimulus-presentation intervals.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
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

iii. The notes list this as part of the raw-versus-converted spot checks: verify that the change series turns on only for the changed-image flash and matches the raw trial/presentation timing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `/processing/running/speed/timestamps` and `/processing/running/speed/data`.

ii. ```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes say this mirrors the AllenSDK-processed running stream already stored in NWB and does not attempt to recompute wheel speed from raw encoder voltages.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent pools raw running values from valid trial windows during the preview pass to compute global quintile edges, then linearly interpolates running speed to each trial’s 100 ms bin centers and discretizes those interpolated values into five bins.

ii. ```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
...
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The notes say the global bins were intentional so the same category definition holds across all sessions, and that interpolation is required because running is stored on its own synchronized time base rather than the ophys frame clock.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five equal-percentile bins using global quantile edges computed over all included running samples from all eligible sessions/trials.

ii. ```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=np.float64)
...
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
```

iii. The notes explicitly state "Running and pupil bin edges will be global, not per-session" so `output_values` have one consistent meaning across the whole dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolating the running time series onto the same trial-relative 100 ms bin centers used for neural data.

ii. ```python
running_interp = interpolate_series(running_times, running_values, centers)
...
output_trial = np.vstack(
    [
        image_series.astype(np.int16),
        change_series.astype(np.int16),
        running_bins,
        ...
    ]
)
```

iii. The notes describe this as alignment of a synchronized-but-separate behavioral clock onto the common ophys-aligned bin centers.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking timestamps plus raw pupil area and blink flags: `/acquisition/EyeTracking/eye_tracking/timestamps`, `/acquisition/EyeTracking/pupil_tracking/area_raw`, and `/acquisition/EyeTracking/likely_blink/data`.

ii. ```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes say this follows the NWB eye-tracking stream already processed by Allen tooling, with blink frames masked rather than estimated.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent masks likely-blink frames to `NaN`, converts pupil area to an equivalent diameter using `2*sqrt(area/pi)`, pools valid samples globally to compute quantile edges, and interpolates the diameter onto each trial’s bin centers before binning.

ii. ```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    ...
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

iii. The notes justify session exclusion when eye tracking is missing because pupil diameter is a required decoder output and the agent did not want to fabricate missing values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-percentile bins using global quantile edges computed across all finite pupil samples from all eligible sessions/trials.

ii. ```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
...
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
```

iii. The notes give the same reasoning as for running speed: the categories should have one stable global definition.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating the blink-masked, diameter-converted pupil trace onto the same trial-relative 100 ms bin centers used for neural data.

ii. ```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
output_trial = np.vstack(
    [
        ...
        pupil_bins,
        outcome_series,
    ]
)
```

iii. The notes repeatedly describe pupil as a separate synchronized stream that must be projected onto the ophys-aligned decoder time base.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four trial-table outcome flags: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
```

iii. The notes say outcome semantics were taken directly from the AllenSDK-style trials table and checked against the raw per-session counts after filtering.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. For each kept trial, the agent requires exactly one true outcome class, converts it to an integer index, and repeats that index across all time bins in the trial so every `output` array stays 2D and time-aligned.

ii. ```python
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. The notes explicitly say the repetition across time bins was a formatting choice to keep all outputs in `(n_output, n_timepoints)` form even when one output is static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or imperfect data in several ad hoc ways: it uses `nan_to_num` when interpreting boolean flags, decodes string-like HDF5 values defensively, masks blink-contaminated pupil samples to `NaN`, fills interpolation outside the observed time range with edge values, returns all-`NaN` arrays when a stream has no valid samples, excludes sessions with missing eye tracking, and raises hard errors when supposedly valid trials have ambiguous outcomes or when interpolated running/pupil values remain non-finite at conversion time.

ii. ```python
if isinstance(value, bytes):
    return value.decode("utf-8")
...
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
...
pupil_area[likely_blink] = np.nan
...
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(...)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The notes frame these as pragmatic fixes to preserve a fully specified decoder dataset while still excluding sessions where a required target (pupil) cannot be trusted or reconstructed.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are the full-file preview scan across all NWBs, reopening eligible files for the conversion pass, and especially the inner per-trial/per-bin neural rebinner that sums event matrices bin by bin.

ii. ```python
eligible, excluded = collect_previews(...)
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
...
for spec in preview.trial_specs:
    ...
    for b in range(T):
        lo = int(frame_starts[b])
        hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. In Step 6 and Step 9, the notes explicitly call out the preview pass and per-bin slice sums as the main runtime costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the inner `for b in range(T)` loop that sums neural events one bin at a time, the per-presentation loop in `make_image_series`, and the preview-time `collect_time_window_values` loop that concatenates slices trial by trial.

ii. ```python
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

iii. The notes acknowledge at least the neural binning inefficiency directly and say it was left in place to reduce peak memory and keep implementation simple.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats file opening and metadata reads across preview and conversion, recomputes trial/presentation parsing in both passes, and computes pupil conversion both while previewing and again during full conversion.

ii. ```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
...
if has_eye_tracking:
    pupil_area = np.asarray(
        f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64
    )
    ...
    pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. The notes explicitly describe the two-pass architecture as intentionally redundant: preview first for eligibility/bin edges, then conversion for the full output arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some diagnostic-only work that is not used by downstream decoding: optional plotting in `plot_processing_summary`, storing preview-only fields such as `raw_trial_count`, `valid_trial_count`, `is_go`, and `is_catch` mainly for summaries/plots, and collecting/recounting statistics purely for validation. These operations support auditing, not the final dataset.

ii. ```python
SHOW_PROCESSING_MAX = 2
...
def plot_processing_summary(...):
    ...
    out_path = Path(f"processing_{preview.experiment_id}.png")
    fig.savefig(out_path, dpi=150)
...
@dataclass
class TrialSpec:
    ...
    is_go: bool
    is_catch: bool
...
@dataclass
class SessionPreview:
    ...
    raw_trial_count: int
    valid_trial_count: int
```

iii. The notes say processing plots were limited to at most two sessions and describe several recount/spot-check steps that were intentionally added for validation rather than as part of the decoder-facing data representation.
