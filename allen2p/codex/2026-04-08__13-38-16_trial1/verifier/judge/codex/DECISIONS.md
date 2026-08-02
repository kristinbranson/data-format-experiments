# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly globbing local NWB files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` and reading them with `h5py`. It does a preview pass over every file to collect eligibility, image labels, and running/pupil pools, then reopens eligible files in a second conversion pass.

ii. ```python
def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))

def collect_previews(...):
    for idx, path in enumerate(files, start=1):
        preview = session_preview(path, bin_size_sec)

with h5py.File(preview.path, "r") as f:
    ...
```

iii. In `CONVERSION_NOTES.md` Step 5 and the trajectory, the AI said it used direct HDF5/NWB reads because `pynwb` failed in the environment and that it would mirror SDK semantics from the serialized NWB tables.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB `/general/subject/subject_id` field. During conversion, each previously unseen `subject_id` is added to `subjects` and mapped to a session-level `subject_idx`.

ii. ```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The AI’s notes describe subject identifiers as the per-file subject metadata and report counts from the local NWB subset rather than from the SDK experiment table.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one session. The code records `ophys_session_id` in `SessionPreview`, but it does not merge multiple experiment files that share the same `ophys_session_id`; conversion operates one file at a time.

ii. ```python
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
```

iii. In Step 2 and Step 5 notes, the AI explicitly framed the local payload as per-experiment NWB files and planned conversion around “eligible sessions,” but those “sessions” are implemented as individual experiment files.

## 1-d. How are the data split into trials?

i. Trials are split using the raw `/intervals/trials` table. For each retained trial row, the code uses `start_time` and `stop_time` as the trial bounds and later constructs uniform 100 ms bins spanning that interval.

ii. ```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    ...

for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
```

iii. The AI justified this in `CONVERSION_NOTES.md` by saying it would “mirror the SDK trial semantics already stored in those files,” using the trial table as the authoritative segmentation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `aborted` and `auto_rewarded` rows and by requiring exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject` to be true. Sessions are additionally excluded if they have missing eye tracking, fewer than 2 valid trials, or too few valid pupil samples.

ii. ```python
if aborted[idx] or auto_rewarded[idx]:
    continue
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(...)
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```

iii. The notes say the AI wanted to keep SDK-valid trials, exclude aborted/auto-rewarded trials, and drop sessions without eye tracking because pupil was treated as a required decoder target.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection matrix `/processing/ophys/event_detection/data` and its ROI index table, not from dF/F traces.

ii. ```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
```

iii. The AI’s Step 5 notes explicitly say “Neural signal = event-detection output, not dF/F” because it interpreted the paper’s analyses as event-based and preferred that over the SDK dF/F representation.

## 2-b. How is the `neural` data processed?

i. Neural data are filtered to valid ROIs, then rebinned into 100 ms trial bins by summing event magnitudes within each bin. The code does not merge multiple imaging planes into one session; each file contributes one neuron-by-time matrix per trial.

ii. ```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
...
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. In the notes, the AI said it wanted a common 100 ms time base across all recordings and would “sum event magnitudes within each 100 ms trial bin.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering ROIs by `valid_roi` using the cell specimen table. No additional neuron-activity thresholding is applied.

ii. ```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI justified this in Step 1 and Step 5 notes by citing the AllenSDK’s default ROI filtering and saying it would preserve that curation when reading NWB files directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start. For each trial, the code creates bins from `start_time` to `stop_time`; neural events falling within each bin are summed into that bin.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```

iii. The notes state “Alignment event = trial start,” with `off_start = 0.0` and variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms bin size. Yes, temporal rebinning is applied: native event samples and continuous behavior are all projected onto this common 100 ms trial-relative grid.

ii. ```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. The AI explicitly justified 100 ms bins in Step 5 as a compromise across single-plane and multi-plane sampling rates while still resolving stimulus timing.

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
```

iii. The notes say the AI chose presentation tables so it could project actual flashes onto rebinned trial bins and represent gray/omission periods explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds a global vocabulary `["gray"] + sorted(unique image names)`, initializes every bin to `gray`, then overwrites bins that overlap non-omitted stimulus presentations with that presentation’s `image_name`.

ii. ```python
image_values = ["gray"] + image_names
...
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
...
if not omitted:
    image_name = str(presentations["image_name"][idx])
    image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. Step 5 notes say image identity would include an explicit `gray` class because trials contain gray inter-stimulus intervals and omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same 100 ms trial bins as neural data by checking which bin centers fall within each presentation interval.

ii. ```python
in_window = (centers >= start) & (centers < stop)
...
image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. The AI’s notes repeatedly describe stimulus identity as being “projected onto rebinned trial bins” sharing the ophys-aligned trial window.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the same stimulus-presentation tables, using `is_change` and `omitted` for each presentation interval.

ii. ```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI justified this choice by tying image change to actual changed-image presentations rather than to trial-table `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector for each trial and sets bins to 1 wherever a non-omitted presentation interval is marked `is_change`.

ii. ```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. In the notes, the AI described image change as “1 during the changed-image presentation immediately after a true image-identity change.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is stored as a binary categorical variable with categories `no_change` and `change`.

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The AI treated image change as a naturally binary variable, so no additional thresholding beyond 0/1 assignment was needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is aligned on the same 100 ms trial bins by marking bins whose centers lie inside changed-presentation intervals.

ii. ```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. The AI’s notes describe all outputs as sharing the common rebinned ophys-aligned trial window.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `/processing/running/speed/timestamps` and `/processing/running/speed/data` in each NWB file.

ii. ```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The AI notes identify these NWB datasets as the raw running stream corresponding to the SDK running-speed table.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code pools raw running values from valid trial windows across eligible sessions to compute global quintile edges. For each trial, it linearly interpolates running speed to the 100 ms bin centers and discretizes those interpolated values with the global edges.

ii. ```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. The AI justified this as one consistent categorical definition across sessions and as part of the shared 100 ms time base.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global percentile bins labeled `q1` through `q5`.

ii. ```python
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
...
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
```

iii. The notes explicitly say running bin edges will be global rather than per-session so categories are consistent throughout the dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same 100 ms trial-bin centers used for neural event binning.

ii. ```python
running_interp = interpolate_series(running_times, running_values, centers)
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

iii. The AI’s notes describe running as one of the streams resampled onto the ophys-aligned rebinned trial axis.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `/acquisition/EyeTracking/pupil_tracking/area_raw`, `/acquisition/EyeTracking/eye_tracking/timestamps`, and `/acquisition/EyeTracking/likely_blink/data`.

ii. ```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
```

iii. The notes state the AI chose eye-tracking pupil area and converted it to diameter after blink masking, based on the whitepaper’s ellipse/area description.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code marks blink samples as `NaN`, converts pupil area to equivalent diameter with `2*sqrt(area/pi)`, pools valid samples across eligible sessions to compute global quintile edges, then interpolates the per-session diameter to 100 ms bin centers and discretizes it.

ii. ```python
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. In Step 5 notes, the AI justified this as a blink-filtered, diameter-like pupil measure on the same common binning scheme as the other outputs.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins labeled `q1` through `q5`.

ii. ```python
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_bin_edges(pupil_pool, 5)
```

iii. The AI notes say pupil bin edges are global for consistency across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same 100 ms trial-bin centers used for the neural event bins.

ii. ```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(...)
```

iii. The AI’s notes explicitly group pupil with the other streams that are resampled to the rebinned ophys-aligned trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the raw trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
```

iii. The AI treated those four SDK-derived flags as the canonical outcome labels and required exactly one of them per retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcome booleans are converted to a categorical index by position in `TRIAL_OUTCOME_VALUES`; that index is then repeated across all bins in the trial.

ii. ```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
outcome_idx = outcome_flags.index(True)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. The notes say trial outcome was kept as a static per-trial variable but repeated across bins so every output trial has a uniform `(n_output, n_timepoints)` shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing-eye-tracking sessions are dropped entirely. Blink samples are set to `NaN` before pupil conversion. Interpolation uses `np.interp`, which extends edge values rather than producing `NaN` outside the sampled range; the code then raises if any non-finite interpolated running/pupil values remain. The AI also fixed a bug where `omitted` had leaked into the image vocabulary even though omitted periods were encoded as `gray`.

ii. ```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
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

iii. The notes and trajectory say the AI excluded missing-eye sessions because pupil was considered mandatory, and later documented a corrective fix removing an unused `omitted` image label from `output_values`.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the two full passes over large NWB files and the per-trial/per-bin event summation loop during conversion. Optional diagnostic plotting also adds cost when enabled.

ii. ```python
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

iii. The AI itself noted in Step 6 that preview avoids loading neural matrices but still scans the dataset, and that per-bin event summation trades memory for extra CPU.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the inner `for b in range(T)` loop that sums event data separately for every time bin. The preview pass also loops file-by-file and repeatedly slices running/pupil values trial-by-trial when pooling samples.

ii. ```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
...
for spec in trial_specs:
    lo = np.searchsorted(times, spec.start_time, side="left")
    hi = np.searchsorted(times, spec.stop_time, side="left")
```

iii. The notes acknowledge that the current event-rebin implementation favors lower peak memory over CPU efficiency.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats dataset reads across a preview pass and a conversion pass. It also rereads trials and presentation tables during conversion after already reading them in preview, and it computes global image/running/pupil statistics in preview before recomputing per-trial aligned outputs in conversion.

ii. ```python
preview = session_preview(path, bin_size_sec)
...
with h5py.File(preview.path, "r") as f:
    ...
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. The AI intentionally designed a lightweight first pass for eligibility/bin-edge collection and a second pass for final assembly; that rationale is described in Step 6 notes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores preview-only metadata such as `raw_trial_count`, `valid_trial_count`, `session_type`, and `unique_images` solely to support eligibility checks and logging. It also has optional plotting machinery (`plot_processing_summary`) that is not part of the downstream decoder input. More importantly, it records `ophys_session_id` in previews but never uses it to merge planes into true sessions.

ii. ```python
@dataclass
class SessionPreview:
    ...
    ophys_session_id: int
    session_type: str
    unique_images: List[str]
    raw_trial_count: int
    valid_trial_count: int
...
if show_processing:
    plot_processing_summary(...)
```

iii. The notes present some of this as sanity-check infrastructure, but these computations are discarded once the final converted arrays are written and do not contribute to decoder training.
