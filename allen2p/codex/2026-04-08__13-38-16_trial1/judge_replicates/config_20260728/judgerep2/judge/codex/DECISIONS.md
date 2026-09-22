# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the AllenSDK cache API. It globbed local NWB experiment files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, ran a preview pass over each file with `h5py`, and then reopened each eligible file in a second conversion pass. In practice, "all data" means all local experiment files, not all sessions discovered from the SDK experiment table.

ii. 
```python
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

iii. In Step 6 of `CONVERSION_NOTES.md`, the AI said it used direct HDF5 reads because the installed NWB stack was incompatible. In trajectory step 120, it explicitly justified a two-pass direct-HDF5 loader that mirrors SDK semantics from the serialized NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file's `/general/subject/subject_id` field, then deduplicated in first-seen order while iterating through eligible experiment files.

ii. 
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
...
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
```

iii. The notes describe subject identifiers as coming from NWB subject metadata. The AI treated that field as the subject split key rather than using `mouse_id` from the AllenSDK experiment table.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one session. The code records `ophys_session_id` in `SessionPreview`, but it never groups multiple experiments that share an `ophys_session_id`; conversion proceeds one experiment file at a time.

ii. 
```python
@dataclass
class SessionPreview:
    path: Path
    experiment_id: int
    ophys_session_id: int
    subject_id: str
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview,
        ...
    )
```

iii. In Step 2 and Step 9 of `CONVERSION_NOTES.md`, the AI repeatedly framed the local payload as "284 experiment files" and reported "281 eligible local experiment files." In trajectory steps 62 and 228 it used experiment-file counts, not grouped `ophys_session_id` sessions, as the conversion unit.

## 1-d. How are the data split into trials?

i. Trials are read directly from `/intervals/trials`. For each non-aborted, non-auto-rewarded entry with exactly one valid outcome flag, the AI stores `start_time` and `stop_time` in a `TrialSpec`, then builds 100 ms bins spanning the full start-to-stop window during conversion.

ii. 
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
    ...
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        specs.append(
            TrialSpec(
                start_time=float(trials["start_time"][idx]),
                stop_time=float(trials["stop_time"][idx]),
                ...
            )
        )
```

iii. The AI's notes say it was keeping SDK trial semantics already serialized in the NWB file. In Step 5 it chose "SDK-valid trials" and in trajectory step 112 it said it would keep SDK trial definitions but put all sessions onto one common trial-relative time base.

## 1-e. How are trials filtered based on quality controls?

i. The code excludes trials with `aborted` or `auto_rewarded` set, and it requires exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject` to be true. At the session/experiment level, it excludes files with missing eye tracking, fewer than two valid trials, or too few finite pupil samples.

ii. 
```python
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

iii. Step 5 of `CONVERSION_NOTES.md` explicitly says sessions without eye tracking are excluded because pupil diameter is a required output. Trajectory step 112 gives the same justification and treats the no-eye-tracking exclusion as a design choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is taken from `/processing/ophys/event_detection/data`, with ophys timestamps taken from `/processing/ophys/dff/traces/timestamps`. The AI did not use dF/F traces as the actual neural signal.

ii. 
```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    ...

ophys_timestamps = np.asarray(
    f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
)
event_data, _ = load_neural_events(f)
```

iii. Step 5 of `CONVERSION_NOTES.md` states the key decision directly: "Neural signal = event-detection output, not dF/F." Trajectory steps 95 and 112 say this was chosen because the paper's analyses used discrete calcium events.

## 2-b. How is the `neural` data processed?

i. The AI filters event rows to `valid_roi`, then rebins each trial to 100 ms bins and sums event magnitudes within each bin. It does not merge multiple imaging planes into a grouped session; each experiment file is converted independently.

ii. 
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
...
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 6 says the code uses "per-bin slice sums" and a common 100 ms time base. In trajectory step 112, the AI justified this as needed to keep one shared bin size across single-plane and multi-plane recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural rows are filtered by the NWB `valid_roi` mask before trial extraction. No additional event-amplitude thresholding or trial-level neural filtering is applied.

ii. 
```python
event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. The AI's Step 1 and Step 4 notes emphasize that the AllenSDK filters invalid ROIs by default, and because it bypassed the SDK it manually reproduced that part of the curation logic.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start and cover the full `start_time` to `stop_time` interval. Bin boundaries are converted to ophys frame indices with `np.searchsorted`, so neural values are summed over ophys-timestamp-defined frame ranges inside each trial bin.

ii. 
```python
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
    frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
    frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
    ...
```

iii. Step 5 says "Alignment event = trial start," and trajectory step 112 repeats that trial start was chosen as the natural unit for the requested trialized decoder dataset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are explicitly rebinned to 100 ms bins (`BIN_SIZE_SEC = 0.1`). Neural activity is re-expressed as the sum of event magnitudes within each 100 ms bin, and the metadata records this as the binning rule.

ii. 
```python
BIN_SIZE_SEC = 0.1
...
"metadata": {
    ...
    "time_bin_size": BIN_SIZE_SEC * 1000.0,
    "neural_representation": "ophys event-detection magnitudes",
    "binning_rule": "sum event magnitudes within each 100 ms trial bin",
}
```

iii. Step 5 calls 100 ms a tentative common bin size, justified as coarse enough to avoid pathological upsampling of 11 Hz recordings while still resolving stimulus flashes. Step 6 says the common 100 ms base was chosen to satisfy the shared-bin-size target format requirement.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation tables under `/intervals/*_presentations`, specifically `image_name`, `start_time`, `stop_time`, and `omitted`. It is not derived from the trial table's `initial_image_name` and `change_image_name`.

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
...
image_name = str(presentations["image_name"][idx])
image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. In Step 5, the AI's mapping table says image identity comes from stimulus presentations projected onto trial bins. The notes justify this by trying to represent both flashes and gray inter-stimulus intervals explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a global image vocabulary from presentation-table image names, prepends a synthetic `gray` class, and then fills each trial bin based on whether the bin center falls inside a non-omitted presentation interval. Omitted flashes and uncovered periods are encoded as `gray`.

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

iii. Step 5 explicitly says "Image identity will include a `gray` class" because trials contain gray ISIs and omission periods. Step 10 later notes a bug fix where `omitted` was removed from the class vocabulary and mapped only to `gray`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned to the same 100 ms trial bins used for neural data. The code marks bins whose centers lie inside a stimulus presentation interval, so alignment is through common rebinned trial centers rather than through native ophys frames.

ii. 
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
in_window = (centers >= start) & (centers < stop)
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

iii. The notes and trajectory present the common 100 ms trial grid as the master alignment axis for all outputs, with ophys timestamps used only to anchor the neural event sums.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the presentation table's `is_change`, `start_time`, `stop_time`, and `omitted` fields. It is not driven by trial-table `change_time` and `go`.

ii. 
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 says the binary image-change output should mark the changed-image presentation itself. Trajectory step 112 summarizes this as keeping SDK trial definitions but projecting presentation-level variables onto trial bins.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector for each trial and sets bins to 1 whenever the bin center falls inside a non-omitted presentation whose `is_change` flag is true. This marks the changed-image presentation interval, not just an instant or a fixed 750 ms post-change window.

ii. 
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
in_window = (centers >= start) & (centers < stop)
...
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 decision 9 says "Image-change target will mark the changed-image presentation, not only a single instant," because the AI considered that more robust after binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no numeric thresholding step. The code treats image change as an already-binary categorical series with values 0 and 1, stored as `["no_change", "change"]`.

ii. 
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The justification is implicit in the task design and Step 5 mapping: image change is a binary event target, so the AI carried over a categorical 0/1 representation instead of computing a threshold from a continuous variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same 100 ms trial bins as neural activity and image identity. The presentation interval is projected onto `centers`, and those bins are stacked into the per-trial output matrix alongside the rebinned neural data.

ii. 
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
neural_trials.append(neural_trial)
output_trials.append(output_trial)
```

iii. The AI's Step 6 and Step 10 notes repeatedly describe all outputs as projected onto the shared 100 ms ophys-anchored trial grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is read directly from the NWB running processing group: `/processing/running/speed/data` and `/processing/running/speed/timestamps`.

ii. 
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. The notes say the AI mirrored SDK semantics already stored in the NWB. For running, that meant using the serialized running-speed stream rather than recomputing wheel velocity.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI pools running values from valid trial windows across eligible files to compute global quantile edges. During conversion, it linearly interpolates running speed to each trial's 100 ms bin centers with `np.interp`, then digitizes the rebinned values into five bins.

ii. 
```python
running_concat = collect_time_window_values(running_times, running_values, specs)
...
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 5 and Step 6 justify the global bin edges as necessary for one consistent categorical output definition. The common 100 ms binning was justified as a target-format compromise across mixed acquisition rates.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded by global quintile edges computed from all finite running samples gathered from eligible trials. `np.digitize` converts each rebinned sample into an integer bin 0-4, labeled `q1` through `q5`.

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

iii. The notes explicitly say running and pupil bin edges are global rather than per-session so that `output_values` has one consistent meaning across the dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to the same 100 ms trial centers used to bin neural events. It is not interpolated to native ophys frames first; instead both signals are expressed on the rebinned trial grid.

ii. 
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
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

iii. The AI's Step 6 notes say it used a "common 100 ms trial-relative time base across all sessions" while keeping alignment anchored to ophys time.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil output comes from raw eye-tracking area measurements in `/acquisition/EyeTracking/pupil_tracking/area_raw`, timestamps in `/acquisition/EyeTracking/eye_tracking/timestamps`, and the blink mask in `/acquisition/EyeTracking/likely_blink/data`.

ii. 
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
```

iii. In Step 5, the AI justified this by saying the whitepaper treated pupil area as convertible to an equivalent diameter and that blink-contaminated samples should be masked before interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI masks blink frames to `NaN`, converts pupil area to an equivalent diameter `2*sqrt(area/pi)`, pools valid within-trial values to compute global quantile edges, interpolates diameter to 100 ms trial centers, and digitizes into five bins.

ii. 
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    ...
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
...
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Step 5 says pupil diameter is required and that sessions with missing eye tracking should be excluded rather than fabricating values. The same notes describe blink filtering before interpolation and global percentile binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded exactly like running speed: global quintile edges are computed across all finite eligible-trial samples and each rebinned value is digitized into bins 0-4, labeled `q1` through `q5`.

ii. 
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
...
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Step 5 decision 10 says running and pupil bin edges are global so their categories are comparable across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is aligned to the same 100 ms trial-bin centers used for neural activity, image outputs, and running speed. Alignment is therefore through the shared rebinned trial grid rather than native ophys frame indices.

ii. 
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
neural_trials.append(neural_trial)
output_trials.append(output_trial)
```

iii. Step 6 and Step 10 both describe the conversion as putting all modalities onto the same 100 ms ophys-anchored trial window.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
```

iii. The notes treat these as the SDK's canonical outcome flags for valid non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI requires exactly one outcome flag to be true, converts that to an integer index 0-3, and then repeats that index across every time bin of the trial so the output matrix stays uniformly time-varying.

ii. 
```python
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
...
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
```

iii. Step 5 decision 11 explicitly says trial outcome is repeated across time bins so every output trial remains shape `(n_output, n_timepoints)`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI decodes byte strings from HDF5, tolerates missing presentation tables by returning empty arrays, masks blink samples to `NaN`, excludes experiment files with missing eye tracking or too few valid pupil samples, and raises an error if running or pupil interpolation still yields non-finite values during conversion. It does not impute missing values into a reserved output bin.

ii. 
```python
if has_eye_tracking:
    ...
else:
    pupil_times = np.array([], dtype=np.float64)
    pupil_diameter = np.array([], dtype=np.float32)
...
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
```

iii. Step 5 and trajectory step 112 justify the no-eye-tracking exclusion because pupil is a required target. Step 6 also says direct HDF5 access was chosen because the installed NWB stack was incompatible, so some of the defensive code is about reading raw NWB payloads safely.

## 9-a. What are the most time-consuming steps of the code?

i. The AI's own notes identify two main costs: the full preview pass over every NWB file and the per-session neural rebinning during conversion, especially the per-bin slice sums over event matrices.

ii. 
```python
eligible, excluded = collect_previews(...)
...
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 6 says the preview/conversion two-pass workflow was intentional, and specifically calls out per-bin event summation as a CPU cost accepted to reduce peak memory.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the inner loop over bins in `convert_session`, which repeatedly slices `event_data` and sums over rows. The preview pass also loops file-by-file and trial-by-trial when pooling running/pupil values.

ii. 
```python
for spec in preview.trial_specs:
    ...
    neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
    for b in range(T):
        lo = int(frame_starts[b])
        hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 6 explicitly mentions that per-bin slice sums could have been replaced with a more vectorized cumulative-sum style implementation, but says the current version was kept as a memory-conscious tradeoff.

## 9-c. What processing does the code repeat multiple times?

i. The code intentionally repeats I/O and some parsing: one preview pass opens every file to gather eligibility, image vocabulary, and quantile pools, and the conversion pass reopens every eligible file to build the final dataset. It also rereads trials and presentations in both passes.

ii. 
```python
preview = session_preview(path, bin_size_sec)
...
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(
        preview=preview,
        ...
    )
```

iii. Step 6 calls this out directly as an identified inefficiency and says it was intentional so global bin edges and image categories could be computed without storing large full-session neural arrays from the first pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes preview summaries, optional diagnostic plots, and per-session timing/stat dictionaries that are not needed by downstream decoder analyses. Even in normal runs it constructs `session_stats` for logging only, and `show-processing` mode adds visualization work whose outputs are not consumed by the final pickle.

ii. 
```python
session_stats: List[Dict[str, int]] = []
...
session_stats.append(stats)
...
if show_processing:
    plot_processing_summary(
        preview=preview,
        ...
    )
```

iii. Step 6 and Step 7 explicitly describe these diagnostics as validation aids. They were justified as sanity-check tooling rather than as information required in `converted_data.pkl`.
