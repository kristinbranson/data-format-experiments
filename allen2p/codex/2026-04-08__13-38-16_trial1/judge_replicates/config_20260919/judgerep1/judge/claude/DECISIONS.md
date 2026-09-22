# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the local NWB/HDF5 files directly with `h5py`. It globs every `behavior_ophys_experiment_*.nwb` file under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (284 files) and treats the file list as the complete dataset. Each file is opened twice: once in a lightweight "preview" pass (trial table, stimulus-presentation tables, running speed, eye tracking — but *not* the neural matrices) to determine eligibility, the global image vocabulary and the global running/pupil quantile edges; and once in a "conversion" pass that additionally reads the event-detection matrix and builds the per-trial arrays. The project metadata CSVs are read only in `--sample` mode (to rank files by cell count). No filtering on `project_code` is performed, so both `VisualBehavior` (239 files) and `VisualBehaviorMultiscope` (45 files) experiments are included.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
```

```python
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
eligible, excluded = collect_previews(
    files=files, bin_size_sec=BIN_SIZE_SEC,
    sample_mode=args.sample, required_eligible=2,
)
...
for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
```

```python
def session_preview(path: Path, bin_size_sec: float) -> SessionPreview:
    with h5py.File(path, "r") as f:
        experiment_id = int(decode_scalar(f["/identifier"][()]))
        ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
        subject_id = decode_scalar(f["/general/subject/subject_id"][()])
        session_type = decode_scalar(f["/general/metadata"].attrs["session_type"])
        brain_region = decode_scalar(f["/general/optophysiology/imaging_plane_1/location"][()])
        has_eye_tracking = "/acquisition/EyeTracking/pupil_tracking/area_raw" in f
        trials = read_trials(f)
        specs = build_trial_specs(trials)
        presentations = read_task_presentations(f)
```

iii. From CONVERSION_NOTES Step 6: "Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment." It argues the reads mirror SDK semantics already serialized into NWB (trials from `/intervals/trials`, stimulus timing from `/intervals/*_presentations`, running from `/processing/running/speed`, pupil from `/acquisition/EyeTracking/*`, neural from `/processing/ophys/event_detection`). The two-pass design is justified as necessary "to avoid storing large neural matrices before global percentile/bin definitions are known." Step 2 notes that the metadata CSVs describe the full release (1936 experiments) while only 284 NWB files are present locally, and Step 4 resolves this as a "release-version and local-download-scope difference," using the local files as authoritative.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `/general/subject/subject_id` strings read from each NWB file. They are registered in first-encountered order (which is sorted-by-experiment-id order) into `subjects`, and each session records an index into that list. 38 subjects result.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
```
```python
if preview.subject_id not in subject_to_idx:
    subject_to_idx[preview.subject_id] = len(subjects)
    subjects.append(preview.subject_id)
...
subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. Step 5 variable mapping: "`/general/subject/subject_id` or metadata `mouse_id` → `subjects`, `subject_idx`; String subject identifiers with session-level index mapping; Session order follows converted session order." The AI verified against the metadata table in Step 2 that there are 38 mice in the locally downloaded subset (vs. 107 in the full release metadata and 82 in the whitepaper), and attributed the difference to download scope.

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one "session".** The AI does not group experiments by `ophys_session_id`, even though it reads that attribute into `SessionPreview.ophys_session_id` (the field is never used again) and even though Step 2 explicitly records "Unique local ophys sessions: 247" and "Experiments per local ophys session: range 1-7, mean 1.15". Consequently the 45 `VisualBehaviorMultiscope` imaging planes that were recorded simultaneously in only 8 behavioural sessions become 45 separate entries in `data['neural']`. The verification log shows the effect directly: `Subject 457841: 45 sessions`. Sessions are ordered by experiment-id file sort, not by acquisition date. Final count: 281 "sessions" (284 files − 3 without eye tracking).

ii.
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])   # read, never used
...
return SessionPreview(path=path, experiment_id=experiment_id,
                      ophys_session_id=ophys_session_id, ...)
```
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
    subject_idx.append(subject_to_idx[preview.subject_id])
    brain_region_idx.append(region_idx_arr)
```
```python
brain_region_idx = np.full(
    event_data.shape[1], brain_region_to_idx[preview.brain_region], dtype=np.int64
)   # one region per "session", i.e. per imaging plane
```

iii. The only justification offered is in the Step 2 table ("experiment files are the unit of raw neural recording payload") and in the Step 9/README wording "Included sessions: 281 ophys experiment files". There is no explicit discussion of the multi-plane case, of why planes recorded simultaneously are not merged, or of the consequence that the same behavioural trials are then replicated across several "sessions". Step 1 does note "One experiment corresponds to one imaging plane in one session", so the AI was aware of the distinction but did not act on it.

## 1-d. How are the data split into trials?

i. Trials come from the SDK trial table serialized at `/intervals/trials`. Every non-aborted, non-auto-rewarded row becomes a trial, and the trial window is the full `[start_time, stop_time)` interval (variable length, ~7–13 s; converted `T` ranges 71–127 bins of 100 ms, mean 85.8). Both go and catch trials are kept. Trial windows are not clipped to the end of the ophys recording.

ii.
```python
def build_trial_specs(trials):
    raw_count = len(trials["id"])
    aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
    auto_rewarded = np.nan_to_num(trials["auto_rewarded"], nan=0.0).astype(bool)
    ...
    for idx in range(raw_count):
        if aborted[idx] or auto_rewarded[idx]:
            continue
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        specs.append(TrialSpec(trial_idx=idx,
                               start_time=float(trials["start_time"][idx]),
                               stop_time=float(trials["stop_time"][idx]), ...))
```
```python
def build_bin_centers(start_time, stop_time, bin_size_sec):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([start_time], dtype=np.float64)
    ends = np.minimum(starts + bin_size_sec, stop_time)
    widths = np.maximum(ends - starts, 1e-6)
    centers = starts + 0.5 * widths
    return starts, ends, centers
```

iii. Step 4: "Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly." Step 5 key decision 2: "Use SDK-valid trials only: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed and consistent with the trial table semantics." Step 5 key decision 7: "Alignment event = trial start. The trial itself is the natural unit requested by the user," with `off_start = 0.0` and variable trial length (`off_end = None`). The AI verified the counts against the raw NWBs: 84,313 eligible trials = raw trials − aborted − auto-rewarded.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: `aborted` and `auto_rewarded` are dropped; any surviving trial that does not have exactly one of `hit`/`miss`/`false_alarm`/`correct_reject` raises an exception (a hard consistency assertion rather than a filter — it never fired). No requirement is imposed that `change_time` be finite. Session-level: an experiment is dropped if it has no eye-tracking group (`missing_eye_tracking`, 3 files), fewer than 2 valid trials (`fewer_than_2_valid_trials`, 0 files), or fewer than 2 finite pupil samples inside trial windows (`insufficient_valid_pupil_samples`, 0 files). Result: 281/284 experiments and 84,313 trials retained; passive sessions (`OPHYS_2_..._passive`, `OPHYS_5_..._passive`) are kept.

ii.
```python
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```
```python
if len(eligible) < 2:
    raise RuntimeError("Need at least 2 eligible sessions after filtering")
```

iii. Step 5 key decision 3: "Keep passive sessions if they have valid GO/CATCH trials: Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs." Key decision 4: "Exclude sessions with missing eye tracking: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values." The ≥2-trial rule follows the format spec's "at least two trials within each session". Step 10 reports a sanity check that converted trial count per session equals raw count minus aborted minus auto-rewarded, and that outcome counts match.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `/processing/ophys/event_detection/data` — the L0-regressed discrete calcium-event magnitudes, shape `(n_ophys_frames, n_rois)` — restricted to ROIs flagged `valid_roi == True` in `/processing/ophys/image_segmentation/cell_specimen_table`. dF/F is *not* used for the neural output (the dF/F group is opened only to read its `timestamps`, which are byte-identical to the event-detection timestamps). 41,871 of 42,147 local ROIs survive the `valid_roi` filter.

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
```python
ophys_timestamps = np.asarray(f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64)
event_data, _ = load_neural_events(f)
```

iii. Step 5 key decision 1: "**Neural signal = event-detection output, not dF/F**: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events. This is the closest match to the paper while still using the SDK-defined loading and ROI filtering." The AI quoted the paper directly: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f." It also decided to use the raw events rather than the SDK's `filtered_events` because the latter is "visualization-only".

## 2-b. How is the `neural` data processed?

i. The only processing is temporal re-binning: for each trial and each 100 ms bin, the event magnitudes of all ophys frames whose timestamps fall in `[bin_start, bin_end)` are summed, per neuron, producing a `(n_neurons, T)` float32 matrix. No normalization, smoothing, z-scoring, or cross-plane stacking is applied (cross-plane stacking never happens because each imaging plane is its own "session" — see 1-c). Bins containing no ophys frame are left at exactly 0.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
T = centers.size
n_neurons = event_data.shape[1]

neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Metadata records the rule explicitly: `"neural_representation": "ophys event-detection magnitudes"`, `"binning_rule": "sum event magnitudes within each 100 ms trial bin"`. Step 5 key decision 5: "Common time base via uniform rebinned trial bins: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions." Summation (rather than averaging) is the natural aggregation for event magnitudes, preserving total event mass per bin. Step 6 notes the loop was preferred over cumulative sums to cap peak memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the SDK's ROI-validity filter: neurons whose `valid_roi` entry is `False` are dropped (276 of 42,147 locally). No further per-neuron QC (SNR, event rate, activity threshold) is applied, and no neuron is dropped for producing all-zero trials. Sessions with as few as 4 neurons are retained.

ii. See the `valid_roi` code in 2-a.

iii. Step 1: "`CellSpecimens.__init__` with `exclude_invalid_rois` — Filters the cell table to `valid_roi == True`, then filters/reorders all traces and events to the remaining ROIs" and "invalid ROIs are removed when `exclude_invalid_rois=True` (default in `BehaviorOphysExperiment.from_nwb/from_lims`)". Step 4: "Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid." Step 3 notes the whitepaper's session-level QC (z-drift >10 µm, motion, sync) was already applied before release, so no re-derivation was attempted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to **trial start**: the 100 ms bin grid begins exactly at `trials.start_time` and runs to `trials.stop_time`. Neural frames are assigned to bins by `np.searchsorted` on the ophys timestamps, so the neural data share the same absolute-time grid as every output stream. Metadata: `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```
```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 5 key decision 7: "Alignment event = trial start. The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`)." Step 3 records that all streams were hardware-synchronized on one NI board at 100 kHz, so absolute-time indexing across streams is valid. Step 10 verified alignment by reconstructing five trials (including a minimum-length trial and three all-zero trials) directly from raw NWB and comparing with `np.allclose()`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms uniform bins for every trial and session** (`time_bin_size = 100.0` ms). This *is* a rebinning: native ophys sampling is 32.3 ms/frame for single-plane `VisualBehavior` experiments (~3 frames per bin) and 93.2 ms/frame for the `VisualBehaviorMultiscope` experiments (~1.07 frames per bin). Running speed and pupil are interpolated onto the bin centers; image identity/change are computed from whether the bin center falls inside a flash interval. The last bin of each trial is truncated at `stop_time` and is therefore shorter than 100 ms.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```
```python
ends = np.minimum(starts + bin_size_sec, stop_time)
widths = np.maximum(ends - starts, 1e-6)
centers = starts + 0.5 * widths
```

iii. Step 5 key decision 6: "**Tentative common bin size = 100 ms**: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams." Step 10's reference-code comparison acknowledges the deviation: "Native SDK data remain at native sample rates; the papers do not prescribe a common decoder bin size … Difference is intentional and required by the target format, not a mismatch in source processing."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The stimulus-presentation interval tables (`/intervals/*_presentations`), specifically `image_name`, `start_time`, `stop_time` and `omitted`. Only interval groups that actually contain an `image_name` column are used, so `natural_movie_one_presentations` and `spontaneous_presentations` are automatically skipped; the concatenated rows are sorted by `start_time`. The trials-table fields `initial_image_name` / `change_image_name` are read but not used to build this output.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    interval_root = f["/intervals"]
    keep_names = ["start_time", "stop_time", "image_name", "omitted",
                  "is_change", "is_sham_change", "trials_id", "active"]
    for key in interval_root.keys():
        if key == "trials":
            continue
        group = interval_root[key]
        if "image_name" not in group or "start_time" not in group or "stop_time" not in group:
            continue
        rows.append(read_interval_group(group, keep_names))
    ...
    order = np.argsort(out["start_time"])
```

iii. Step 5 variable mapping: "Stimulus presentation `image_name` + presentation timing + omission state → `output[0]` (`image_identity`); Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods." The AI preferred the presentation table over the trials table because it gives the literal on-screen stimulus at every moment, including the 500 ms grey ISI and the 5 % omitted flashes described in the whitepaper.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is built by pooling `unique_images` across all eligible sessions, sorting them, and prepending an explicit `"gray"` class at index 0 → 17 classes (`gray` + 16 images: the 8 image-set-A and 8 image-set-B images). Every bin defaults to `gray`; bins whose centre falls inside a non-omitted flash get that flash's image code. Omitted flashes and any unrecognised name fall back to `gray`. The resulting distribution is `gray 0.670` and ~0.020–0.022 for each of the 16 images.

ii.
```python
def unique_nonempty_images(presentations):
    # Omitted flashes are represented as gray in the converted output and should
    # not become a separate image-identity class.
    names = [str(x) for x in presentations["image_name"]
             if str(x) not in {"", "nan", "None", "omitted"}]
    return sorted(set(names))
```
```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```
```python
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
for idx in row_idx:
    ...
    omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
    if not omitted:
        image_name = str(presentations["image_name"][idx])
        image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

iii. Step 5 key decision 8: "Image identity will include a `gray` class: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin." In Step 10 the AI found and fixed a real bug here: `output_values[0]` had initially contained an unused `omitted` label; it was removed so that omitted flashes map only to `gray`, bringing the vocabulary to the expected 17 classes ("consistent with the paper/whitepaper task description of two eight-image sets").

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Same 100 ms trial-relative bin grid as the neural data. A bin takes a flash's label if the bin **centre** lies in `[flash.start_time, flash.stop_time)`; candidate flashes are pre-selected by interval overlap with the trial window. Because flashes are 250 ms and bins 100 ms, ~2–3 bins per flash are labelled with the image and the remaining ~5 bins of each 750 ms cycle stay `gray`.

ii.
```python
def find_presentation_rows(presentations, start_time, stop_time):
    starts = presentations["start_time"]
    ends = presentations["stop_time"]
    mask = (starts < stop_time) & (ends > start_time)
    return np.flatnonzero(mask)
```
```python
for idx in row_idx:
    start = float(presentations["start_time"][idx])
    stop = float(presentations["stop_time"][idx])
    if stop <= start:
        continue
    in_window = (centers >= start) & (centers < stop)
```

iii. Step 10 alignment comparison: "`build_bin_centers()`, event sums on ophys timestamps, running/pupil interpolation to ophys-aligned bin centers … Alignment matches the reference time bases; the only added step is the decoder-required common 100 ms rebinning." Step 12: "Temporal alignment was rechecked using the processing plots generated during sample conversion; stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch." The `--show-processing` plots overlay the raw presentation intervals (`axvspan`) with the binned `image_series` step plot for visual verification.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` flag (together with `omitted`, `start_time`, `stop_time`) from the same stimulus-presentation tables. `is_change` is True only for genuine image changes; sham changes on catch trials carry `is_sham_change` instead, so catch trials correctly receive an all-zero change series. The trials-table `change_time` is parsed into `TrialSpec.change_time` but is used only for the diagnostic plots, not for the output.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 variable mapping: "Stimulus presentation `is_change` + presentation timing → `output[1]` (`image_change`); Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0", with trial `change_time` retained "for sanity checks". Deriving it from the presentation table keeps image identity and image change definitionally consistent (both come from the same rows).

## 4-b. What processing is involved in computing `output` *Image change*?

i. Minimal: a zero-initialised int16 vector per trial, set to 1 for the bins whose centres fall inside the changed flash's 250 ms presentation window (typically 2–3 bins of 100 ms). Omitted flashes cannot be change flashes and are skipped. Pooled positive rate: 2.63 % of bins.

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

iii. Step 5 key decision 9: "Image-change target will mark the changed-image presentation, not only a single instant: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure." This satisfies the instruction "Have value of 1 right after a change in image identity, otherwise 0" while avoiding a single-bin label that would be sensitive to bin-edge placement.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is already binary (`0 = no_change`, `1 = change`), with `output_values[1] = ["no_change", "change"]`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. Not discussed beyond the instruction's specification of image change as a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identical to image identity: computed on the same 100 ms bin-centre grid inside the same trial window, in the same loop over the same overlapping presentation rows, so it is frame-for-frame aligned with the neural matrix.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins, pupil_bins, outcome_series,
])
```

iii. Same as 3-c. The `--show-processing` plot draws a red `axvline` at each raw `is_change` presentation start on top of the binned `change_series`, so misalignment would be visible; Step 12 reports "no visible lag mismatch".

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `/processing/running/speed/data` with `/processing/running/speed/timestamps` — the SDK's filtered running-speed time series (~60 Hz on the stimulus timebase). `speed_unfiltered` and `dx` are not used.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 1 identified `RunningSpeed.from_stimulus_file` as the SDK's running-speed object; Step 3 notes the whitepaper "explicitly points to AllenSDK running-processing code for the implementation," so the pre-computed `speed` series is taken as the reference-processed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`) of the native running series onto the trial's 100 ms bin centres, after dropping non-finite samples and sorting by time. Outside the sampled range, the first/last value is held constant (`left=v[0], right=v[-1]`) rather than producing NaN. Degenerate cases (0 or 1 finite sample) return NaN-filled or constant arrays. A hard check then raises if any non-finite value survives.

ii.
```python
def interpolate_series(times, values, query_times):
    finite = np.isfinite(times) & np.isfinite(values)
    if query_times.size == 0:
        return np.array([], dtype=np.float32)
    if finite.sum() == 0:
        return np.full(query_times.shape, np.nan, dtype=np.float32)
    if finite.sum() == 1:
        v = float(values[finite][0])
        return np.full(query_times.shape, v, dtype=np.float32)
    t = np.asarray(times[finite], dtype=np.float64)
    v = np.asarray(values[finite], dtype=np.float64)
    order = np.argsort(t); t = t[order]; v = v[order]
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
```
```python
running_interp = interpolate_series(running_times, running_values, centers)
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
```

iii. Step 5: "Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins." Interpolation is justified by Step 3/Step 4's finding that all streams are hardware-synced on one 100 kHz clock, so resampling one synchronized stream onto another's grid is valid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **global** quantile bins. The edges are the 20th/40th/60th/80th percentiles of a pool of *raw* running samples taken from inside the valid trial windows of every eligible session (gathered during the preview pass), and `np.digitize` then assigns each interpolated bin value to one of 5 classes. Resulting distribution: `[0.199, 0.201, 0.199, 0.200, 0.200]`.

ii.
```python
def collect_time_window_values(times, values, trial_specs):
    for spec in trial_specs:
        lo = np.searchsorted(times, spec.start_time, side="left")
        hi = np.searchsorted(times, spec.stop_time, side="left")
        if hi > lo:
            segments.append(values[lo:hi])
    return np.concatenate(segments).astype(np.float32, copy=False)
```
```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    return np.asarray(np.quantile(values, quantiles), dtype=np.float64)

def discretize_with_edges(values, edges):
    bins = np.digitize(values, edges, right=False)
    return np.clip(bins, 0, len(edges)).astype(np.int16)
```
```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
```

iii. Step 5 key decision 10: "Running and pupil bin edges will be global, not per-session: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`." Planned sanity check: "global running and pupil bins should each contain roughly 20 % of included samples by construction" — verified in Step 9 (0.199–0.201). Restricting the pool to within-trial samples keeps the quantiles representative of the data actually emitted (the 5-minute grey periods and movie blocks are excluded).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same 100 ms bin centres used for the neural matrix within the same trial window, so index *b* of the running row corresponds to exactly the same absolute-time bin as column *b* of the neural matrix.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 10: "running/pupil interpolation to ophys-aligned bin centers … Alignment matches the reference time bases." The processing plots overlay the raw running trace and the rebinned/discretized series on a shared time axis for the same trial.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `/acquisition/EyeTracking/pupil_tracking/area_raw` (the un-blink-filtered pupil ellipse area) with `/acquisition/EyeTracking/eye_tracking/timestamps`, plus `/acquisition/EyeTracking/likely_blink/data`. Blink frames are set to NaN, then the area is converted to an equivalent circular diameter `d = 2·sqrt(area/π)`. Experiments lacking the eye-tracking group are excluded entirely (3 files).

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area  = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. Step 5 variable mapping: "Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins." Step 3 records the whitepaper's statement that "pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter" — so inverting the area recovers the whitepaper's diameter definition. Step 1 notes the SDK's `EyeTrackingTable` "recomputes likely blinks, and filters blink frames", which the AI reproduces manually.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: NaN (blink) samples are dropped by the `finite` mask inside `interpolate_series`, so the signal is linearly interpolated *across* blink gaps onto the 100 ms bin centres, with constant hold outside the sampled range. A hard check raises on any surviving non-finite value.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Step 5, same rationale as running speed. Blink removal before interpolation prevents blink artefacts from contaminating the bins; conversion to diameter makes the units match the requested variable name.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global quantile bins, computed exactly as for running speed from the pool of within-trial raw diameter samples across all eligible sessions (NaN/blink samples excluded from the pool). Realised distribution: `[0.203, 0.206, 0.193, 0.187, 0.212]` — slightly off uniform because the edges are computed on the raw 30 Hz samples while the emitted values are interpolated bin centres.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
if running_pool.size < 5 or pupil_pool.size < 5:
    raise RuntimeError("Not enough running or pupil samples to compute 5-bin discretization")
pupil_edges = compute_bin_edges(pupil_pool, 5)
```

iii. Step 5 key decision 10 (global, not per-session, bin edges). Note the consequence, visible in the verification log: because edges are global across 38 mice, whole sessions can sit in a single quintile (per-session pupil ranges such as `[4.0, 4.0]` appear), i.e. pupil is largely a between-animal/between-session variable rather than a within-trial one.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto the identical 100 ms bin centres of the same trial window, therefore column-for-column aligned with the neural matrix.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same justification as 5-d; all streams are hardware-synchronized, so interpolation onto the shared ophys-anchored grid preserves alignment. The `--show-processing` figure plots the raw pupil trace, the rebinned trace and the discretized bins on a common axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. NaNs are coerced to `False` before the boolean cast.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
hit = np.nan_to_num(trials["hit"], nan=0.0).astype(bool)
miss = np.nan_to_num(trials["miss"], nan=0.0).astype(bool)
false_alarm = np.nan_to_num(trials["false_alarm"], nan=0.0).astype(bool)
correct_reject = np.nan_to_num(trials["correct_reject"], nan=0.0).astype(bool)
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

iii. Step 1 documents the SDK logic in `Trial._get_trial_data()` (`correct_reject = catch and not false_alarm`; auto-rewarded trials are explicitly prevented from being counted as hit/miss/FA/CR), so after excluding aborted and auto-rewarded trials these four flags form a complete, mutually exclusive partition. The AI turned that into a hard assertion rather than a silent fallback.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code (0–3) is broadcast across all `T` bins of the trial so that the output block stays a uniform `(5, T)` matrix. Pooled distribution: `hit 0.182, miss 0.692, false_alarm 0.010, correct_reject 0.115` — the miss-heavy skew reflects the inclusion of passive sessions.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins,
    pupil_bins,
    outcome_series,
])
```

iii. Step 5 key decision 11: "Trial outcome will be repeated across time bins: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form." Step 10 lists an outcome sanity check: "per session, converted trial-outcome counts must match raw `hit`/`miss`/`false_alarm`/`correct_reject` counts after filtering."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, mostly fail-loud rather than fail-soft:
- **Missing eye tracking** (3 experiments): the whole experiment is excluded and recorded in `metadata['excluded_sessions']` with a reason string, rather than imputing pupil values.
- **Blinks**: masked to NaN and interpolated over.
- **NaN/absent boolean flags** in the trial table: `np.nan_to_num(..., nan=0.0)` → `False`.
- **Non-finite interpolation output**: raises `ValueError` (never triggered in the full run).
- **Inconsistent trial outcome flags**: raises `ValueError` (never triggered).
- **Behavioural coverage gaps at trial edges**: constant hold (`left=v[0], right=v[-1]`) instead of NaN.
- **Degenerate trials**: `build_bin_centers` guarantees at least one bin; sessions with <2 trials or <2 finite pupil samples are excluded; the run aborts if fewer than 2 sessions survive.
- **Byte/unicode string decoding** and heterogeneous HDF5 dtypes are handled by `decode_scalar` / `decode_string_array`.
- **Not handled explicitly**: bins containing no ophys frame (common in the 93.2 ms-per-frame multiscope experiments, and for any part of a trial extending past the end of the recording) are silently left at 0 rather than being clipped or marked. This is the mechanism behind the 3,947 "all neural data is zero" verification warnings (4.68 % of trials), which the AI attributed entirely to genuine event sparsity.

ii.
```python
excluded_reason = None
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
```
```python
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
```
```python
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
if starts.size == 0:
    starts = np.array([start_time], dtype=np.float64)
```
```python
"excluded_sessions": [
    {"experiment_id": p.experiment_id, "session_type": p.session_type, "reason": p.excluded_reason}
    for p in excluded
],
```

iii. Step 5 key decision 4: sessions without eye tracking are "dropped rather than fabricating pupil values." Step 10: "`all neural data is zero` warnings remained after the rerun: investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition." The AI verified three of the warned trials by reconstructing them from raw NWB with `np.allclose()`.

## 9-a. What are the most time-consuming steps of the code?

i. The AI instrumented both passes. Full run: preview pass 65.0 s, conversion pass 686.0 s, total 757.7 s. The conversion pass dominates, and within it the cost is (a) reading the `(n_frames, n_rois)` event-detection matrix into memory for each of 281 files (files are 48k–150k frames × up to 666 ROIs) and (b) the Python-level per-bin summation loop over ~84,313 trials × ~86 bins ≈ 7.2 M iterations. Per-session elapsed time is printed for every experiment.

ii.
```python
session_start = time.perf_counter()
...
stats = {"experiment_id": preview.experiment_id, "trial_count": len(neural_trials),
         "neuron_count": int(event_data.shape[1]),
         "elapsed_sec": int(round(time.perf_counter() - session_start))}
```
```python
log(f"Preview pass completed in {time.perf_counter() - preview_start:.1f}s")
...
log(f"Conversion pass completed in {time.perf_counter() - convert_start:.1f}s")
```

iii. Step 6: "Preview pass avoids loading neural event matrices. Conversion uses direct dataset reads and processes one session at a time to cap peak memory." Step 7 estimated "~5.95 s / session … conservative full-conversion upper bound ~28.2 min," and the actual full run came in at 12.6 min, under the 15-minute guidance in the instructions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- The per-bin neural summation `for b in range(T)` in `convert_session` — replaceable by a single `np.add.reduceat` (or a cumulative sum differenced at the bin boundaries) over the frame axis, eliminating ~7.2 M Python iterations. The AI identified this one itself.
- `make_image_series`: loops over every overlapping presentation row and builds a full-length boolean mask `(centers >= start) & (centers < stop)` for each. A single `np.searchsorted(presentation_starts, centers)` lookup would label all bins at once.
- `find_presentation_rows`: evaluates a mask over the *entire* session presentation array (~4,800 rows) once per trial, i.e. ~400 M element comparisons over the dataset; two `searchsorted` calls on the pre-sorted `start_time` array would be O(log n).
- `build_trial_specs`: a per-trial Python loop plus a `TrialSpec` dataclass construction, where the flag logic is already fully vectorized above the loop (`outcome_idx` could be `np.argmax` over a stacked boolean array).

ii.
```python
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```
```python
for idx in row_idx:
    ...
    in_window = (centers >= start) & (centers < stop)
    if not np.any(in_window):
        continue
```
```python
mask = (starts < stop_time) & (ends > start_time)
return np.flatnonzero(mask)
```

iii. The AI only acknowledged the first: Step 6, "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU." The other three are not mentioned in CONVERSION_NOTES. Since the total runtime met the instruction's 15-minute target, no further optimisation was pursued.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB file is opened and parsed **twice**. `read_trials`, `build_trial_specs`, `read_task_presentations`, and the running-speed and pupil reads (including the blink mask and the `2·sqrt(area/π)` conversion over the full ~135k-sample eye trace) all run once in `session_preview` and again in `convert_session`. Additionally, `find_presentation_rows` re-scans the whole presentation table once per trial, and `collect_time_window_values` re-slices running/pupil per trial in the preview pass. The preview also holds the concatenated raw running and pupil samples for all 281 sessions in memory just to compute two sets of quantiles.

ii.
```python
# preview pass
with h5py.File(path, "r") as f:
    trials = read_trials(f); specs = build_trial_specs(trials)
    presentations = read_task_presentations(f)
    running_times = ...; running_values = ...
    pupil_area[blink] = np.nan; pupil_diameter = pupil_area_to_diameter(pupil_area)
```
```python
# conversion pass — the same reads again
with h5py.File(preview.path, "r") as f:
    running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
    pupil_area[likely_blink] = np.nan
    pupil_diameter = pupil_area_to_diameter(pupil_area)
    trials = read_trials(f)
    presentations = read_task_presentations(f)
```

iii. Step 6 states this is deliberate: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known." The trade is explicit — the global quantile edges and the global image vocabulary must be known before any trial can be written, and caching the per-session behavioural/trial tables between passes was not implemented. The cost measured is 65 s out of 758 s (~9 %).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:
- **Empty input arrays**: an `np.empty((0, T), dtype=np.float32)` is allocated and stored for all 84,313 trials even though the task specifies no decoder inputs.
- **`ophys_session_id`** is read from every file and stored on `SessionPreview` but never used (this is precisely the field that would have been needed to group imaging planes into sessions — see 1-c).
- **`TrialSpec.change_time`, `is_go`, `is_catch`** are computed for all 84,313 trials but used only by the `--show-processing` plots; `change_time` never influences any output.
- **Trials-table columns read but unused**: `initial_image_name`, `change_image_name`, `change_frame`, `catch`, `id` (used only for a count).
- **Presentation columns read but unused**: `is_sham_change`, `trials_id`, `active`.
- **`load_neural_events` returns `event_rois[valid_mask]`**, which the caller discards (`event_data, _ = ...`).
- **Preview-pass pupil conversion**: `pupil_area_to_diameter` is applied to the entire ~135k-sample session trace purely to build the quantile pool, then recomputed in the conversion pass.
- **`raw_trial_count`** is tracked for every session but only rendered in the optional plot.

ii.
```python
input_trial = np.empty((0, T), dtype=np.float32)
...
input_trials.append(input_trial)
```
```python
event_data, _ = load_neural_events(f)        # second return value discarded
```
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])   # never used
```
```python
change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
is_go=bool(go[idx]),
is_catch=bool(catch[idx]),
```

iii. The AI documented the empty-input decision (Step 5 key decision 12: "No decoder inputs: `input_names` will be empty and every `input` trial entry will be an empty 2D array with the same time dimension as the corresponding trial") as a format requirement rather than waste. The other items are not discussed in CONVERSION_NOTES; they are low-cost reads kept for diagnostics and sanity checking (e.g. `change_time` is described in Step 5 as retained "for sanity checks"). None of them materially affects runtime.
