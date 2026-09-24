# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the locally cached NWB/HDF5 files directly with `h5py`. It globs every `behavior_ophys_experiment_*.nwb` file under `/app/data/.../behavior_ophys_experiments` (284 files) and treats each file as one unit of data. No filtering by `project_code` is applied, so both `VisualBehavior` (239 experiments) and `VisualBehaviorMultiscope` (45 experiments) files are ingested. Loading is done in two passes: a lightweight "preview" pass that opens every file and reads only trials, stimulus presentations, running, and eye-tracking (no neural matrices), and a "conversion" pass that re-opens each eligible file and reads the event-detection matrix, timestamps, behaviour, trials, and presentations. Within each file the AI reads `/intervals/trials`, all `/intervals/*_presentations` groups that contain `image_name`, `/processing/ophys/event_detection`, `/processing/ophys/dff/traces/timestamps`, `/processing/running/speed`, and `/acquisition/EyeTracking/*`.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
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

```python
def convert_session(preview, image_to_idx, running_edges, pupil_edges, ...):
    with h5py.File(preview.path, "r") as f:
        ophys_timestamps = np.asarray(f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64)
        event_data, _ = load_neural_events(f)
        running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
        running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
        pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
        pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
```

iii. From CONVERSION_NOTES Step 6: "Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment." The AI documented (Step 10, "Reference code comparison") that it reads "the same underlying NWB tables/fields" that the SDK object model (`BehaviorOphysExperiment`, `Trials`, `Presentations`, `CellSpecimens`, `RunningSpeed`, `EyeTrackingTable`) would read, so the direct-HDF5 path is a faithful substitute for the SDK path. The two-pass design was justified as "intentional to avoid storing large neural matrices before global percentile/bin definitions are known." The AI observed the two project codes present locally (trajectory step 48: `project_codes ['VisualBehavior', 'VisualBehaviorMultiscope']`) but never stated a reason for including both.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `/general/subject/subject_id` strings read from each NWB file. Subjects are registered in first-encountered (file-sorted) order, and each converted session gets an index into that subject list. 38 subjects result.

ii.
```python
subject_id = decode_scalar(f["/general/subject/subject_id"][()])
```

```python
for i, preview in enumerate(eligible, start=1):
    if preview.subject_id not in subject_to_idx:
        subject_to_idx[preview.subject_id] = len(subjects)
        subjects.append(preview.subject_id)
    ...
    subject_idx.append(subject_to_idx[preview.subject_id])
```

iii. CONVERSION_NOTES Step 5 mapping table: "`/general/subject/subject_id` or metadata `mouse_id` → `subjects`, `subject_idx`; String subject identifiers with session-level index mapping; Session order follows converted session order." The AI cross-checked the count (38 local mice) against the metadata CSVs in Step 2 and Step 9.

## 1-c. How are the data split into sessions?

i. One converted "session" = one NWB **experiment** file = one imaging plane. The AI never groups experiments by `ophys_session_id`, even though it reads that field into the `SessionPreview` dataclass (it is never used afterwards). For the 239 single-plane `VisualBehavior` experiments this is equivalent to one session per file, but the 45 local `VisualBehaviorMultiscope` experiments come from only 8 real recording sessions of a single mouse (e.g. experiments 960410023/26/28/38/42 all share `ophys_session_id 959458018`), so those 8 sessions are emitted as 45 separate "sessions" with identical trial timing, identical behaviour, and identical output labels. The verification log shows the consequence: "Subject 457841: 45 sessions", vs 5–11 for every other mouse.

ii.
```python
@dataclass
class SessionPreview:
    path: Path
    experiment_id: int
    ophys_session_id: int      # read but never used for grouping
    ...

for i, preview in enumerate(eligible, start=1):
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```

iii. The AI did not explicitly justify per-plane sessions. The closest statements are CONVERSION_NOTES Step 1 ("One experiment corresponds to one imaging plane in one session") and Step 5 decision 5 ("Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions"), plus Step 2's observation that local sessions contain 1–7 experiments. Everywhere else the notes treat "session" and "experiment file" as synonyms ("`281` eligible local experiment files" reported in the Sessions row of the consistency table).

## 1-d. How are the data split into trials?

i. Trials come from the NWB `/intervals/trials` table (the SDK `Trials` object serialized to NWB). Each kept trial spans `start_time` to `stop_time` (variable length, ~7–13 s, 71–127 bins). Aborted and auto-rewarded trials are dropped, so only go and catch trials survive. The trial window is then tiled with 100 ms bins.

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

iii. CONVERSION_NOTES Step 4: "Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly." Step 5 decision 2: "Use SDK-valid trials only: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed and consistent with the trial table semantics." Step 5 decision 7: "Alignment event = trial start. The trial itself is the natural unit requested by the user." The AI verified in Step 1 that trials are defined by the SDK from the behaviour `trial_log` using consecutive `trial_start` frames, not inferred from image changes.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filters: drop `aborted`, drop `auto_rewarded`, and hard-fail (`ValueError`) if a surviving trial does not have exactly one of `hit/miss/false_alarm/correct_reject` set. Session-level filters: exclude a session if (a) eye tracking is absent (`/acquisition/EyeTracking/pupil_tracking/area_raw` missing, 3 sessions), (b) fewer than 2 valid go/catch trials, or (c) fewer than 2 finite pupil samples inside trial windows. No filter on `change_time` validity and no clipping of trials that would run past the end of the ophys recording. Result: 281/284 sessions, 84,313 trials.

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
if preview.excluded_reason is None:
    eligible.append(preview)
else:
    excluded.append(preview)
```

iii. Step 5 decision 2 (aborted/auto-rewarded per the instructions), decision 3 ("Keep passive sessions if they have valid GO/CATCH trials: Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs"), and decision 4 ("Exclude sessions with missing eye tracking: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values"). The ≥2-trial rule follows the target-format requirement "There needs to be at least two trials within each session." The AI also ran a planned sanity check that "per session, valid converted trial count must equal raw trial count minus aborted minus auto_rewarded" and reported it passed (Step 10, check 4).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `/processing/ophys/event_detection/data` — the L0 event-detection magnitudes per ROI per ophys frame — not dF/F. ROI identity comes from `/processing/ophys/event_detection/rois`, and the ROI quality flag from `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`. The ophys time base is taken from `/processing/ophys/dff/traces/timestamps` (same length as the event matrix).

ii.
```python
def load_neural_events(f: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
    event_rois = np.asarray(f["/processing/ophys/event_detection/rois"], dtype=np.int64)
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
    return event_data, event_rois[valid_mask]
```

iii. Step 4 discrepancy table, "Neural representation" row: "SDK loads both dF/F traces and event-detection outputs ... whitepaper describes dF/F processing; paper states analyses were performed on discrete calcium events → prefer event-detection outputs for neural activity because the analysis paper explicitly uses them, while preserving SDK ROI filtering and timestamps." Step 5 decision 1 repeats this: "The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events. This is the closest match to the paper while still using the SDK-defined loading and ROI filtering." The mapping table also records "Use raw events, not visualization-only `filtered_events`." Trajectory step 118 shows the AI explicitly checked whether the event matrix ROI order matches the dF/F ROI order before writing the loader.

## 2-b. How is the `neural` data processed?

i. Two operations: (1) drop invalid ROIs, (2) rebin from native ophys frames to 100 ms trial-relative bins by **summing** event magnitudes of all ophys frames whose timestamp falls in `[bin_start, bin_end)`. No normalization, smoothing, z-scoring, or dF/F recomputation. Output dtype float32, shape `(n_neurons, T)` per trial.

ii.
```python
for spec in preview.trial_specs:
    starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
    frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
    frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
    T = centers.size
    n_neurons = event_data.shape[1]

    neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
    for b in range(T):
        lo = int(frame_starts[b]); hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. Step 5 mapping table: "Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions." Metadata records the rule explicitly: `"binning_rule": "sum event magnitudes within each 100 ms trial bin"`. Step 1 notes that dF/F needs no recomputation ("`DFFTraces.from_nwb` ... Loads already-computed dF/F traces; no dF/F recomputation is required by the SDK") and that `filtered_events` is only a visualization smoothing, so raw events were used.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the SDK's ROI-validity filter: ROIs with `valid_roi == False` are dropped. 42,147 local ROI rows → 41,871 neurons in the eligible sessions. No further per-neuron QC (no SNR, event-rate, or activity threshold), and no trials/neurons are removed for being all-zero — 3,947 trials (4.68%) have an entirely zero neural matrix and were deliberately kept.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. Step 4: "SDK filters to `valid_roi` by default ... Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid." Step 1 lists `CellSpecimens.__init__ with exclude_invalid_rois` as the CURATION step in the reference code. For the all-zero trials, Step 10: "investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition." The AI verified three of the warned trials against raw NWB with `np.allclose()`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is **trial start**. The 100 ms bin grid begins exactly at `trials.start_time` and runs to `stop_time`; ophys frames are assigned to bins by `np.searchsorted` on the ophys timestamps, so the neural data and every output share the same trial-relative grid. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable trial length). There is no re-alignment to `change_time`.

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

iii. Step 5 decision 7: "Alignment event = trial start. The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`)." Step 10 comparison table: "Alignment matches the reference time bases; the only added step is the decoder-required common 100 ms rebinning," relying on the whitepaper statement that all streams are hardware-synchronized on a single 100 kHz board.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms for every trial and session (`metadata['time_bin_size'] = 100.0`). This is an explicit rebinning: single-plane data are acquired at ~31 Hz (32.3 ms/frame, ~3 frames per bin) and the multiscope data at ~10.7 Hz (93.2 ms/frame, ~1 frame per bin), and both are mapped onto the same 100 ms grid. Trials have 71–127 bins (mean 85.8). Note that because the grid is built with `np.arange(start, stop, 0.1)`, the final bin of each trial is truncated at `stop_time` and can be shorter than 100 ms.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
"neural_representation": "ophys event-detection magnitudes",
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. Step 5 decisions 5 and 6: "Common time base via uniform rebinned trial bins: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions"; "Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams. This choice will be validated during sample conversion." Step 10 comparison table frames the difference from the reference as intentional: "Native SDK data remain at native sample rates; the papers do not prescribe a common decoder bin size ... Difference is intentional and required by the target format, not a mismatch in source processing."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The stimulus-presentation table (`/intervals/Natural_Images_..._presentations`), using `image_name`, `start_time`, `stop_time`, and `omitted`. It is **not** derived from the trials table's `initial_image_name`/`change_image_name`. All `/intervals/*` groups that have an `image_name` column are merged and sorted by start time (in practice only the natural-images group qualifies; `natural_movie_one_presentations` and `spontaneous_presentations` have no `image_name` and are skipped).

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
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

iii. Step 5 mapping table: "Stimulus presentation `image_name` + presentation timing + omission state → `output[0]` (`image_identity`); Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods." The AI identified `Presentations.from_stimulus_file` in Step 1 as the reference function that builds this table with `image_name`, `start_time`, `is_change`, `omitted`, `flashes_since_change`, `trials_id`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A per-bin categorical series. The default value for every bin is a dedicated `"gray"` class (index 0); bins whose centre falls inside a non-omitted image presentation get that image's index. Omitted flashes stay `gray`. The vocabulary is global across all eligible sessions: `["gray"] + sorted(unique image names)` = 17 classes (8 image-set-A + 8 image-set-B + gray). Because the flash is 250 ms and the cycle 750 ms, 67% of all bins are `gray` and each image occupies ~2% of bins.

ii.
```python
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    change_series = np.zeros(centers.shape, dtype=np.int16)
    for idx in row_idx:
        start = float(presentations["start_time"][idx]); stop = float(presentations["stop_time"][idx])
        if stop <= start:
            continue
        in_window = (centers >= start) & (centers < stop)
        if not np.any(in_window):
            continue
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
```

```python
image_names = sorted({img for p in eligible for img in p.unique_images})
image_values = ["gray"] + image_names
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. Step 5 decision 8: "Image identity will include a `gray` class: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin." In Step 10 the AI found and fixed a real bug here — the literal string `omitted` had been entering the image vocabulary as an unused class — by filtering it out in `unique_nonempty_images()`, which brought the class count to the expected 17 and matched "the paper/whitepaper task description of two eight-image sets."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same `centers` array used to build the neural bins for that trial, so element *t* of the image series corresponds to element *t* of the neural matrix by construction. A bin is assigned an image if its **centre** lies in `[presentation.start_time, presentation.stop_time)`; presentations are pre-selected by interval overlap with the trial window.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

```python
def find_presentation_rows(presentations, start_time, stop_time):
    starts = presentations["start_time"]; ends = presentations["stop_time"]
    mask = (starts < stop_time) & (ends > start_time)
    return np.flatnonzero(mask)
```

iii. Step 10 comparison table (temporal alignment row): stimulus, running, eye and ophys streams are all sync-derived, so projecting them onto a single ophys-anchored bin grid preserves alignment. The AI validated this visually with the `--show-processing` plots (raw presentation intervals drawn as spans with the binned image/change series overlaid) and numerically in Step 10 check 2, where output rows for five specific trials were reconstructed from raw NWB and compared with `np.allclose()`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` (and `omitted`) columns of the stimulus-presentation table, together with the presentation start/stop times. `is_change` is true only for genuine image changes; catch trials are flagged `is_sham_change` and therefore produce no change bins, matching the reference's "go trials only" behaviour.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 mapping table: "Stimulus presentation `is_change` + presentation timing → `output[1]` (`image_change`); Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0 ... trial `change_time` for sanity checks." The AI listed a planned sanity check: "verify the converted binary change series turns on only for the changed-image flash and matches trial `change_time` / presentation `is_change`," and reported it passing in Step 10.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-bin series, 1 for the bins covering the changed image's 250 ms presentation and 0 elsewhere (≈2–3 bins of ~86 per trial; 2.63% of all bins are 1). It is *not* extended through the following grey interval, and it is not restricted to a single bin. Omitted flashes cannot be changes (guarded explicitly, and the raw data never omits a change).

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

iii. Step 5 decision 9: "Image-change target will mark the changed-image presentation, not only a single instant: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure." Step 12 notes the resulting sparsity (2.63% positive) and uses it to interpret the modest 0.604 balanced accuracy.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed or applied — the variable is already binary. It is stored as int16 with `output_values[1] = ["no_change", "change"]`, i.e. class 0 = no change, class 1 = change.

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

iii. The instructions define image change as "binary variable. Have value of 1 right after a change in image identity, otherwise 0," and the raw `is_change` flag is already boolean, so the AI carried it through directly. Verification confirmed the realized range `image_change: [0.0, 1.0]` with distribution `{no_change 0.974, change 0.026}`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: it is produced inside the same `make_image_series()` call, on the same `centers` grid as the trial's neural matrix, using presentation intervals selected by overlap with the trial window. No shift or lag is applied relative to the neural bins.

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

iii. Same justification as 3-c: all streams are hardware-synchronized, the change flags come from the same presentation table used for image identity, and Step 10's raw-NWB reconstruction check compared the whole `output` matrix (including the change row) for five trials with `np.allclose()`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `/processing/running/speed/data` with `/processing/running/speed/timestamps` — the SDK's processed running speed in cm/s on the stimulus time base. The unfiltered variant (`speed_unfiltered`) and the raw encoder signal (`dx`) were noted as present but not used.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 1 identifies `RunningSpeed.from_stimulus_file` as the reference processing function ("Computes running speed aligned to stimulus timestamps with zero monitor delay; includes polarity correction if mean speed is implausibly negative"), and Step 3 records the whitepaper's statement that running speed is "computed from wheel encoder voltage, with unwrapping, wrap detection, and derivative-based conversion to linear speed ... whitepaper explicitly points to AllenSDK running-processing code." The AI therefore used the already-processed SDK output rather than recomputing from `dx`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the native ~60 Hz running trace onto the trial's 100 ms bin centres (constant extrapolation at the edges, NaNs dropped before interpolation), then discretization into 5 global quantile bins. The quantile edges are computed once, in the preview pass, by pooling *raw* running samples that fall inside the valid trial windows of all eligible sessions.

ii.
```python
def interpolate_series(times, values, query_times):
    finite = np.isfinite(times) & np.isfinite(values)
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
```

```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 5 mapping table: "Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins ... Use global bin edges computed over all included samples in included sessions/trials." Step 5 decision 10: "Running and pupil bin edges will be global, not per-session: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`." The planned sanity check "global running and pupil bins should each contain roughly 20% of included samples by construction" was confirmed: `[0.199, 0.201, 0.199, 0.200, 0.200]`.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins. `compute_bin_edges` takes the 20/40/60/80th percentiles of the pooled raw running samples; `discretize_with_edges` applies `np.digitize` with those 4 interior edges (right-open) and clips to `[0, 4]`. Value names are `q1..q5`. Any non-finite interpolated value would raise rather than be silently binned.

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

```python
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
```

iii. Directly follows the Decoder Task instruction "Running speed, discretized into five equal percentile bins." The AI verified the achieved marginal distribution in Step 9 against the by-construction expectation of 20% per bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated at the same `centers` used for the trial's neural bins, so row 2 of `output` is element-wise aligned with the neural matrix. Running lives on the stimulus/behaviour clock, which the whitepaper states is hardware-synchronized with the ophys clock, so interpolation across clocks is treated as valid.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
output_trial = np.vstack([image_series, change_series, running_bins, pupil_bins, outcome_series])
```

iii. Step 4: "SDK keeps separate stimulus, running, eye, and ophys timestamp streams; all are sync-derived ... Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases." The `--show-processing` plots overlay the raw running trace, the interpolated values at bin centres, and the discretized bins on the same time axis specifically to demonstrate the absence of a lag (Step 12: "stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch").

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `/acquisition/EyeTracking/pupil_tracking/area_raw` (raw pupil ellipse area) with `/acquisition/EyeTracking/eye_tracking/timestamps`, plus `/acquisition/EyeTracking/likely_blink/data` to mask blinks. Area is converted to an equivalent diameter with `d = 2*sqrt(area/pi)`. Sessions where `area_raw` is absent are excluded entirely (3 sessions).

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)

def pupil_area_to_diameter(area):
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. Step 3: "whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter" — so inverting the area gives back the whitepaper's diameter. Step 5 mapping table: "Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins. Requires eye-tracking availability. Sessions with missing eye tracking will be excluded." Step 1 identifies `EyeTrackingTable.from_nwb` as the reference function that "recomputes likely blinks, and filters blink frames," which the AI mirrors by NaN-ing blink samples.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples → NaN → dropped; remaining samples converted to diameter; linearly interpolated onto the trial's 100 ms bin centres (so blink gaps are bridged by interpolation, edges held constant); then discretized into 5 global quantile bins whose edges come from the pooled raw (blink-masked) diameter samples inside valid trial windows across all eligible sessions.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Same as 5-b/6-a: Step 5 decision 10 (global edges), and the blink handling follows `EyeTrackingTable`'s blink filtering identified in Step 1. The AI checked the resulting marginal distribution `[0.203, 0.206, 0.193, 0.187, 0.212]` against the 20%-per-bin expectation and accepted the small deviation as an effect of binning/interpolation relative to the raw-sample quantiles.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identically to running speed: 20/40/60/80th percentiles of the pooled sample, `np.digitize` right-open, clipped to `[0, 4]`, value names `q1..q5`.

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
...
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
```

iii. Follows the Decoder Task instruction "Pupil diameter, discretized into five equal percentile bins," with the same global-edge rationale as running speed (Step 5 decision 10). Because quantile binning is rank-preserving, the area→diameter conversion does not change the resulting bins — the AI did not note this.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed: interpolated at the trial's bin centres, giving row 3 of `output` element-wise aligned with the neural matrix. Eye-camera timestamps are already synchronized to the common clock, so no additional offset is applied.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
output_trial = np.vstack([image_series, change_series, running_bins, pupil_bins, outcome_series])
```

iii. Step 4 ("Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases") and the Step 5 planned sanity check "compare converted pupil-diameter bins to raw eye-tracking-derived diameter after interpolation at selected timestamps," reported as passing in Step 10 check 2.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. Aborted and auto-rewarded trials (which the SDK never labels with these flags) are already removed.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

iii. Step 1 documents the SDK logic in `Trial._get_trial_data()` (including "Auto-rewarded trials are explicitly prevented from being counted as hit/miss/false_alarm/correct_reject"), and Step 3 records "Trial structure consists of GO and CATCH trial types, which combine with behavior to yield HIT, MISS, FALSE ALARM, and CORRECT REJECTION outcomes." The strict one-hot assertion was added as an edge-case guard; Step 5 also planned a check that "converted trial-outcome counts must match raw `hit`/`miss`/`false_alarm`/`correct_reject` counts after filtering."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome index (0–3) is broadcast to a constant series across every bin of the trial so that all five output rows are time-varying with the same length. `output_values[4] = ["hit","miss","false_alarm","correct_reject"]`. Realized distribution: hit 18.2%, miss 69.2%, false alarm 1.0%, correct reject 11.5% (miss-heavy because passive sessions are retained).

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

iii. Step 5 decision 11: "Trial outcome will be repeated across time bins: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form." Step 12 discusses the imbalance ("`trial_outcome` is imbalanced ... and is a four-class static label, making it harder than the paper's qualitative two-class hit/miss analyses").

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Mixed strategy — some cases are handled gracefully, others are deliberately fatal:
- **Missing eye tracking** (3 files): the whole session is excluded up front, with the reason recorded in `metadata['excluded_sessions']`.
- **Blinks**: masked to NaN, dropped before interpolation, then bridged by linear interpolation.
- **NaNs in boolean trial columns**: `np.nan_to_num(..., nan=0.0).astype(bool)` treats missing flags as False.
- **Out-of-range interpolation**: constant edge extrapolation (`left=v[0], right=v[-1]`) rather than NaN, so no missing values leak into the discretized outputs.
- **Degenerate trial windows**: `build_bin_centers` falls back to a single bin if `stop <= start + bin`; bins containing no ophys frames stay zero.
- **Degenerate sessions**: <2 valid trials or <2 finite pupil samples → excluded; <2 eligible sessions overall → `RuntimeError`.
- **Unexpected data**: a trial without exactly one outcome flag, or a non-finite interpolated running/pupil value, raises and aborts the entire run. There is no per-session `try/except`, so one malformed file would stop the whole conversion.
- **All-zero neural trials** (3,947, 4.68%): kept, after verifying against raw NWB that they are genuinely zero.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
...
if starts.size == 0:
    starts = np.array([start_time], dtype=np.float64)
...
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
...
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
...
if len(eligible) < 2:
    raise RuntimeError("Need at least 2 eligible sessions after filtering")
...
"excluded_sessions": [
    {"experiment_id": p.experiment_id, "session_type": p.session_type, "reason": p.excluded_reason}
    for p in excluded
],
```

iii. Step 5 decision 4: sessions without eye tracking are "dropped rather than fabricating pupil values." Step 10 check 5: "Checked missing-eye-tracking handling (`3` excluded sessions), verified no experiments had fewer than two valid go/catch trials after filtering, validated a minimum-length trial with raw reconstruction, and confirmed that all-zero neural warnings arise from genuine zero-event trials in the raw event-detection matrices rather than off-by-one errors at trial boundaries." The fail-loud guards are consistent with the instruction to "be critical of results after every step" — the AI preferred an explicit crash over silently encoding bad values.

## 9-a. What are the most time-consuming steps of the code?

i. Measured on the full run: preview pass 65.0 s, conversion pass 686.0 s, total 757.7 s (~12.6 min). Within the conversion pass the dominant costs are (1) reading the full `(n_frames, n_rois)` event matrix for each of 281 files off a 247 GB dataset, and (2) the pure-Python per-bin rebinning loop (`for b in range(T)` × ~86 bins × 84,313 trials ≈ 7.2 M iterations, each a NumPy slice-sum). Per-session times scale with neuron count (1 s for ~8-neuron sessions, 6 s for ~600-neuron sessions). The preview pass adds a second full open of all 284 files.

ii.
```python
neural_trial = np.zeros((n_neurons, T), dtype=np.float32)
for b in range(T):
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

```python
log(f"Preview pass completed in {time.perf_counter() - preview_start:.1f}s")
...
log(f"Conversion pass completed in {time.perf_counter() - convert_start:.1f}s")
...
stats = {..., "elapsed_sec": int(round(time.perf_counter() - session_start))}
```

iii. Step 6: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known. Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU." Step 7 estimated "~5.95 s/session ... conservative full-conversion upper bound ~28.2 min," and the actual 12.6 min came in under the 15-minute guidance in the instructions, so no further optimization was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, all vectorizable:
- The per-bin neural rebinning loop — replaceable with `np.add.reduceat` or a cumulative-sum difference over the frame axis (the single biggest win).
- `make_image_series`, which loops over every presentation overlapping the trial and builds a full-length boolean mask `(centers >= start) & (centers < stop)` on each iteration — replaceable with one `np.searchsorted` of `centers` into the presentation boundaries.
- `find_presentation_rows`, which builds a mask over the whole ~4,800-row presentation table once per trial instead of using a sorted-search on `start_time`.
- `build_trial_specs`, which iterates over all raw trials in Python to build dataclasses after already computing the boolean arrays vectorized.
The AI acknowledged only the first of these.

ii.
```python
for b in range(T):                                   # per-bin neural sum
    ...
for idx in row_idx:                                  # per-presentation mask
    in_window = (centers >= start) & (centers < stop)
    ...
mask = (starts < stop_time) & (ends > start_time)    # full-table scan per trial
...
for idx in range(raw_count):                         # per-trial spec construction
    if aborted[idx] or auto_rewarded[idx]:
        continue
```

iii. Step 6 ("Code inefficiencies identified") explicitly flags the per-bin loop as a deliberate memory/CPU trade-off: "rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU." No justification is given for the other three loops; the AI's speed-ups instead targeted I/O ("Preview pass avoids loading neural event matrices", "Sample-mode preview now short-circuits once 2 eligible sessions are found").

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and parsed twice. `read_trials`, `build_trial_specs`, `read_task_presentations` (including the concatenate + argsort of the whole presentation table), the running speed arrays, the eye-tracking arrays, and the blink-masking/area→diameter conversion are all executed once in `session_preview` and again in `convert_session`. `collect_time_window_values` also re-derives trial windows that `build_trial_specs` has just computed. In addition, `find_presentation_rows` rescans the entire presentation table once per trial, and `read_task_presentations` re-sorts all presentations even though they are already time-ordered.

ii.
```python
# in session_preview
trials = read_trials(f); specs = build_trial_specs(trials)
presentations = read_task_presentations(f)
running_times = ...; running_values = ...
pupil_area[blink] = np.nan; pupil_diameter = pupil_area_to_diameter(pupil_area)

# again in convert_session, for the same file
trials = read_trials(f)
presentations = read_task_presentations(f)
running_times = ...; running_values = ...
pupil_area[likely_blink] = np.nan; pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. Step 6: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known." The trade-off is real — global quantile edges and the global image vocabulary genuinely require a first pass — but the AI did not consider caching the (small) preview-parsed tables to avoid re-reading them, and it did not note the per-trial rescans of the presentation table.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations have no effect on the saved dataset:
- **`pupil_area_to_diameter`**: quantile binning is rank-preserving and `2*sqrt(area/pi)` is strictly monotone in area, so the sqrt conversion produces exactly the same five bins as binning the raw area. Only the (unsaved) continuous values differ.
- **`trials` and `presentations` re-read in `convert_session`**: `trials` is used only by the plotting function, so on a normal `--full` run (no `--show-processing`) the entire trials table is read and decoded for nothing.
- **`load_neural_events` returns `event_rois[valid_mask]`**, which the caller discards (`event_data, _ = ...`).
- **`build_bin_centers` computes `ends` and `widths`**; `widths` is only used to form `centers`, and for all but the last bin equals the bin size.
- **`read_task_presentations` keeps `is_sham_change`, `trials_id` and `active`** columns that are never used, and re-sorts an already-sorted table.
- **`SessionPreview` fields** `ophys_session_id`, `raw_trial_count`, `has_eye_tracking`, `unique_images` (partly) are carried but barely used; `ophys_session_id` is never used at all.
- **`sort_files_for_mode`** reads the full `ophys_cells_table.csv` to rank files by cell count, which only matters in `--sample` mode.

ii.
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)   # monotone → identical quantile bins
...
event_data, _ = load_neural_events(f)                 # returned roi ids discarded
...
trials = read_trials(f)                               # only consumed by plot_processing_summary
presentations = read_task_presentations(f)
...
starts, ends, centers = build_bin_centers(...)        # `ends` used only for searchsorted, `widths` only for centers
```

iii. The AI did not identify any of these as wasted work; CONVERSION_NOTES' "Code inefficiencies identified" section lists only the double file pass and the per-bin loop. The area→diameter conversion was justified on interpretability grounds (Step 3: the whitepaper treats the ellipse major axis as diameter and derives area from it, so inverting recovers the diameter), which is a reasonable motivation even though the transform is invisible in the discretized output.
