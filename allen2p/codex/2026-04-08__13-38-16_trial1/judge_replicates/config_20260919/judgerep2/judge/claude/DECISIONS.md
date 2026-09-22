# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK `VisualBehaviorOphysProjectCache` API. It discovered `pynwb` was incompatible with the installed HDF5/namespace stack in this environment (step 50 of the trajectory: `pynwb.NWBHDF5IO(...).read()` raised), so it reads the released NWB/HDF5 files directly with `h5py`, replicating the SDK's field semantics by hand. Data discovery is a glob over every `behavior_ophys_experiment_*.nwb` file in `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` (284 files). Project metadata CSVs (`ophys_experiment_table.csv`, `ophys_cells_table.csv`) are only consulted for sample-mode ordering, **not** for filtering. In particular, no `project_code` filter is applied, so the 45 `VisualBehaviorMultiscope` experiments present in the local subset are loaded alongside the 239 `VisualBehavior` experiments.

Loading uses a two-pass design:
1. a lightweight **preview pass** (`session_preview`) that opens every file and reads only trials, stimulus presentations, running speed, eye tracking and metadata — enough to decide eligibility and to build the global image vocabulary and global running/pupil percentile edges;
2. a **conversion pass** (`convert_session`) that re-opens each eligible file and reads the large neural event matrix and builds the per-trial arrays.

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
        ...
```

```python
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
eligible, excluded = collect_previews(files=files, bin_size_sec=BIN_SIZE_SEC,
                                      sample_mode=args.sample, required_eligible=2)
```

iii. From CONVERSION_NOTES.md Step 6 and Step 10: *"Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment"* and *"Same underlying NWB tables/fields are used; I read them directly with `h5py` because `pynwb` failed in this environment."* The AI mapped each SDK object it had identified in Step 1 (`Trials`, `Presentations`, `RunningSpeed`, `EyeTrackingTable`, `CellSpecimens`, `Events`, `DFFTraces`) onto the corresponding NWB path, and re-verified the mapping in Step 10 by reconstructing five trials directly from raw NWB with `np.allclose()`. The two-pass design is justified as *"intentional to avoid storing large neural matrices before global percentile/bin definitions are known."* The AI recorded that the local subset spans two project codes (`VisualBehavior`, `VisualBehaviorMultiscope`; trajectory step 48) but gave no rationale for including both.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `/general/subject/subject_id` strings read out of each NWB file. A subject is registered the first time a file belonging to it is converted, and `subject_idx` indexes into that registration order. This yields 38 mice (37 `VisualBehavior` mice plus mouse `457841` from the Multiscope subset).

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
```python
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"`/general/subject/subject_id` or metadata `mouse_id` → `subjects`, `subject_idx`; String subject identifiers with session-level index mapping; Session order follows converted session order."* The subject id in the NWB `/general/subject` group is the same `mouse_id` used by the SDK experiment table, so this is the dataset's canonical animal identifier.

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one session.** The AI never groups experiments by `ophys_session_id`. It reads `ophys_session_id` in `session_preview` and stores it on `SessionPreview`, but the field is never used anywhere downstream. Sessions are ordered by sorted filename (i.e. ascending experiment id), not by acquisition date.

For the 239 single-plane `VisualBehavior` experiments this is harmless, because each of those sessions contains exactly one imaging plane. For the 45 `VisualBehaviorMultiscope` experiments it is not: those 45 planes come from only **8** real ophys sessions of a single mouse, so each real session is emitted 5–7 times as separate "sessions", each carrying a different subset of the simultaneously recorded neurons but an identical copy of the same behavioral trials. The verification log shows the symptom directly — `Subject 457841: 45 sessions`, and repeated blocks of identical trial counts (`209, 209, 209, 209, 209, 209, 209, ... 309 × 7, ... 406 × 7, ...`). Net effect: 281 emitted sessions and 84,313 trials instead of 236 sessions and 71,242 trials (the extra 13,071 trials are duplicated behavior).

ii. Read but unused:
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
...
return SessionPreview(
    path=path,
    experiment_id=experiment_id,
    ophys_session_id=ophys_session_id,
    ...
)
```

Sessions are simply the list of eligible files:
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```

iii. No justification is given anywhere in CONVERSION_NOTES.md or the trajectory. The AI had the information required to make the distinction: trajectory step 48 printed `local_ophys_sessions 247`, `experiments_per_session_mean 1.1497...` and `project_codes ['VisualBehavior', 'VisualBehaviorMultiscope']`, and CONVERSION_NOTES.md Step 2 records *"Unique local ophys sessions: `247`"* and *"Experiments per local ophys session: range `1-7`, mean `1.15`"*. Step 9 nevertheless reports the session count as *"`281` eligible local experiment files"*, i.e. the AI silently redefined "session" as "experiment file" without discussing the multi-plane case.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `/intervals/trials` table (the SDK `Trials` table serialized into the file). Every non-aborted, non-auto-rewarded row becomes one trial, spanning the full `start_time` → `stop_time` window (variable length, ~7–13 s, mean ~8.5 s). Within a trial the time axis is a uniform 100 ms grid anchored at `start_time`; the final bin is truncated at `stop_time`. `change_time`, `go` and `catch` are parsed into the `TrialSpec` but `change_time`/`is_go`/`is_catch` are used only for diagnostic plotting, not for building the outputs.

ii.
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    names = ["id", "start_time", "stop_time", "go", "catch", "aborted",
             "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
             "change_time", "change_frame", "initial_image_name", "change_image_name"]
    return read_interval_group(group, names)
```
```python
for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
    outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
    if sum(int(x) for x in outcome_flags) != 1:
        raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
    outcome_idx = outcome_flags.index(True)
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

iii. CONVERSION_NOTES.md Step 5, Key Decision 2: *"**Use SDK-valid trials only**: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed and consistent with the trial table semantics."* Key Decision 7: *"**Alignment event = trial start**: The trial itself is the natural unit requested by the user."* Step 4 records that trial logic was cross-checked across SDK code, NWB fields and the methods text and found consistent: *"Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly."*

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all at the trial or session level:
- **Trial level**: `aborted == True` or `auto_rewarded == True` are dropped (the only trial-level exclusions). A hard `ValueError` is raised if a surviving trial does not have exactly one of `hit/miss/false_alarm/correct_reject` set — i.e. malformed trials abort the run rather than being skipped. Unlike the reference, there is no explicit `change_time.notna()` requirement, and no clipping/skipping of trials that would run past the end of the ophys recording.
- **Session level**: a session is excluded if (a) eye tracking is absent, (b) it has fewer than 2 valid trials, or (c) it has fewer than 2 finite pupil samples inside trial windows.
- No filtering by session type: passive sessions (`OPHYS_2/5_..._passive`) are deliberately retained.

In the full run exactly 3 sessions were excluded, all for `missing_eye_tracking`; 0 for the other two reasons.

ii.
```python
if aborted[idx] or auto_rewarded[idx]:
    continue
```
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

iii. CONVERSION_NOTES.md Step 4: *"Exclude `aborted` and `auto_rewarded` trials from converted trial set, as required by the user task"*; Step 3 curation notes record that aborted trials are those with premature licking before the change. Key Decision 4: *"**Exclude sessions with missing eye tracking**: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values."* Key Decision 3: *"**Keep passive sessions if they have valid GO/CATCH trials**: Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs."* Step 10 Check 5 confirms *"verified no experiments had fewer than two valid go/catch trials after filtering."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the **event-detection output**, not dF/F: `/processing/ophys/event_detection/data` (shape `(n_timepoints, n_rois)`, the L0-regularized discrete calcium event magnitudes). The ROI→cell mapping comes from `/processing/ophys/event_detection/rois` and is used to index `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`. Timestamps are taken from `/processing/ophys/dff/traces/timestamps`, which is the same ophys frame clock as the event series (both length 140,204 in the file I checked). dF/F traces are read only for their timestamps, never for signal.

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

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: *"**Neural signal = event-detection output, not dF/F**: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events. This is the closest match to the paper while still using the SDK-defined loading and ROI filtering."* Step 3 records the supporting reading: *"the paper's main figures operate on discrete calcium events regressed from raw fluorescence traces, not directly on dF/F."* Step 4's discrepancy table resolves the dF/F-vs-events tension the same way. The mapping table adds: *"Use raw events, not visualization-only `filtered_events`"* (the SDK's `filtered_events` applies a causal half-Gaussian described as for visualization).

## 2-b. How is the `neural` data processed?

i. Processing is: (1) restrict columns to valid ROIs; (2) for each trial, for each 100 ms bin, **sum** the event magnitudes of all ophys frames whose timestamp falls in `[bin_start, bin_end)`; (3) transpose to `(n_neurons, n_timepoints)` and cast to `float32`. No normalization, smoothing, z-scoring, baseline subtraction or neuropil correction is added (the SDK pipeline has already done motion correction, demixing and neuropil correction upstream of event detection). Bins containing no ophys frame stay exactly zero.

Because sessions are not grouped by `ophys_session_id`, neurons from simultaneously imaged planes of a Multiscope session are **not** merged into a single neuron population — each plane keeps its own neuron set in its own emitted session.

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
        lo = int(frame_starts[b])
        hi = int(frame_ends[b])
        if hi > lo:
            neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions."* Step 10 Check 3: *"Per-bin event sums and percentile discretization for running/pupil ... Difference is intentional and required by the target format, not a mismatch in source processing."* Metadata records the rule explicitly: `"binning_rule": "sum event magnitudes within each 100 ms trial bin"`. Step 10 Check 2 verified a reconstructed trial against raw NWB with `np.allclose()` on experiment 877018118.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level quality control is the SDK's `valid_roi` flag: ROIs with `valid_roi == False` in `cell_specimen_table` are dropped before any binning. No activity-based, SNR-based or variance-based neuron filtering is applied, and no neurons are dropped for being silent (4.68% of trials end up entirely zero as a result, which the AI investigated and kept). Session-level QC described in the whitepaper (z-drift, motion) is treated as already applied by the Allen release.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. CONVERSION_NOTES.md Step 1 identified the SDK rule: *"`CellSpecimens.__init__` with `exclude_invalid_rois` ... Filters the cell table to `valid_roi == True`, then filters/reorders all traces and events to the remaining ROIs."* Step 4's resolution: *"In local NWBs all ROI rows inspected so far are marked valid (`valid_roi` sum equals ROI count), but the field exists explicitly ... Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid."* Step 10: *"`all neural data is zero` warnings remained after the rerun: investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. The 100 ms bin grid is generated by `np.arange(start_time, stop_time, 0.1)`, so bin 0 begins exactly at the trials-table `start_time`; the last bin is truncated at `stop_time`. Ophys frames are assigned to bins by `np.searchsorted` on the ophys timestamps (half-open `[bin_start, bin_end)`), so every frame in the trial window contributes to exactly one bin. Metadata declares `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (trials are variable length). All other streams are placed on the same grid, so neural and outputs are aligned by construction.

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

iii. CONVERSION_NOTES.md Key Decision 7: *"**Alignment event = trial start**: The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`)."* Step 4 established that all streams share a hardware sync clock: *"all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz ... imaging, visual stimulation, behavior, and eye-tracking streams are therefore aligned by shared synchronization signals."* Step 12: *"Temporal alignment was rechecked using the processing plots generated during sample conversion; stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms** (`time_bin_size = 100.0`), and yes — the data are explicitly rebinned. Native ophys rates are ~31 Hz (32.31 ms/frame) for the single-plane `VisualBehavior` experiments and ~11 Hz (~93 ms/frame) for the Multiscope planes. The AI chose a single 100 ms grid so that one bin size applies to all sessions, downsampling the 31 Hz majority by ~3× (summing ~3 frames/bin) and leaving the 11 Hz data at ~1 frame/bin (with occasional empty bins). Resulting trial lengths: T mean 85.8, min 71, max 127 bins (≈8.6 s mean), which corresponds to the same physical window as the reference (264 frames × 32.31 ms).

ii.
```python
BIN_SIZE_SEC = 0.1
```
```python
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
ends = np.minimum(starts + bin_size_sec, stop_time)
```
```python
"time_bin_size": BIN_SIZE_SEC * 1000.0,
```

iii. CONVERSION_NOTES.md Key Decisions 5 and 6: *"**Common time base via uniform rebinned trial bins**: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions."* and *"**Tentative common bin size = 100 ms**: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."* Step 10 Check 3 frames the rebinning as *"the only added step is the decoder-required common 100 ms rebinning ... intentional and required by the target format, not a mismatch in source processing."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations table**, not from the trials table. The AI merges every `/intervals/*_presentations` group that carries an `image_name` column (in practice only `Natural_Images_Lum_Matched_set_training_2017_presentations`; the `natural_movie_one` and `spontaneous` tables have no `image_name` and are correctly skipped), and uses the columns `image_name`, `start_time`, `stop_time`, `omitted`, `is_change`. The trials-table fields `initial_image_name` / `change_image_name` are read but never used to build the output.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    rows = []
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
    for name in list(out.keys()):
        out[name] = out[name][order]
    return out
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Stimulus presentation `image_name` + presentation timing + omission state → `output[0]` (`image_identity`); Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods."* The presentations table is the stimulus log itself, so it gives the exact on-screen interval of every flash rather than requiring reconstruction from `change_time`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global image vocabulary is built by pooling the unique non-empty `image_name` values across all eligible sessions, sorted, with an explicit **`gray` class prepended at index 0**. Per trial, the series is initialised to `gray` everywhere; then for each presentation overlapping the trial, bins whose **center** falls inside `[presentation_start, presentation_stop)` are set to that image's code. Omitted flashes are skipped, so they remain `gray`. The result is 17 classes (`gray` + 16 images) in which **`gray` occupies 67.0% of all bins** — the 500 ms inter-stimulus gray screen of each 750 ms flash cycle — while each image gets ~2% of bins. The reference instead holds the image identity constant across the whole trial (16 classes, ~6.25% each, no gray class).

An earlier bug here was found and fixed in Step 10: the literal string `"omitted"` was leaking into `output_values[0]` as an unused 18th class.

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
def make_image_series(presentations, row_idx, centers, image_to_idx):
    image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
    change_series = np.zeros(centers.shape, dtype=np.int16)
    for idx in row_idx:
        start = float(presentations["start_time"][idx])
        stop = float(presentations["stop_time"][idx])
        if stop <= start:
            continue
        in_window = (centers >= start) & (centers < stop)
        if not np.any(in_window):
            continue
        omitted = bool(np.nan_to_num(presentations["omitted"][idx], nan=0.0))
        if not omitted:
            image_name = str(presentations["image_name"][idx])
            image_series[in_window] = image_to_idx.get(image_name, image_to_idx["gray"])
        ...
```

iii. CONVERSION_NOTES.md Key Decision 8: *"**Image identity will include a `gray` class**: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin."* Step 10 Issue log: *"`output_values[0]` mistakenly contained `omitted` as an unused image class: fixed by excluding `omitted` from image vocabulary construction and re-running affected sample/full artifacts"* and *"The corrected `image_identity` vocabulary is now `17` classes total (`gray + 16` task images), consistent with the paper/whitepaper task description of two eight-image sets."*

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on the same per-trial 100 ms `centers` grid used for the neural bins, so index *t* of the image series and index *t* of the neural matrix refer to the same 100 ms window. A presentation is mapped onto a bin when the bin **center** lies inside the flash interval; presentations are pre-selected by interval overlap with the trial window.

ii.
```python
def find_presentation_rows(presentations, start_time, stop_time):
    starts = presentations["start_time"]
    ends = presentations["stop_time"]
    mask = (starts < stop_time) & (ends > start_time)
    return np.flatnonzero(mask)
```
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                          running_bins, pupil_bins, outcome_series])
```

iii. CONVERSION_NOTES.md Step 10 Check 3: *"`build_bin_centers()`, event sums on ophys timestamps, running/pupil interpolation to ophys-aligned bin centers ... Alignment matches the reference time bases."* The AI's planned sanity check — *"Raw-vs-converted image identity spot check: for chosen trials, verify bins overlapping raw stimulus-presentation intervals carry the correct image label and gray periods remain gray"* — was executed in Step 10 (`cache/step10_raw_checks.py`) and reported passing with `np.allclose()` on five trials.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the presentations table's `is_change` flag together with `omitted` and the presentation `start_time`/`stop_time`. It is *not* derived from the trials table's `change_time` or `go` columns (those are read but only used in plots). In the Allen data `is_change` is True only for genuine image changes — catch/sham trials have `is_sham_change` True and `is_change` False — so catch trials receive an all-zero change series, matching the reference's use of the `go` flag.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Stimulus presentation `is_change` + presentation timing → `output[1]` (`image_change`); Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0"*, with *"trial `change_time` for sanity checks"*. Step 10 Check 3 lists `is_change` among the SDK-provided fields whose semantics were matched.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Minimal: a zero-initialised `int16` series with 1s written into the bins whose center falls inside the changed flash's `[start_time, stop_time)` interval. A flash is 250 ms, so at 100 ms bins this marks 2–3 consecutive bins per change. Overall 2.63% of bins are labelled `change` (reference: 7.65%, because the reference marks a 750 ms window = flash + following gray). Omitted presentations can never set the flag.

ii. See the snippet in 4-a, in context:
```python
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
change_series = np.zeros(centers.shape, dtype=np.int16)
for idx in row_idx:
    ...
    in_window = (centers >= start) & (centers < stop)
    ...
    is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
    if is_change and not omitted:
        change_series[in_window] = 1
```

iii. CONVERSION_NOTES.md Key Decision 9: *"**Image-change target will mark the changed-image presentation, not only a single instant**: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is needed — the variable is natively binary. The only "threshold" is the temporal extent of the positive class: the 250 ms changed-flash interval (bins with centers inside it), giving `output_values[1] = ["no_change", "change"]` with codes 0/1 and a 97.4/2.6 split.

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

iii. Same as 4-b — Key Decision 9 justifies using the flash interval rather than a single instant as the positive window. CONVERSION_NOTES.md Step 12 notes the consequence: *"`image_change` positive bins are sparse (`2.63%`), so balanced accuracy modestly above `0.5` is still meaningful."*

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — same `centers` grid, same trial window, produced by the same `make_image_series` call, so row 1 of the output matrix is bin-for-bin aligned with the neural matrix.

ii.
```python
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                          running_bins, pupil_bins, outcome_series])
```

iii. CONVERSION_NOTES.md planned sanity check: *"Raw-vs-converted image change spot check: verify the converted binary change series turns on only for the changed-image flash and matches trial `change_time` / presentation `is_change`"*, reported as passing in Step 10. The `--show-processing` plot overlays the presentation intervals, a red `axvline` at each `is_change` flash onset, and the binned `change_series` to visually confirm no lag.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `/processing/running/speed/data` with `/processing/running/speed/timestamps` — the SDK's processed (filtered) running speed, i.e. the same object as `dataset.running_speed`. The unfiltered variant `/processing/running/speed_unfiltered` and the raw `dx` are noted in Step 2 but not used.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. CONVERSION_NOTES.md Step 5 mapping table names `RunningSpeed.from_nwb` as the corresponding reference function. Step 3: *"computed from wheel encoder voltage, with unwrapping, wrap detection, and derivative-based conversion to linear speed; whitepaper explicitly points to AllenSDK running-processing code for the implementation"* — i.e. the AI deliberately takes the SDK's already-processed speed rather than recomputing it.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the native ~60 Hz running clock onto the trial's 100 ms bin centers (`np.interp`, with edge **clamping** rather than NaN extrapolation), then discretisation into 5 globally-defined quantile bins. The quantile edges are computed once, in the preview pass, from the pool of **raw-rate** running samples that fall inside the included trial windows across all eligible sessions (not from the interpolated binned values). Non-finite interpolation results raise an error rather than being silently accepted.

ii.
```python
def interpolate_series(times, values, query_times):
    finite = np.isfinite(times) & np.isfinite(values)
    ...
    t = np.asarray(times[finite], dtype=np.float64)
    v = np.asarray(values[finite], dtype=np.float64)
    order = np.argsort(t); t = t[order]; v = v[order]
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
```
```python
def collect_time_window_values(times, values, trial_specs):
    segments = []
    for spec in trial_specs:
        lo = np.searchsorted(times, spec.start_time, side="left")
        hi = np.searchsorted(times, spec.stop_time, side="left")
        if hi > lo:
            segments.append(values[lo:hi])
    return np.concatenate(segments).astype(np.float32, copy=False)
```
```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
...
running_interp = interpolate_series(running_times, running_values, centers)
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins ... Use global bin edges computed over all included samples in included sessions/trials."* Key Decision 10: *"**Running and pupil bin edges will be global, not per-session**: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`."* Planned sanity check: *"Raw-vs-converted running spot check: compare rebinned running-speed values against raw `running/speed` sampled at the same absolute times"* (passed in Step 10). Achieved distribution `[0.199, 0.201, 0.199, 0.200, 0.200]` matched the *"Output-balance sanity check"*.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count bins. `compute_bin_edges` takes the 20/40/60/80th percentiles of the global pool (`np.quantile` with `np.linspace(0,1,6)[1:-1]`), and `discretize_with_edges` assigns bins with `np.digitize(..., right=False)` clipped to `[0, 4]`. Edges from the full run: running `[-0.0229, 0.1175, 10.039, 31.567]` cm/s. Value names are `q1..q5`.

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
RUN_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
```

iii. Directly follows the Decoder Task instruction *"Running speed, discretized into five equal percentile bins"*; CONVERSION_NOTES.md Key Decision 10 explains the choice of global (rather than per-session) edges. Step 9's consistency table reports the realised fractions `[0.1994, 0.2013, 0.1992, 0.1999, 0.2002]` as matching the *"approximately quintiles by construction"* expectation.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running trace is interpolated directly at the trial's 100 ms bin centers, which are the same bin centers that define the neural bins, so row 2 of the output matrix is aligned with the neural matrix by construction. Because both are anchored to `spec.start_time` on the shared hardware-synced clock, no cross-stream resampling offset is introduced.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")  # neural
...
running_interp = interpolate_series(running_times, running_values, centers)  # running, same grid
```

iii. CONVERSION_NOTES.md Step 4: *"all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz ... Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases."* The `--show-processing` plot (panel 3) overlays the raw running trace, the interpolated values at bin centers, and the resulting bin index on a common time axis to make any lag visible; Step 12 reports *"no visible lag mismatch."*

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `/acquisition/EyeTracking/pupil_tracking/area_raw` (the un-blink-masked pupil ellipse area), with `/acquisition/EyeTracking/eye_tracking/timestamps` as the clock and `/acquisition/EyeTracking/likely_blink/data` as the blink mask. Blink frames are set to NaN and then excluded, and the area is converted to an equivalent circular diameter `d = 2·sqrt(area/π)`. The presence of `pupil_tracking/area_raw` is also used as the test for whether a session has eye tracking at all. (The reference instead uses the `pupil_width` column; since the discretisation is by percentile and the area→diameter map is monotone, the two are effectively interchangeable for this output.)

ii.
```python
has_eye_tracking = "/acquisition/EyeTracking/pupil_tracking/area_raw" in f
...
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=np.float64)
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins."* Step 1 identified the SDK behaviour: *"`EyeTrackingTable.from_nwb` ... Loads pupil/eye ellipse data, recomputes likely blinks, and filters blink frames."* Step 3 records the whitepaper convention: *"whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter"* — i.e. inverting area back to a diameter recovers the whitepaper's quantity.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames → NaN → dropped by the `finite` mask inside `interpolate_series`; the surviving samples are linearly interpolated (with edge clamping) onto the trial's 100 ms bin centers; the result is discretised into 5 global quantile bins whose edges are computed in the preview pass from the pooled raw-rate in-trial pupil samples. A non-finite interpolated value raises an error. Sessions with fewer than 2 finite in-trial pupil samples are excluded entirely rather than imputed.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
```
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Same rationale as running speed (Key Decision 10, global edges), plus Key Decision 4 for excluding eye-tracking-free sessions *"rather than fabricating pupil values."* CONVERSION_NOTES.md Step 9 flags that the realised distribution is only approximately uniform — `[0.2029, 0.2058, 0.1929, 0.1866, 0.2119]` — a consequence of computing edges on the native-rate pool but applying them to the 10 Hz interpolated series.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same machinery as running speed: 20/40/60/80th percentiles of the global pooled pupil-diameter samples, then `np.digitize` with clipping to `[0, 4]`; value names `q1..q5`. Full-run edges: `[72.63, 83.38, 93.05, 104.90]` (pixel-derived diameter units).

ii.
```python
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```
```python
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
...
"pupil_bin_edges": pupil_edges.tolist(),
```

iii. Directly follows the Decoder Task instruction *"Pupil diameter, discretized into five equal percentile bins"*; global edges per Key Decision 10. The edges are stored in metadata so the mapping is recoverable.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated at the same trial bin centers as the neural bins and the other outputs, so row 3 of the output matrix is bin-for-bin aligned with the neural matrix. Blink gaps are bridged by the interpolation over the surviving samples rather than being shifted or dropped, which preserves the time base.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
...
output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                          running_bins, pupil_bins, outcome_series])
```

iii. Same synchronisation argument as running speed (Step 4: all streams share the 100 kHz sync board). The AI's planned check — *"Raw-vs-converted pupil spot check: compare converted pupil-diameter bins to raw eye-tracking-derived diameter after interpolation at selected timestamps"* — is reported as passing in Step 10, and panel 4 of the `--show-processing` figure overlays raw pupil, interpolated pupil and bin index on the trial time axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually-exclusive boolean columns of `/intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, read in that fixed order. NaNs in these columns are coerced to False before the boolean cast. The AI asserts that exactly one flag is set on every surviving trial and raises otherwise (rather than falling back to an "other" class as the reference does).

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

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Trial outcome flags `hit`, `miss`, `false_alarm`, `correct_reject` → `output[4]` (`trial_outcome`) ... Aborted and auto-rewarded trials excluded before conversion."* Step 1 notes the SDK guarantee the assertion relies on: *"Auto-rewarded trials are explicitly prevented from being counted as hit/miss/false_alarm/correct_reject."* Step 3: *"Trial structure consists of GO and CATCH trial types, which combine with behavior to yield HIT, MISS, FALSE ALARM, and CORRECT REJECTION outcomes."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to a fixed integer code 0–3 in the order `hit, miss, false_alarm, correct_reject` (the index of the True flag), and then **broadcast across every time bin of the trial** so that all five output rows are time-varying and share the same shape. Realised distribution over bins: hit 0.182, miss 0.692, false_alarm 0.010, correct_reject 0.115 — essentially identical to the reference's 0.187 / 0.687 / 0.011 / 0.115.

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                          running_bins, pupil_bins, outcome_series])
```
```python
"output_values": [..., TRIAL_OUTCOME_VALUES],
```

iii. CONVERSION_NOTES.md Key Decision 11: *"**Trial outcome will be repeated across time bins**: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form."* The AI's planned check — *"Outcome sanity check: per session, converted trial-outcome counts must match raw `hit`/`miss`/`false_alarm`/`correct_reject` counts after filtering"* — is reported as passing in Step 10.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is explicit and mostly fail-loud rather than fail-silent:
- **Missing eye tracking** (3 files): whole session excluded up front, rather than imputing pupil values.
- **Too few valid pupil samples / fewer than 2 valid trials**: whole session excluded.
- **Blinks**: `likely_blink` frames set to NaN and then dropped from the interpolation support, so blink artefacts do not propagate.
- **NaNs in trial boolean columns**: coerced to False via `np.nan_to_num` before the boolean cast; the same coercion is applied to `omitted`/`is_change` in the presentations table.
- **Extrapolation beyond a behavioural stream's coverage**: `np.interp` is called with `left=v[0], right=v[-1]`, so edges are clamped and no NaNs reach the discretiser (in contrast to the reference, which produces NaNs and maps them to bin 0 — i.e. silently into the *lowest* speed/pupil class). A post-hoc `np.any(~np.isfinite(...))` guard raises if anything non-finite survives.
- **Degenerate streams**: `interpolate_series` returns all-NaN if no finite samples exist and a constant if exactly one sample exists; `build_bin_centers` emits at least one bin if a trial is shorter than one bin width; zero-width presentations are skipped.
- **Malformed trials**: a trial without exactly one outcome flag raises `ValueError`, aborting the run. There is no per-session `try/except`, so one bad file would kill the whole conversion (the reference wraps each session and skips on failure).
- **Not handled**: trials are never clipped to the end of the ophys recording. Bins past the last ophys timestamp would yield `hi == lo` and remain silently zero rather than being dropped. I checked 13 sessions and found no trial whose `stop_time` exceeded the last ophys timestamp, so in this dataset the omission is latent rather than active.
- **All-zero neural trials** (3,947 = 4.68%): investigated, confirmed genuine, and kept. I independently verified one of them (experiment 1007107386, converted trial 108 = raw trial 145): 225 ophys frames in the window with exactly 0 non-zero event samples. The event matrix is 0.25% non-zero overall, so with 13 neurons all-zero trials are expected.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
```
```python
finite = np.isfinite(times) & np.isfinite(values)
if finite.sum() == 0:
    return np.full(query_times.shape, np.nan, dtype=np.float32)
if finite.sum() == 1:
    v = float(values[finite][0])
    return np.full(query_times.shape, v, dtype=np.float32)
...
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
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

iii. CONVERSION_NOTES.md Key Decision 4: sessions without eye tracking are dropped *"rather than fabricating pupil values."* Step 10 Check 5: *"Checked missing-eye-tracking handling (`3` excluded sessions), verified no experiments had fewer than two valid go/catch trials after filtering, validated a minimum-length trial with raw reconstruction, and confirmed that all-zero neural warnings arise from genuine zero-event trials in the raw event-detection matrices rather than off-by-one errors at trial boundaries."* Step 10 Issue log: *"`all neural data is zero` warnings remained after the rerun: investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero ... Removing them or replacing events with dF/F would diverge from the reference processing and task definition."*

## 9-a. What are the most time-consuming steps of the code?

i. Measured on the full run (`conversion_full_out.txt`): preview pass 65.0 s, conversion pass 686.0 s, total 757.7 s (~12.6 min), just inside the instructions' 15-minute guidance. Within the conversion pass the two costs are (1) reading the full-session event matrix out of HDF5 (`np.asarray(f["/processing/ophys/event_detection/data"])` materialises up to ~150,000 × 666 float32 ≈ 400 MB for the largest sessions — per-session times scale with neuron count: 1–2 s for 10-neuron sessions, 6 s for the 591-neuron session), and (2) the pure-Python per-bin rebinning loop, which executes roughly 84,313 trials × ~86 bins ≈ 7.2 M iterations, each doing a small NumPy slice-sum. Reading the trials/presentations/running/pupil tables happens **twice** per file (once in each pass), contributing the 65 s preview.

ii. The two hot spots:
```python
event_data = np.asarray(f["/processing/ophys/event_detection/data"], dtype=np.float32)
```
```python
for b in range(T):
    lo = int(frame_starts[b])
    hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```
Timing instrumentation is built in:
```python
log(f"Preview pass completed in {time.perf_counter() - preview_start:.1f}s")
...
log(f"Conversion pass completed in {time.perf_counter() - convert_start:.1f}s")
...
stats = {..., "elapsed_sec": int(round(time.perf_counter() - session_start))}
```

iii. CONVERSION_NOTES.md Step 6 identifies both: *"Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known"* and *"Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."* Speedups actually implemented: *"Preview pass avoids loading neural event matrices; Conversion uses direct dataset reads and processes one session at a time to cap peak memory; Processing plots are limited to at most 2 sessions; Sample-mode preview now short-circuits once 2 eligible sessions are found."*

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
1. **The per-bin neural rebinning loop** (`for b in range(T)`) — the big one, ~7.2 M Python iterations. It is exactly `np.add.reduceat` / a `cumsum`-difference over the frame axis: `cs = np.cumsum(event_data, axis=0); neural = (cs[frame_ends] - cs[frame_starts]).T`, a single vectorised expression per trial (or even per session).
2. **The per-presentation loop in `make_image_series`** — builds an `(n_bins,)` boolean mask per presentation (~11 presentations/trial × 84,313 trials). Since presentations are sorted and non-overlapping, one `np.searchsorted` of `centers` into the presentation boundaries would assign every bin in one shot.
3. **The per-trial loop in `build_trial_specs`** — a Python loop over all ~172,000 raw trial rows doing scalar indexing and building dataclasses, when the whole filter is already computed as boolean arrays; `np.argmax` over the stacked outcome flags would replace the inner `outcome_flags.index(True)`.

Also `collect_time_window_values` loops per trial doing two `searchsorted` calls each; these could be batched into one vectorised `np.searchsorted` over all trial boundaries.

ii.
```python
for b in range(T):                       # (1)
    lo = int(frame_starts[b]); hi = int(frame_ends[b])
    if hi > lo:
        neural_trial[:, b] = event_data[lo:hi].sum(axis=0, dtype=np.float32)
```
```python
for idx in row_idx:                      # (2)
    ...
    in_window = (centers >= start) & (centers < stop)
```
```python
for idx in range(raw_count):             # (3)
    if aborted[idx] or auto_rewarded[idx]:
        continue
    outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
```
```python
for spec in trial_specs:                 # (4)
    lo = np.searchsorted(times, spec.start_time, side="left")
    hi = np.searchsorted(times, spec.stop_time, side="left")
```

iii. The AI identified only (1), and explicitly chose not to fix it: *"Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."* Loops (2)–(4) are not mentioned anywhere. Since the full run finished in ~12.6 min, under the 15-minute threshold in Step 7/9, the AI had no forcing function to optimise further.

## 9-c. What processing does the code repeat multiple times?

i. The dominant repetition is the **two-pass file scan**: every eligible NWB file is opened twice, and `read_trials`, `build_trial_specs`, `read_task_presentations`, and the running-speed and pupil reads/blink-masking/area→diameter conversion are all performed once in `session_preview` and then again in `convert_session`. Nothing computed in the preview (`trial_specs` excepted) is carried forward — `preview.running_values` / `preview.pupil_values` are used only to build the global quantile pools, and the raw arrays are re-read from disk for the conversion. Smaller repetitions:
- `build_bin_centers`, `find_presentation_rows` and `make_image_series` are re-run from scratch inside `plot_processing_summary` for the plotted trial, duplicating work already done in the conversion loop.
- `str(presentations["image_name"][idx])` is re-decoded per presentation per trial instead of once per session.
- `read_trials` is called in `convert_session` solely to pass `trials` to the plotting function, even when plotting is off.

ii.
```python
# pass 1
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        trials = read_trials(f); specs = build_trial_specs(trials)
        presentations = read_task_presentations(f)
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        ...
        pupil_area[blink] = np.nan
        pupil_diameter = pupil_area_to_diameter(pupil_area)

# pass 2 — all of the above again
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        ...
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        pupil_area[likely_blink] = np.nan
        pupil_diameter = pupil_area_to_diameter(pupil_area)
        trials = read_trials(f)
        presentations = read_task_presentations(f)
```

iii. CONVERSION_NOTES.md Step 6 acknowledges and justifies it: *"Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."* The design constraint is real — global percentile edges and the global image vocabulary must be known before any trial's outputs can be finalised — but it is only the *neural* matrices that are large; the trials/presentations/running/pupil arrays it re-reads are small enough to have been cached on the `SessionPreview`. The reference solves the same ordering problem by holding the extracted per-trial slices in memory and discretising in a second in-memory pass, with no second disk read.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, all small relative to the two bottlenecks:
- **`pupil_area_to_diameter` is a no-op for the result.** `d = 2√(area/π)` is strictly monotone in `area`, and the only consumer is percentile binning, which is invariant under monotone transforms. The bin assignment would be bit-identical if raw area were binned directly. (The transform is still worth keeping for interpretability of the stored `pupil_bin_edges`.)
- **Unused trial columns are read and decoded** on every file, twice: `id`, `go`, `catch`, `change_time`, `change_frame`, `initial_image_name`, `change_image_name`. Of these, only `go`/`catch`/`change_time` are ever touched, and only inside plotting. `initial_image_name` / `change_image_name` are string columns requiring per-element `.decode()` and are never used at all.
- **`TrialSpec.change_time`, `.is_go`, `.is_catch`, `.trial_idx`** are computed for all ~84,000 trials but only read in `plot_processing_summary` (≤2 sessions).
- **`read_trials(f)` and `read_task_presentations`' extra columns** (`is_sham_change`, `trials_id`, `active`) are parsed but never used.
- **`load_neural_events` returns `event_rois[valid_mask]`**, discarded at the call site (`event_data, _ = load_neural_events(f)`).
- **`build_bin_centers` computes `widths`** only to derive `centers`; `starts`/`ends`/`centers` are all returned but `widths` is not needed by any caller.
- **`SessionPreview.running_values` / `.pupil_values`** retain the full concatenated in-trial behavioural samples for all 281 sessions simultaneously (hundreds of millions of float32 in the full run) purely to compute 8 quantile edges; reservoir sampling or per-session histograms would give the same edges at a fraction of the memory.
- **`sort_files_for_mode`** reads and groups the whole `ophys_cells_table.csv` in sample mode only; harmless but wasted in that path since the resulting order is then truncated to 2 sessions.

ii.
```python
event_data, _ = load_neural_events(f)          # second return value discarded
```
```python
names = ["id", "start_time", "stop_time", "go", "catch", "aborted", "auto_rewarded",
         "hit", "miss", "false_alarm", "correct_reject", "change_time", "change_frame",
         "initial_image_name", "change_image_name"]   # several never used
```
```python
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)  # monotone; irrelevant to percentile bins
```
```python
running_concat = collect_time_window_values(running_times, running_values, specs)
...
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
```

iii. The AI does not flag any of these. CONVERSION_NOTES.md Step 6's "Code inefficiencies identified" section lists only the double file scan and the non-cumsum rebinning loop. The plotting-only fields are a deliberate design (the `--show-processing` mode is a required CLI option), but they are computed unconditionally for every trial of every session rather than lazily for the ≤2 plotted sessions.
