# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the released NWB/HDF5 files directly with `h5py`. Data discovery is a filesystem glob over `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_*.nwb` (284 files locally). No project-level metadata table is consulted to scope the data (the `ophys_experiment_table.csv` / manifest are never read; `ophys_cells_table.csv` is read only in `--sample` mode to rank files by cell count). In particular **no `project_code` filter is applied**, so the 45 `VisualBehaviorMultiscope` plane-experiments present in the local subset are loaded alongside the 239 `VisualBehavior` single-plane experiments.

Loading is done in two passes over the same files:
1. a lightweight *preview* pass (`session_preview`) that reads metadata, the trials table, the stimulus-presentation tables, running speed, and eye tracking, in order to determine session eligibility, the global image vocabulary, and global running/pupil percentile edges;
2. a *conversion* pass (`convert_session`) that reopens each file and additionally reads the event-detection matrix and ophys timestamps, then builds the per-trial arrays.

Per-experiment fields read: `/identifier`, `/general/metadata` attrs (`ophys_session_id`, `session_type`), `/general/subject/subject_id`, `/general/optophysiology/imaging_plane_1/location`, `/intervals/trials`, `/intervals/*_presentations`, `/processing/ophys/dff/traces/timestamps`, `/processing/ophys/event_detection/*`, `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`, `/processing/running/speed/*`, `/acquisition/EyeTracking/*`.

Final scope: 284 files scanned → 281 converted, 38 subjects, 84,313 trials, 41,871 neurons.

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
    eligible, excluded = collect_previews(
        files=files, bin_size_sec=BIN_SIZE_SEC,
        sample_mode=args.sample, required_eligible=2)
```

iii. From CONVERSION_NOTES.md Step 6: *"Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment"*, and it claims the reads *"mirror SDK semantics already serialized into NWB"* (trials from `/intervals/trials`, stimulus timing from `/intervals/*_presentations`, running from `/processing/running/speed`, pupil from `/acquisition/EyeTracking/*`, neural from `/processing/ophys/event_detection`). Step 10's "Reference code comparison" table records: *"Same underlying NWB tables/fields are used; I read them directly with `h5py` because `pynwb` failed in this environment."* Step 2 notes explicitly record that the metadata CSVs describe the full release (703 sessions / 1936 experiments) while only 284 NWB files are available locally, and Step 4 resolves this as a *"release-version and local-download-scope difference"*, deciding to *"Use local files as authoritative for actual conversion scope."* No justification is given anywhere for including the `VisualBehaviorMultiscope` experiments — the project code is never mentioned as a scoping criterion.

## 1-b. How are the data split into subjects?

i. Subjects are the string `subject_id` read from `/general/subject/subject_id` of each NWB file. Subjects are registered into the `subjects` list in first-encountered order over the eligible files (files are globbed in sorted filename order), and `subject_idx` stores the index of the owning subject for every session. Result: 38 subjects (37 from the `VisualBehavior` project + 1 Multiscope mouse, `457841`).

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

iii. From the Step 5 variable-mapping table: *"`/general/subject/subject_id` or metadata `mouse_id` → `subjects`, `subject_idx`; String subject identifiers with session-level index mapping; Session order follows converted session order."* The AI treats the per-file subject metadata as the canonical mouse identifier, which is the same field the SDK exposes as `mouse_id`.

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one session.** The AI never groups experiments by `ophys_session_id`; `ophys_session_id` is parsed in `session_preview` but is stored only on the `SessionPreview` dataclass and is never used to merge or key anything. Every eligible file contributes exactly one entry to `neural`/`input`/`output`/`subject_idx`/`brain_region_idx`, so its neurons form an independent population and its trials are an independent trial list.

For the 239 single-plane `VisualBehavior` experiments this is harmless (1 experiment = 1 session). For the 45 `VisualBehaviorMultiscope` plane-experiments (mouse `457841`, 8 real ophys sessions, up to 7 simultaneously-recorded planes each) it splits each real session into up to 7 "sessions" that each contain a small subset of the neurons (4–40) but a **duplicated copy of the identical trial list and identical behavioral outputs**. This is directly visible in the conversion log (e.g. experiments `958527464/471/474/479/481/485/488` all report exactly `309 trials`, and `959388788…802` all report `406 trials`) and in the verification log (`Subject 457841: 45 sessions` while every other mouse has 4–11).

Session ordering is sorted filename (i.e. experiment-id) order, not acquisition date.

ii.
```python
        ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])
        ...
        return SessionPreview(
            path=path,
            experiment_id=experiment_id,
            ophys_session_id=ophys_session_id,   # stored, never used again
            ...
        )
```
```python
    for i, preview in enumerate(eligible, start=1):
        ...
        neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_sessions.append(output_trials)
```

iii. The AI never states a rationale for using the experiment file as the session unit. It was aware of the distinction: Step 1 notes *"One experiment corresponds to one imaging plane in one session. For multiplane sessions, `OphysTimestampsMultiplane.from_sync_file()` subsamples interleaved frame times by plane group before validation"*, and Step 2 records *"Unique local ophys sessions: `247`"* and *"Experiments per local ophys session: range `1-7`, mean `1.15`"*. Despite this, all downstream tables (Step 9 consistency check, README) equate "sessions" with *"`281` eligible local experiment files"*. The only related reasoning found in the trajectory (step 112) is about bin size, not session identity: *"rebin all sessions to one common trial-relative time base so the dataset format stays consistent across single-plane and multi-plane recordings."*

## 1-d. How are the data split into trials?

i. Trials come from the SDK trials table serialized at `/intervals/trials`. `build_trial_specs` iterates the raw trial rows, drops `aborted` and `auto_rewarded` rows, and keeps the rest (i.e. go and catch trials). Each kept trial is represented by a `TrialSpec` holding `start_time`, `stop_time`, `change_time`, the outcome index, and the `go`/`catch` flags. The trial window used downstream is the full `[start_time, stop_time)` interval, which is then tiled with fixed 100 ms bins — producing variable-length trials (71–127 bins, mean 85.8 bins ≈ 8.6 s). `change_time` is parsed but is never used to define the window or any output.

ii.
```python
def build_trial_specs(trials: Dict[str, np.ndarray]) -> List[TrialSpec]:
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

iii. Step 5 Key Decision 2: *"**Use SDK-valid trials only**: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed and consistent with the trial table semantics."* Step 1 records the SDK's trial construction (*"defined from the behavior `trial_log` using each trial's `trial_start` frame and the next trial's start frame as the current trial end boundary"*) and Step 4 concludes *"Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly."* Step 5 Key Decision 7 fixes the trial window: *"**Alignment event = trial start** … Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`)."*

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: only `aborted` and `auto_rewarded` are excluded; there is no `change_time` validity requirement, no clipping against the end of the ophys recording, and no removal of trials whose neural data end up empty. `build_trial_specs` **hard-fails** (raises) if a kept trial does not carry exactly one of `hit`/`miss`/`false_alarm`/`correct_reject`. `convert_session` also hard-fails if any interpolated running or pupil value in any trial is non-finite.

Session-level: a session is dropped if (a) `/acquisition/EyeTracking/pupil_tracking/area_raw` is absent, (b) it has fewer than 2 valid go/catch trials, or (c) it has fewer than 2 finite pupil samples inside its trial windows. In the full run only criterion (a) fired, excluding 3 of 284 files. Exclusion reasons are recorded in `metadata['excluded_sessions']`.

Not filtered: 3,947 trials (4.68%) whose entire neural matrix is zero were kept; the verification log emits one warning per such trial.

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
    if len(eligible) < 2:
        raise RuntimeError("Need at least 2 eligible sessions after filtering")
```

iii. Step 5 Key Decision 4: *"**Exclude sessions with missing eye tracking**: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values."* Key Decision 3: *"**Keep passive sessions if they have valid GO/CATCH trials**: Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs."* Step 4 resolution on aborted trials: *"Exclude `aborted` and `auto_rewarded` trials from converted trial set, as required by the user task."* On the all-zero trials, Step 10 records: *"investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the **ophys event-detection output**, `/processing/ophys/event_detection/data` (shape `(n_frames, n_rois)`), *not* from dF/F. dF/F is touched only for its timestamps (`/processing/ophys/dff/traces/timestamps`), which are the ophys frame times. Columns are restricted to ROIs whose `valid_roi` flag in `/processing/ophys/image_segmentation/cell_specimen_table` is True, resolved through the event table's `rois` index. Raw (unsmoothed) events are used, not the SDK's `filtered_events`.

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
        ophys_timestamps = np.asarray(
            f["/processing/ophys/dff/traces/timestamps"], dtype=np.float64
        )
        event_data, _ = load_neural_events(f)
```

iii. Step 5 Key Decision 1: *"**Neural signal = event-detection output, not dF/F**: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events. This is the closest match to the paper while still using the SDK-defined loading and ROI filtering."* Step 3 records the textual basis: *"the paper's main figures operate on discrete calcium events regressed from raw fluorescence traces, not directly on dF/F"*. Step 4's discrepancy table resolves dF/F-vs-events as: *"Use the available processed neural signal in a way that matches the paper/code path. Tentative resolution for later mapping: prefer event-detection outputs for neural activity because the analysis paper explicitly uses them, while preserving SDK ROI filtering and timestamps."* The Step 5 mapping table adds: *"Use raw events, not visualization-only `filtered_events`."*

## 2-b. How is the `neural` data processed?

i. The only processing is **temporal rebinning by summation**. For each trial, a 100 ms grid is laid over `[start_time, stop_time)`; for each bin the ophys frames whose timestamps fall in `[bin_start, bin_end)` are located with `np.searchsorted` and their event magnitudes are **summed** (not averaged, and not normalised by the number of contributing frames or by bin duration). Bins that contain no ophys frame are left at 0. No baseline correction, smoothing, z-scoring, or per-neuron normalisation is applied. Output dtype is `float32`, shape `(n_neurons, n_bins)`.

Because the bin width (100 ms) is not an integer multiple of the frame period, the number of frames contributing per bin is not constant: 3 or 4 frames for the 31 Hz single-plane experiments (~±14 % amplitude jitter), and 1 or 2 frames (occasionally 0) for the 11 Hz multi-plane experiments. The sums are not rescaled to compensate.

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

iii. Step 5 mapping table: *"Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions."* Key Decision 5: *"**Common time base via uniform rebinned trial bins**: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions. I will rebin all trials to a common bin width based on ophys-time alignment."* The metadata field records the rule verbatim: `"binning_rule": "sum event magnitudes within each 100 ms trial bin"`. Step 6 notes the implementation trade-off: *"Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."* No justification is given for summing rather than averaging, nor is the variable frames-per-bin issue discussed anywhere.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Exactly one neuron-level filter: ROIs with `valid_roi == False` in the NWB `cell_specimen_table` are dropped, reproducing the AllenSDK's default `exclude_invalid_rois=True` behaviour. No further filtering by event rate, SNR, activity, or trial-wise responsiveness is applied; neurons that emit no events in a trial (or in the whole session) are retained, which is why 4.68 % of trials contain an all-zero neural matrix. Total retained: 41,871 ROIs over 281 sessions.

ii.
```python
    valid_roi = np.asarray(
        f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
    )
    valid_mask = valid_roi[event_rois]
    event_data = event_data[:, valid_mask]
```

iii. Step 1 identified the SDK rule: *"`CellSpecimens.__init__` with `exclude_invalid_rois` … Filters the cell table to `valid_roi == True`, then filters/reorders all traces and events to the remaining ROIs."* Step 4's resolution: *"Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid."* Step 1 also notes *"I did not find electrophysiology unit-quality filtering in the relevant Visual Behavior ophys path because this task is 2-photon calcium imaging, not ephys"*, and *"dF/F traces are loaded directly from `DFFTraces`; no extra dF/F computation is needed."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. The bin grid for a trial begins exactly at the trials-table `start_time` and steps forward in 100 ms increments until `stop_time` (last bin truncated). Neural frames are selected against the ophys timestamp vector by `np.searchsorted(..., side="left")` on the bin start/end times, so a frame contributes to the bin whose half-open interval contains its timestamp. All other streams (image identity, image change, running, pupil) are evaluated on the *same* `centers` vector, so every output row is aligned bin-for-bin with the neural matrix by construction. `metadata` records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable length).

ii.
```python
            starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
            frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
            frame_ends = np.searchsorted(ophys_timestamps, ends, side="left")
```
```python
            row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
            image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
            running_interp = interpolate_series(running_times, running_values, centers)
            pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```
```python
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
```

iii. Step 5 Key Decision 7: *"**Alignment event = trial start**: The trial itself is the natural unit requested by the user."* Step 10's comparison table: *"`build_bin_centers()`, event sums on ophys timestamps, running/pupil interpolation to ophys-aligned bin centers … Alignment matches the reference time bases; the only added step is the decoder-required common 100 ms rebinning."* Step 3 records the physical justification for cross-stream alignment: *"all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz"*, hence *"Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases"* (Step 4).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms** (`BIN_SIZE_SEC = 0.1`, written to metadata as `time_bin_size = 100.0`). Yes — rebinning is applied to every stream. Neural data are rebinned from the native ophys rate (31 Hz ≈ 32.3 ms single-plane; 11 Hz ≈ 90.9 ms multi-plane) by summing events per bin; running speed and pupil are resampled by linear interpolation at bin *centres* (point-sampling, not bin-averaging); image identity / change are evaluated by asking which presentation interval contains each bin centre. Trials therefore have 71–127 bins (mean 85.8), versus the ~264 native frames the same windows would give at 31 Hz.

ii.
```python
BIN_SIZE_SEC = 0.1
```
```python
            "time_bin_size": BIN_SIZE_SEC * 1000.0,
            "neural_representation": "ophys event-detection magnitudes",
            "binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. Step 5 Key Decision 6: *"**Tentative common bin size = 100 ms**: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams. This choice will be validated during sample conversion."* Key Decision 5 gives the driver: the target format requires a single bin size across all sessions while native rates differ between single-plane and multi-plane recordings. Step 10's comparison table classifies this as an intentional deviation: *"Native SDK data remain at native sample rates; the papers do not prescribe a common decoder bin size. Difference is intentional and required by the target format, not a mismatch in source processing."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus-presentation table**, not from the trials table. `read_task_presentations` concatenates every `/intervals/*` group (excluding `trials`) that carries `image_name`, `start_time` and `stop_time`, and sorts by `start_time`. In these files that resolves to the single `*_presentations` table of task flashes (the `natural_movie_one_presentations` and `spontaneous_presentations` groups have no `image_name` and are skipped). The columns used for image identity are `start_time`, `stop_time`, `image_name` and `omitted`. The trials-table columns `initial_image_name` / `change_image_name` are read but never used to build this output.

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, np.ndarray]] = []
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

iii. Step 5 mapping table: *"Stimulus presentation `image_name` + presentation timing + omission state → `output[0]` (`image_identity`): Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods … Time-varying categorical output. Global category set will include all local image names plus `gray`."* Step 1 had identified `Presentations.from_stimulus_file` as the SDK function that *"Creates the per-stimulus presentation table with `image_name`, `start_time`, `is_change`, `omitted`, `flashes_since_change`, and `trials_id`."*

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A single **global** vocabulary is built by pooling the unique non-empty `image_name` values across all eligible sessions, sorting them, and prepending a synthetic `"gray"` class at index 0 (`image_values = ["gray"] + image_names` → 17 classes: gray + im000…im106). For each trial the series is initialised to `gray` everywhere; then for each presentation overlapping the trial, the bins whose centre falls inside `[flash_start, flash_stop)` are set to that flash's code. **Omitted flashes are deliberately left as `gray`**, and any unrecognised name falls back to `gray`. Since each flash is 250 ms and the cycle is 750 ms, roughly two thirds of all bins end up labelled `gray`: the released fractions are `gray 0.670` with each of the 16 real images at ≈0.020–0.022.

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
```

iii. Step 5 Key Decision 8: *"**Image identity will include a `gray` class**: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin."* Step 10 reports a bug fix in this area: *"`output_values[0]` incorrectly included an unused `omitted` image label even though omitted flashes were encoded as `gray` in the time series. Fixed by excluding `omitted` from `unique_nonempty_images()`"*, after which *"The corrected `image_identity` vocabulary is now `17` classes total (`gray + 16` task images), consistent with the paper/whitepaper task description of two eight-image sets."* Step 12 acknowledges the resulting decoding accuracy (0.2504 validation vs 0.0588 chance) but attributes no problem to the gray class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same `centers` array used to build the neural bins for that trial, so alignment is bin-for-bin by construction. Presentations are pre-selected by interval overlap with the trial window (`find_presentation_rows`), then assigned by testing bin centres against `[flash_start, flash_stop)`. The presentation timestamps and the ophys timestamps are both on the hardware-synchronised session clock, so no offset correction is applied.

ii.
```python
def find_presentation_rows(presentations, start_time, stop_time):
    starts = presentations["start_time"]
    ends = presentations["stop_time"]
    mask = (starts < stop_time) & (ends > start_time)
    return np.flatnonzero(mask)
```
```python
            row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
            image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
            ...
            output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                                      running_bins, pupil_bins, outcome_series])
```

iii. Step 10's temporal-alignment row: *"Alignment matches the reference time bases; the only added step is the decoder-required common 100 ms rebinning."* Step 12: *"Temporal alignment was rechecked using the processing plots generated during sample conversion; stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch."* The `--show-processing` plot overlays raw presentation spans (`axvspan`) on the rebinned `image_series` step plot for this purpose.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the stimulus-presentation table's `is_change` flag together with the flash `start_time`/`stop_time` and `omitted` columns. The trials-table `change_time` / `go` columns are parsed into `TrialSpec` but are **not** used for this output. I verified on a source file that `is_change` is True only for the genuine change flash of go trials (323/323 go trials flagged, 0/42 catch trials flagged; catch trials are separately marked by `is_sham_change`), so this source is functionally equivalent to "change_time on go trials only".

ii.
```python
    keep_names = ["start_time", "stop_time", "image_name", "omitted",
                  "is_change", "is_sham_change", "trials_id", "active"]
```
```python
        is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
        if is_change and not omitted:
            change_series[in_window] = 1
```

iii. Step 5 mapping table: *"Stimulus presentation `is_change` + presentation timing → `output[1]` (`image_change`): Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0 … trial `change_time` for sanity checks. Time-varying binary output."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector of length `n_bins` is created per trial; bins whose centre falls inside the changed-image flash interval (~250 ms → typically 2–3 bins at 100 ms) are set to 1. Omitted flashes can never set the flag. Nothing is marked for catch/sham trials, and the flag does not extend into the following grey inter-stimulus interval. Resulting global distribution: `no_change 0.974`, `change 0.026`.

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

iii. Step 5 Key Decision 9: *"**Image-change target will mark the changed-image presentation, not only a single instant**: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure."* Step 12 discusses the consequence: *"`image_change` positive bins are sparse (`2.63%`), so balanced accuracy modestly above `0.5` is still meaningful."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed or performed — the variable is natively binary. The two categories are taken directly from the boolean `is_change` flag and named `["no_change", "change"]` in `output_values[1]`; the stored dtype is `int16` with values in `{0, 1}` (verification reports range `[0.0, 1.0]`).

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

iii. The instruction specified image change as a binary variable, and the source field is already boolean; the AI's Step 5 mapping simply calls it a *"Time-varying binary output"*. No discretisation rationale was needed or given.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — it is produced by the same `make_image_series` call on the same `centers` grid as the neural bins, from presentations pre-filtered by overlap with the trial window. No lag or offset is applied.

ii.
```python
            row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
            image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. Same justification as 3-c: all streams are placed on the shared, hardware-synchronised session clock and then evaluated on one common per-trial bin grid; the `--show-processing` figure draws a red `axvline` at each raw `is_change` flash start over the rebinned `change_series` to make any misalignment visible (Step 12: *"no visible lag mismatch"*).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `/processing/running/speed/data` with its own `/processing/running/speed/timestamps` — the SDK's processed (filtered) running speed, i.e. the same object the SDK exposes as `dataset.running_speed`. The alternative `speed_unfiltered` and `dx` series present in the NWB are not used. Roughly 270k–288k samples per experiment (~60 Hz).

ii.
```python
        running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
        running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 5 mapping table: *"Running speed timeseries → `output[2]` (`running_speed_bin`) … `RunningSpeed.from_nwb`."* Step 1 records the SDK semantics: *"`RunningSpeed.from_stimulus_file` … Computes running speed aligned to stimulus timestamps with zero monitor delay; includes polarity correction if mean speed is implausibly negative"*, and Step 3 notes the whitepaper *"explicitly points to AllenSDK running-processing code for the implementation"* — i.e. the AI deliberately consumed the already-processed SDK output rather than recomputing from the encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps. (1) **Resampling**: `np.interp` linearly interpolates the running trace onto the trial's 100 ms bin *centres*; non-finite samples are dropped first, the series is sorted by time, and values outside the recorded range are held constant at the first/last value (`left=v[0], right=v[-1]`) rather than becoming NaN. Note this point-samples the ~60 Hz signal at one instant per 100 ms bin rather than averaging within the bin. (2) **Discretisation**: see 5-c. A hard `ValueError` is raised if any interpolated value is still non-finite.

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
    order = np.argsort(t)
    t = t[order]; v = v[order]
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
```
```python
            running_interp = interpolate_series(running_times, running_values, centers)
            if np.any(~np.isfinite(running_interp)):
                raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
            running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 5 mapping table: *"Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins."* Step 4's alignment resolution: *"Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases."* Step 1 notes the motivation — *"Running speed is built on stimulus-timebase timestamps with `monitor_delay=0.0`, not on ophys timestamps"* — so a resampling step is unavoidable.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into **five global quintile bins**. During the preview pass the raw running samples falling inside every eligible trial window of every eligible session are pooled (`collect_time_window_values`), non-finite values are dropped, and the 20/40/60/80th percentiles are taken as four inner edges; `np.digitize(..., right=False)` then maps each interpolated value to 0–4, clipped to that range. The same edges are applied to every session and are stored in `metadata['running_bin_edges']` = `[-0.0229, 0.1175, 10.039, 31.567]`. Achieved distribution: `[0.199, 0.201, 0.199, 0.200, 0.200]`. Note the edges are computed from the *raw, native-rate* samples, not from the interpolated 100 ms values that are actually discretised.

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
    running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
    running_pool = running_pool[np.isfinite(running_pool)]
    running_edges = compute_bin_edges(running_pool, 5)
```

iii. Step 5 Key Decision 10: *"**Running and pupil bin edges will be global, not per-session**: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`."* Planned sanity check: *"Output-balance sanity check: global running and pupil bins should each contain roughly 20% of included samples by construction"*, verified in Step 9 as `[0.1994, 0.2013, 0.1992, 0.1999, 0.2002]`. Step 7 also notes the preview was changed to *"pool raw running/pupil samples within valid trials instead of interpolating every trial during preview"* as a speed optimisation — the source of the raw-vs-interpolated edge discrepancy.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same per-trial `centers` vector that defines the neural bins, so row 2 of the output matrix is aligned bin-for-bin with the neural matrix. The running timestamps and the ophys timestamps share the synchronised session clock, so no offset is applied.

ii.
```python
            starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
            ...
            running_interp = interpolate_series(running_times, running_values, centers)
            ...
            output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                                      running_bins, pupil_bins, outcome_series])
```

iii. Step 3: *"all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz; imaging, visual stimulation, behavior, and eye-tracking streams are therefore aligned by shared synchronization signals."* Step 10: *"running/pupil interpolation to ophys-aligned bin centers … Alignment matches the reference time bases."* The `--show-processing` panel overlays the raw running trace, the rebinned trace and the discretised bins on one time axis as the visual check.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `/acquisition/EyeTracking/pupil_tracking/area_raw` with timestamps from `/acquisition/EyeTracking/eye_tracking/timestamps`, plus the blink mask `/acquisition/EyeTracking/likely_blink/data`. Blink frames are set to NaN *before* any further processing, and the area is converted to an equivalent diameter as `2·sqrt(area/π)`. I confirmed against the source files that `area_raw = π·max(width, height)²` where width/height are the ellipse semi-axes, so `2·sqrt(area_raw/π)` returns exactly the **ellipse major axis**, i.e. the quantity the whitepaper calls the pupil diameter. Presence of `area_raw` is also the eligibility test for keeping a session at all.

ii.
```python
        has_eye_tracking = "/acquisition/EyeTracking/pupil_tracking/area_raw" in f
        ...
            pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
            pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
            blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
            pupil_area = pupil_area.copy()
            pupil_area[blink] = np.nan
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

iii. Step 5 mapping table: *"Eye-tracking pupil signal → `output[3]` (`pupil_diameter_bin`): Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins … Requires eye-tracking availability. Sessions with missing eye tracking will be excluded."* The conversion formula follows Step 3's reading of the whitepaper: *"whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter"* — i.e. the AI inverted the whitepaper's own area definition to recover the diameter. Step 1 recorded the SDK blink handling: *"`EyeTrackingTable.from_nwb` … Loads pupil/eye ellipse data, recomputes likely blinks, and filters blink frames."*

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink frames → NaN; (2) area → diameter via `2·sqrt(area/π)`; (3) the same `interpolate_series` helper as running speed drops all non-finite samples and linearly interpolates the surviving diameter samples onto the trial's 100 ms bin centres — which means blink gaps (and the ~4.6 % NaN samples) are bridged by linear interpolation across them rather than left missing; constant extrapolation is used outside the recorded range; (4) discretisation into 5 global quintile bins. A hard `ValueError` is raised if any interpolated pupil value is non-finite.

ii.
```python
            pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
            if np.any(~np.isfinite(pupil_interp)):
                raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
            pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Same Step 5 mapping row as 6-a. The design intent is that blink artefacts must not enter the signal (blinks → NaN → excluded from the interpolant), and that pupil must share the neural time base (interpolation to bin centres). Step 4's alignment resolution again applies: *"Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases."*

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Into **five global quintile bins**, by exactly the same machinery as running speed: preview pools the blink-masked raw diameter samples inside all eligible trial windows across all eligible sessions, drops non-finite values, takes the 20/40/60/80th percentiles as inner edges, and `np.digitize`/`np.clip` maps values to 0–4. Edges are stored in `metadata['pupil_bin_edges']` = `[72.63, 83.38, 93.05, 104.90]`. Because the edges come from the raw pre-interpolation samples while the discretised values come from the blink-bridged interpolated series, the realised distribution is only approximately uniform: `[0.203, 0.206, 0.193, 0.187, 0.212]`.

ii.
```python
    pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
    pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
    if running_pool.size < 5 or pupil_pool.size < 5:
        raise RuntimeError("Not enough running or pupil samples to compute 5-bin discretization")
    pupil_edges = compute_bin_edges(pupil_pool, 5)
```
```python
PUPIL_BIN_VALUES = [f"q{i}" for i in range(1, 6)]
```

iii. Step 5 Key Decision 10 (global, not per-session edges) and the planned *"Output-balance sanity check: global running and pupil bins should each contain roughly 20% of included samples by construction."* Step 9 records the realised fractions and marks them a match, attributing the residual non-uniformity to the pupil path: *"Derived from global local pupil-diameter percentiles after blink masking."*

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed: interpolated onto the trial's `centers` grid, which is the grid the neural bins were built on, so row 3 of the output matrix is bin-for-bin aligned with the neural matrix. Eye-camera timestamps and ophys timestamps are both on the synchronised session clock; no offset correction is applied.

ii.
```python
            pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
            pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
            output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                                      running_bins, pupil_bins, outcome_series])
```

iii. Step 1: *"Eye tracking is aligned to synchronized eye-camera frame times; likely blinks are recomputed and blink frames are filtered."* Step 3/Step 4: all streams share the 100 kHz-sampled sync board, so interpolation onto the ophys-derived bin centres is valid. Step 12 confirms via the processing plots that *"pupil interpolation … [is] synchronized on the same ophys-aligned trial window with no visible lag mismatch."*

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of `/intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. NaN entries are coerced to False. The code requires exactly one of the four to be set for every retained (non-aborted, non-auto-rewarded) trial and raises otherwise, so there is no "other"/fallback category. Released distribution: `hit 0.182, miss 0.692, false_alarm 0.010, correct_reject 0.115`.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
```
```python
        outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
        if sum(int(x) for x in outcome_flags) != 1:
            raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
        outcome_idx = outcome_flags.index(True)
```

iii. Step 5 mapping table: *"Trial outcome flags `hit`, `miss`, `false_alarm`, `correct_reject` → `output[4]` (`trial_outcome`) … `Trials.from_nwb`, `Trial._get_trial_data`. Aborted and auto-rewarded trials excluded before conversion."* Step 1 documents why the four flags are exhaustive and exclusive for the retained set: *"`correct_reject = catch and not false_alarm`"* and *"Auto-rewarded trials are explicitly prevented from being counted as hit/miss/false_alarm/correct_reject."* Step 3's curation notes: *"Trial structure consists of GO and CATCH trial types, which combine with behavior to yield HIT, MISS, FALSE ALARM, and CORRECT REJECTION outcomes."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is encoded as the index into the fixed list `["hit", "miss", "false_alarm", "correct_reject"]` (0–3) and then **broadcast as a constant across every time bin of the trial**, so that the whole output block keeps the uniform `(n_output, n_timepoints)` shape. No other processing.

ii.
```python
            outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
            output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                                      running_bins, pupil_bins, outcome_series])
```
```python
        "output_values": [image_values, ["no_change", "change"],
                          RUN_BIN_VALUES, PUPIL_BIN_VALUES, TRIAL_OUTCOME_VALUES],
```

iii. Step 5 Key Decision 11: *"**Trial outcome will be repeated across time bins**: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form."* This matches the target-format guidance that outputs be time-varying where possible and that all output rows share one time axis.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's posture is mostly **fail-loud rather than impute**:
- **Missing eye tracking** (3 of 284 files): the whole session is dropped, with the reason recorded in `metadata['excluded_sessions']`. Same for `<2` valid trials or `<2` finite pupil samples (neither fired).
- **Blinks / NaN pupil samples**: set to NaN and excluded from the interpolant; the gap is then bridged by linear interpolation of the surrounding valid samples.
- **NaN flags in the trials table**: coerced to False via `np.nan_to_num` before boolean casting (`aborted`, `auto_rewarded`, `go`, `catch`, outcomes, `omitted`, `is_change`).
- **Out-of-range query times**: `np.interp` holds the first/last value constant instead of producing NaN; any residual non-finite value then raises a `ValueError` and aborts the run rather than being silently binned.
- **Degenerate time series**: `interpolate_series` returns a constant array if only one finite sample exists, and an empty array for an empty query.
- **Zero-width presentations** (`stop <= start`) and presentations that contain no bin centre are skipped.
- **Degenerate bin grids**: `build_bin_centers` falls back to a single bin if `stop_time - start_time < bin_size`, and clips the final bin end to `stop_time`.
- **Unknown image names**: fall back to the `gray` code.

Gaps: there is **no try/except around session loading**, so one corrupt file would abort the whole conversion (it did not happen in practice); a trial whose `stop_time` runs past the end of the ophys recording is **not clipped or dropped** — `searchsorted` simply returns `len(timestamps)` for those bins and they are silently filled with zeros; and the 3,947 all-zero-neural trials (4.68 %) are retained rather than dropped or flagged in metadata.

ii.
```python
    aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
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
    if finite.sum() == 0:
        return np.full(query_times.shape, np.nan, dtype=np.float32)
    if finite.sum() == 1:
        v = float(values[finite][0])
        return np.full(query_times.shape, v, dtype=np.float32)
    ...
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
```
```python
            "excluded_sessions": [
                {"experiment_id": p.experiment_id, "session_type": p.session_type,
                 "reason": p.excluded_reason} for p in excluded],
```

iii. Step 5 Key Decision 4: *"local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values."* Planned sanity check: *"Session-inclusion sanity check: exactly the 3 sessions with missing eye tracking should be excluded for pupil-output completeness, unless another issue is discovered"* — confirmed in Step 9 (*"Exclusions: `3` experiments with missing eye tracking, `0` experiments with fewer than two valid go/catch trials"*). On the all-zero trials, Step 10: *"investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition."* Step 10 also records an edge-case review: *"validated a minimum-length trial with raw reconstruction, and confirmed that all-zero neural warnings arise from genuine zero-event trials … rather than off-by-one errors at trial boundaries."*

## 9-a. What are the most time-consuming steps of the code?

i. Measured on the full run (`conversion_full_out.txt`): preview pass **65.0 s**, conversion pass **686.0 s**, total **757.7 s** — so the conversion pass is ~90 % of runtime, averaging 2.4 s/session and scaling with neuron count (6 s for the 500–600-neuron sessions, 1 s for the small multiscope planes). Within `convert_session` the two dominant costs are (1) the bulk HDF5 reads — the full-session event matrix (up to `(150k frames × 600 ROIs)` ≈ 360 MB), plus ~280k running samples and up to ~273k pupil samples; and (2) the pure-Python `for b in range(T)` bin loop, which executes ~(trials × bins) ≈ 26k–35k slice-and-sum operations per session. The preview pass costs a second full open-and-read of the trials, presentation, running and eye-tracking tables for all 284 files.

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
```

iii. Step 6: *"Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known"* and *"Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."* Step 7's timing table estimated *"~5.95 s / session"* from the (deliberately neuron-heavy) sample sessions and a *"Conservative full-conversion upper bound ~28.2 min"*, noting the true figure would be lower because *"the sample sessions are much larger than the dataset average (635 neurons/sample session vs ~148 neurons mean locally)"* — the realised 12.6 min confirms that estimate, so no further optimisation was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops that remain in Python and are vectorisable:
- **`convert_session`'s per-bin loop** (`for b in range(T)`) — the main hot loop. Since the frames are contiguous and sorted, this is exactly `np.add.reduceat(event_data, frame_starts)` or a `cumsum` difference: `csum[frame_ends] - csum[frame_starts]`, computed once per session for all trials at once. The AI identified this one explicitly.
- **`make_image_series`'s per-presentation loop** — the repeated `(centers >= start) & (centers < stop)` scan is O(n_presentations × n_bins); a single `np.searchsorted` of `centers` into the flash boundaries would assign all bins in one pass.
- **`build_trial_specs`'s per-trial loop** — the flag arrays are already vectorised; the outcome index is `np.argmax(np.stack([hit, miss, fa, cr]), axis=0)` and the validity check is a single `sum(axis=0) == 1` assertion.
- **`collect_time_window_values`'s per-trial loop** in the preview — `searchsorted` on all trial bounds at once, then a single fancy-index/concatenate.
- **The outer per-trial loop** in `convert_session`, which recomputes bin grids and re-interpolates running/pupil trial by trial instead of interpolating the whole session once onto a concatenated query vector.

ii.
```python
    for idx in row_idx:
        ...
        in_window = (centers >= start) & (centers < stop)     # O(n_pres x n_bins)
```
```python
    for spec in trial_specs:
        lo = np.searchsorted(times, spec.start_time, side="left")
        hi = np.searchsorted(times, spec.stop_time, side="left")
        if hi > lo:
            segments.append(values[lo:hi])
```

iii. The AI documented only the first of these, in Step 6: *"Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."* The stated reason for not vectorising was peak-memory control, and Step 7's timing estimate (well under the 15-minute guidance in the instructions) meant no further optimisation was judged necessary. The other loops are not mentioned anywhere in the notes.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design repeats a substantial amount of work:
- **Every NWB file is opened and parsed twice.** `read_trials`, `build_trial_specs` and `read_task_presentations` run once in `session_preview` and again in `convert_session`; the running-speed and eye-tracking arrays (~270k and ~100–270k samples) are read, blink-masked and area-converted in both passes.
- **`plot_processing_summary` recomputes** the bin grid, image/change series, and running/pupil interpolation for the chosen trial that `convert_session` already computed moments earlier.
- **`build_bin_centers`** is re-derived per trial (unavoidable given variable trial lengths, but the interpolation of running/pupil could be done once per session).
- `find_presentation_rows` re-scans the full presentation array for every trial (O(n_trials × n_presentations)) instead of a single sorted merge.

ii.
```python
        trials = read_trials(f)            # in session_preview
        specs = build_trial_specs(trials)
        presentations = read_task_presentations(f)
```
```python
        trials = read_trials(f)            # again, in convert_session
        presentations = read_task_presentations(f)
```
```python
        running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
        running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
        pupil_times = ...; pupil_area = ...; likely_blink = ...
        pupil_diameter = pupil_area_to_diameter(pupil_area)   # duplicated in both passes
```

iii. Step 6 explicitly acknowledges and justifies the duplication: *"Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."* The global image vocabulary and the global running/pupil percentile edges genuinely must be known before any trial can be written, and Step 6 lists the compensating measures: *"Preview pass avoids loading neural event matrices"*, *"Conversion uses direct dataset reads and processes one session at a time to cap peak memory"*, *"Sample-mode preview now short-circuits once 2 eligible sessions are found."* Measured cost of the duplicated pass is 65 s of 758 s (~9 %).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose result never reaches the pickle:
- **Unused trial columns.** `read_trials` decodes `change_frame`, `change_time`, `initial_image_name`, `change_image_name`, `go`, `catch` and `id`. `initial_image_name`/`change_image_name`/`change_frame` are never referenced at all; `TrialSpec.change_time` is populated for every trial and never read; `is_go`/`is_catch` are used only to pick which trial to draw in `--show-processing`.
- **Unused presentation columns.** `is_sham_change`, `trials_id` and `active` are read, concatenated and sorted for every session but never consulted.
- **Discarded return values.** `load_neural_events` builds and returns `event_rois[valid_mask]`, which the caller throws away (`event_data, _ = load_neural_events(f)`); `build_bin_centers` computes `widths` only to derive `centers`.
- **`trials = read_trials(f)` inside `convert_session`** is used solely by `plot_processing_summary`, i.e. for at most 2 sessions in `--show-processing` mode, yet it is re-read for all 281.
- **Full-resolution pooling for four quantiles.** The preview concatenates every raw running and pupil sample inside every trial of every session (tens of millions of values) purely to compute 4+4 percentile edges; a subsample would give the same edges to far better than bin precision.
- **`ophys_session_id`** is parsed for every file and never used (see 1-c).
- Per-trial empty `(0, T)` input arrays are allocated for all 84,313 trials — required by the target format, so arguably not waste, but they carry no information.

None of this is a large fraction of the 758 s runtime; the dominant costs (bulk array reads, per-bin sums) are all load-bearing.

ii.
```python
    names = ["id", "start_time", "stop_time", "go", "catch", "aborted", "auto_rewarded",
             "hit", "miss", "false_alarm", "correct_reject", "change_time", "change_frame",
             "initial_image_name", "change_image_name"]
```
```python
        event_data, _ = load_neural_events(f)     # event_rois discarded
```
```python
                change_time=float(change_time[idx]) if np.isfinite(change_time[idx]) else math.nan,
```
```python
            input_trial = np.empty((0, T), dtype=np.float32)
```

iii. CONVERSION_NOTES.md does not address this question. Step 6's "Code inefficiencies identified" covers only the double file open and the per-bin summation loop; the unused-field reads, the discarded `event_rois`, the unconditional `read_trials` in `convert_session`, and the full-resolution quantile pooling are not identified anywhere. The closest related statement is the Step 7 optimisation note that the preview was changed to *"pool raw running/pupil samples within valid trials instead of interpolating every trial during preview"* — which reduced CPU but left the full-resolution pooling in place.
