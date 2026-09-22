# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object model. It discovered that `pynwb` could not read these NWB files in the environment (abstract-class error from an hdmf/pynwb version mismatch), so it reads the NWB/HDF5 files directly with `h5py`, mirroring the SDK's field semantics by hand. It globs every `behavior_ophys_experiment_*.nwb` file under `/app/data/.../behavior_ophys_experiments` (284 files) and processes **all** of them, with **no filtering on `project_code`**. It is aware from `ophys_experiment_table.csv` that the local subset contains 239 `VisualBehavior` experiments plus 45 `VisualBehaviorMultiscope` experiments, but it never filters or discusses that split. Loading is done in two passes over the files: a lightweight "preview" pass (trials, stimulus presentations, running, pupil — no neural matrices) used to build global image vocabulary / percentile bin edges / eligibility, then a "conversion" pass that re-opens each file and reads the neural event matrix and everything else again. Per file it reads: `/intervals/trials`, every `/intervals/*_presentations` group that has an `image_name` column, `/processing/running/speed`, `/acquisition/EyeTracking/*`, `/processing/ophys/event_detection`, `/processing/ophys/dff/traces/timestamps`, `/processing/ophys/image_segmentation/cell_specimen_table/valid_roi`, and metadata attributes.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
```
```python
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
log(f"Found {len(files)} NWB experiment files under {EXPERIMENT_DIR}")
eligible, excluded = collect_previews(files=files, bin_size_sec=BIN_SIZE_SEC,
                                      sample_mode=args.sample, required_eligible=2)
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

iii. From CONVERSION_NOTES.md Step 6: "Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment" and "Mirrors SDK semantics already serialized into NWB". Step 10's reference-code comparison states: "Same underlying NWB tables/fields are used; I read them directly with `h5py` because `pynwb` failed in this environment." The two-pass design is justified as: "lightweight preview pass to identify eligible sessions, collect global image categories, and compute global running/pupil percentile edges" then "conversion pass to build neural/input/output trial arrays", "intentional to avoid storing large neural matrices before global percentile/bin definitions are known." No justification is given anywhere for including the `VisualBehaviorMultiscope` experiments alongside the `VisualBehavior` ones.

## 1-b. How are the data split into subjects?

i. Subjects are taken from each NWB file's `/general/subject/subject_id` string. A subject registry is built in first-encounter order while iterating the eligible files, and `subject_idx` records, per session, the index into `subjects`. This yields 38 mice (37 from `VisualBehavior` plus 1 from `VisualBehaviorMultiscope`).

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

iii. From the Step 5 mapping table: "`/general/subject/subject_id` or metadata `mouse_id` → `subjects`, `subject_idx`; String subject identifiers with session-level index mapping; Session order follows converted session order." Step 9 cross-checks the count: "Subjects | `82` mice in whitepaper/paper scope | `38` local mice represented in eligible experiments | `38` | `38` | Matches local code/data; paper differs because of scope."

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one "session".** The AI does not group imaging planes by `ophys_session_id`, even though it reads that attribute into `SessionPreview.ophys_session_id` (the field is never used afterwards). For the 239 single-plane `VisualBehavior` experiments this is correct (1 plane per session). For the 45 `VisualBehaviorMultiscope` experiments, which come from only **8** real ophys sessions of a single mouse (3–7 planes each), the same behavioural session is emitted 3–7 times as separate entries in `neural`/`output`, each with a different subset of the simultaneously recorded neurons and with identical trial timing and identical output labels. The final dataset therefore reports 281 "sessions" where the reference definition would give 247 (239 + 8), and 239 under the reference's project filter.

ii.
```python
@dataclass
class SessionPreview:
    path: Path
    experiment_id: int
    ophys_session_id: int      # read, but never used to group planes
    subject_id: str
    brain_region: str
    ...
```
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```
```python
brain_region_idx = np.full(
    event_data.shape[1], brain_region_to_idx[preview.brain_region], dtype=np.int64
)
```

iii. No explicit justification is given for the file-per-session equivalence; CONVERSION_NOTES.md consistently uses "sessions" and "experiment files" interchangeably ("Eligible local experiment files: `281`", "Sessions | ... | `281` eligible local experiment files"). Step 4 does note "`experiments_per_session_mean 1.1497`" was observed in the metadata table, and Step 5 decision 5 acknowledges the two acquisition regimes ("Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz)"), so the AI knew multiplane data were present but resolved only the sampling-rate consequence (via rebinning), not the plane-grouping consequence.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `/intervals/trials` table (the SDK trials table). Every row that is not `aborted` and not `auto_rewarded` becomes a trial, i.e. the go and catch trials. The trial window is the full `start_time` → `stop_time` interval (variable length; converted trials are 71–127 bins, i.e. ~7–12.7 s, median ≈ 86 bins). Each trial is then laid out on a fixed 100 ms grid anchored at `start_time`. 84,313 trials over 281 sessions.

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

iii. Step 5 decision 2: "**Use SDK-valid trials only**: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed and consistent with the trial table semantics." Step 4 resolution: "Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly." Step 5 decision 7: "**Alignment event = trial start**: The trial itself is the natural unit requested by the user."

## 1-e. How are trials filtered based on quality controls?

i. Trial/session filters applied:
- `aborted` and `auto_rewarded` trials dropped (per instructions).
- A trial must have exactly one of `hit`/`miss`/`false_alarm`/`correct_reject` True, otherwise the conversion **raises** (fail-fast rather than skip). It never triggered on the 284 files.
- Session excluded if the NWB has no eye-tracking group (`missing_eye_tracking`) — 3 sessions dropped.
- Session excluded if it has fewer than 2 valid trials (`fewer_than_2_valid_trials`) — 0 sessions.
- Session excluded if fewer than 2 finite pupil samples exist inside trial windows (`insufficient_valid_pupil_samples`) — 0 sessions.
- No filtering on session type: the passive sessions (`OPHYS_2_images_A_passive`, `OPHYS_5_images_B_passive`) are kept.
- Trial windows are never clipped to the recording end; `np.searchsorted` simply yields empty frame ranges and those bins stay zero.

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

iii. Step 5 decision 3: "**Keep passive sessions if they have valid GO/CATCH trials**: Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs." Step 5 decision 4: "**Exclude sessions with missing eye tracking**: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values." Step 4: "Exclude `aborted` and `auto_rewarded` trials from converted trial set, as required by the user task."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the **event-detection** output, `/processing/ophys/event_detection/data` (the L0-regressed discrete calcium event magnitudes), **not** from dF/F. ROIs are restricted to those with `valid_roi == True` in `/processing/ophys/image_segmentation/cell_specimen_table`, indexed through `/processing/ophys/event_detection/rois`. Timestamps are taken from `/processing/ophys/dff/traces/timestamps` (verified identical to the event-detection timestamps). Total 41,871 neurons.

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

iii. Step 4 discrepancy resolution: "SDK loads both dF/F traces and event-detection outputs ... Whitepaper describes dF/F processing; paper states analyses were performed on discrete calcium events → Use the available processed neural signal in a way that matches the paper/code path. Tentative resolution: prefer event-detection outputs for neural activity because the analysis paper explicitly uses them, while preserving SDK ROI filtering and timestamps." Step 5 key decision 1: "**Neural signal = event-detection output, not dF/F**: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events. This is the closest match to the paper while still using the SDK-defined loading and ROI filtering." Also noted: "Use raw events, not visualization-only `filtered_events`."

## 2-b. How is the `neural` data processed?

i. No neuropil correction / dF/F computation is done (the events are already fully preprocessed by the Allen pipeline). Processing consists of (1) valid-ROI selection, and (2) **temporal rebinning**: for each trial and each 100 ms bin, the event magnitudes of all ophys frames whose timestamps fall in `[bin_start, bin_end)` are **summed** per neuron. The result is a `(n_neurons, n_bins)` float32 matrix. No z-scoring, smoothing, baseline subtraction, or normalisation by bin width/frame count is applied. Because the trial's last bin is truncated at `stop_time`, it usually covers <100 ms and its sums are correspondingly smaller. Because 31 Hz frames do not divide evenly into 100 ms, bins alternately contain 3 or 4 frames. 4.68% of trials end up with an all-zero neural matrix (genuine zero-event trials in sparse sessions with few neurons).

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
Metadata records the rule:
```python
"neural_representation": "ophys event-detection magnitudes",
"binning_rule": "sum event magnitudes within each 100 ms trial bin",
```

iii. Step 5 mapping: "Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions." Step 10, on the all-zero warnings: "investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level QC is the SDK's `valid_roi` flag (`valid_roi[event_rois]`). Nothing else — no SNR, no event-rate, no motion/z-drift screen (those were already applied upstream at experiment release). In the local files every ROI is in fact flagged valid, so the filter is a no-op here but preserves SDK semantics.

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool
)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. Step 3 neuron curation notes: "The SDK code path identified in Step 1 also filters invalid ROIs (`valid_roi == True`) at load time; this is consistent with the whitepaper's QC emphasis." Step 4 resolution: "In local NWBs all ROI rows inspected so far are marked valid (`valid_roi` sum equals ROI count), but the field exists explicitly → Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid." Step 10 verified the count: 41,871 valid ROIs, `VISp 41,633` / `VISl 238`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. The bin grid runs from `start_time` to `stop_time` in 100 ms steps, and neural frames are assigned to bins by `np.searchsorted` on the ophys timestamps, so neural, stimulus, running and pupil all share the identical bin grid of that trial. Metadata declares `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")
frame_ends   = np.searchsorted(ophys_timestamps, ends,   side="left")
```
```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 5 decision 7: "**Alignment event = trial start**: The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`)." Step 4: "Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases", supported by the whitepaper statement that all streams are hardware-synchronised on a single 100 kHz board.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — everything is rebinned to a uniform 100 ms grid** (`BIN_SIZE_SEC = 0.1`, `metadata['time_bin_size'] = 100.0`). Native ophys resolution is 31 Hz (~32.3 ms) for the single-plane `VisualBehavior` experiments and ~11 Hz for the multiscope experiments the AI also included; 100 ms was chosen as the common grid. The grid is regular within a trial except for the last bin, which is clipped at `stop_time` and is therefore shorter than 100 ms (its neural sums are not rescaled).

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
```
```python
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
ends = np.minimum(starts + bin_size_sec, stop_time)
```

iii. Step 5 decision 5: "**Common time base via uniform rebinned trial bins**: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions." Decision 6: "**Tentative common bin size = 100 ms**: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams." Step 10 comparison: "Native SDK data remain at native sample rates; the papers do not prescribe a common decoder bin size → Difference is intentional and required by the target format, not a mismatch in source processing."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations table**, not the trials table: every `/intervals/*_presentations` group that has an `image_name` column (here `Natural_Images_Lum_Matched_set_training_2017_presentations`; the `natural_movie_one` and `spontaneous` tables are skipped because they lack `image_name`). The fields used are `start_time`, `stop_time`, `image_name`, and `omitted`. The global vocabulary is `['gray'] + sorted(all image names)` = 17 classes (gray + 16 images across image sets A and B).

ii.
```python
def read_task_presentations(f):
    for key in interval_root.keys():
        if key == "trials":
            continue
        group = interval_root[key]
        if "image_name" not in group or "start_time" not in group or "stop_time" not in group:
            continue
        rows.append(read_interval_group(group, keep_names))
```
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

iii. Step 5 mapping: "Stimulus presentation `image_name` + presentation timing + omission state → `output[0]` (`image_identity`); Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods; Time-varying categorical output. Global category set will include all local image names plus `gray`." The presentations table is preferred because it gives the actual on-screen flash timing rather than a per-trial summary.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per trial, the presentations overlapping the trial window are found, and for each non-omitted presentation the bins whose **centre** falls inside `[presentation start, presentation stop)` are labelled with that image's integer code. All other bins — the 500 ms grey inter-stimulus intervals and omitted flashes — keep the default `gray` code (0). This makes `gray` the modal class: 67.0% of all bins (consistent with 250 ms flash / 750 ms cycle). Codes are int16 from a global, sorted, cross-session mapping so codes are comparable across sessions.

ii.
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

iii. Step 5 decision 8: "**Image identity will include a `gray` class**: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin." Step 10 records a bug found and fixed here: "`output_values[0]` incorrectly included an unused `omitted` image label even though omitted flashes were encoded as `gray` in the time series. Fixed by excluding `omitted` from `unique_nonempty_images()` and re-running" — after which "`image_identity` verification range is now `[0.0, 16.0]`, matching `gray + 16` task images with no stray omitted-stimulus class."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same `centers` array used to build the neural bins of that trial, so the two are aligned by construction; both derive from absolute experiment time and the ophys timestamps. A bin is assigned an image iff its centre lies in the flash interval, so a flash shorter than a bin, or straddling a bin boundary, is resolved by the centre rule.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
frame_starts = np.searchsorted(ophys_timestamps, starts, side="left")   # neural
...
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```
```python
def find_presentation_rows(presentations, start_time, stop_time):
    starts = presentations["start_time"]; ends = presentations["stop_time"]
    mask = (starts < stop_time) & (ends > start_time)
    return np.flatnonzero(mask)
```

iii. Step 10: "Alignment matches the reference time bases; the only added step is the decoder-required common 100 ms rebinning." Step 12: "Temporal alignment was rechecked using the processing plots generated during sample conversion; stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch." Step 10's `cache/step10_raw_checks.py` reconstructed output rows for 5 trials straight from the NWB and matched them with `np.allclose()`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the presentations table's `is_change` flag together with `omitted` and the presentation `start_time`/`stop_time`. `is_change` is True only for true image changes; sham changes on catch trials are flagged `is_sham_change` instead, so catch trials correctly receive an all-zero change series. (The trials table's `change_time`/`go` columns are read into `TrialSpec` but are only used for the diagnostic plots, not for the output.)

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 mapping: "Stimulus presentation `is_change` + presentation timing → `output[1]` (`image_change`); Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0; ... trial `change_time` for sanity checks."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary int16 series, 1 only in the bins whose centre falls within the **changed-image flash itself** (~250 ms, i.e. 2–3 bins at 100 ms) and 0 everywhere else, including the following grey period. Pooled over the dataset this gives 2.63% positive bins.

ii. See 4-a; the series is stacked as row 1 of the output matrix:
```python
output_trial = np.vstack([
    image_series.astype(np.int16),
    change_series.astype(np.int16),
    running_bins,
    pupil_bins,
    outcome_series,
])
```

iii. Step 5 decision 9: "**Image-change target will mark the changed-image presentation, not only a single instant**: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure." Step 12 notes the consequence: "`image_change` positive bins are sparse (`2.63%`), so balanced accuracy modestly above `0.5` is still meaningful."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is natively binary (`no_change` / `change`) and is read straight off the `is_change` boolean flag. The only implicit "threshold" is temporal: the positive window is the flash duration, and a bin is positive iff its centre lies inside that flash.

ii.
```python
"output_values": [ image_values, ["no_change", "change"], RUN_BIN_VALUES, PUPIL_BIN_VALUES, TRIAL_OUTCOME_VALUES ],
```

iii. Not applicable — the AI (like the reference) treats this as an inherently categorical/binary event indicator; no continuous quantity is being discretised.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: same `centers` grid, same per-bin centre-in-interval rule, produced by the same `make_image_series()` call that produces the image identity row, and therefore on the same grid as the neural bins.

ii.
```python
row_idx = find_presentation_rows(presentations, spec.start_time, spec.stop_time)
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. Same as 3-c; the Step 7 `--show-processing` plots overlay the raw presentation intervals (`axvspan`) and the raw `is_change` markers (`axvline`) on the rebinned `change_series` specifically so that any misalignment would be visible. Step 5 planned sanity check: "Raw-vs-converted image change spot check: verify the converted binary change series turns on only for the changed-image flash and matches trial `change_time` / presentation `is_change`", reported as passing in Step 10.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `/processing/running/speed/data` and `/processing/running/speed/timestamps` — the SDK's processed `running_speed` (cm/s from the wheel encoder), read at its native ~60 Hz/30 Hz sampling.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 5 mapping: "Running speed timeseries → `output[2]` (`running_speed_bin`); Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins; `RunningSpeed.from_nwb`." Step 3 notes the whitepaper "explicitly points to AllenSDK running-processing code for the implementation", i.e. the stored `speed` is already the reference-processed quantity.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the raw speed onto the trial's 100 ms bin **centres**, with constant (edge-value) extrapolation beyond the recorded range and non-finite samples dropped first. No smoothing, no rectification, no |speed| transform. The interpolated value is then discretised (5-c). A hard assertion rejects any non-finite interpolated value.

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
running_interp = interpolate_series(running_times, running_values, centers)
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
```

iii. Step 4: "Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases", justified by the whitepaper's statement that "all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **global** (cross-session) equal-count percentile bins. The bin edges are the 20/40/60/80th percentiles of the pool of all *raw* running samples that fall inside any included trial window of any included session (collected in the preview pass); the edges are then applied with `np.digitize` to the *interpolated* per-bin values. Realised edges: `[-0.0229, 0.1175, 10.039, 31.567]` cm/s. Verified marginal distribution: `[0.1994, 0.2013, 0.1992, 0.1999, 0.2002]`.

ii.
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
```python
def collect_time_window_values(times, values, trial_specs):
    for spec in trial_specs:
        lo = np.searchsorted(times, spec.start_time, side="left")
        hi = np.searchsorted(times, spec.stop_time, side="left")
        if hi > lo:
            segments.append(values[lo:hi])
    return np.concatenate(segments).astype(np.float32, copy=False)
```

iii. Step 5 decision 10: "**Running and pupil bin edges will be global, not per-session**: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`." Planned sanity check: "Output-balance sanity check: global running and pupil bins should each contain roughly 20% of included samples by construction" — confirmed in Step 9.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly at the same `centers` used for the neural bins of that trial, so the two share the grid exactly; no separate resampling step or lag correction is applied.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Same synchronisation argument as 5-b: streams are hardware-synced, so absolute timestamps are directly comparable. Panel 3 of the `--show-processing` figure plots the raw running trace, the interpolated values at bin centres, and the resulting bin labels on one time axis to make misalignment or mis-discretisation visible.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `/acquisition/EyeTracking/pupil_tracking/area_raw` (the raw ellipse-fit pupil area), `/acquisition/EyeTracking/eye_tracking/timestamps`, and `/acquisition/EyeTracking/likely_blink/data`. Blink frames are set to NaN, then area is converted to an equivalent diameter `d = 2*sqrt(A/π)`. (In these files `area_raw = π * major_axis²`, so this recovers `2 × major axis`; masking blinks on `area_raw` reproduces the SDK's `area` column exactly.)

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
def pupil_area_to_diameter(area):
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```

iii. Step 5 mapping: "Eye-tracking pupil signal → `output[3]` (`pupil_diameter_bin`); Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins; `EyeTrackingTable.from_nwb`; Requires eye-tracking availability. Sessions with missing eye tracking will be excluded." Step 3 records the whitepaper basis: "whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter" — i.e. the square-root transform inverts the stored quantity back to a diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames → NaN → dropped; the remaining samples are linearly interpolated onto the trial's bin centres (so blink gaps are bridged rather than filled with a sentinel), with edge-value extrapolation; non-finite results raise. The interpolated diameter is then discretised into the 5 global percentile bins.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```
(`interpolate_series` drops the NaN'd blink samples via its `finite` mask before `np.interp`.)

iii. Same as 5-b/6-a: blink removal follows the SDK's `likely_blink` flag; interpolation is justified by hardware synchronisation of the eye-tracking and ophys clocks. Step 5 decision 4 explains why whole sessions are dropped instead of fabricating pupil values when eye tracking is absent.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global equal-count percentile bins, computed exactly as for running speed from the pooled raw in-trial pupil samples of all eligible sessions. Realised edges: `[72.63, 83.38, 93.05, 104.90]` (pixels). Realised distribution `[0.2029, 0.2058, 0.1929, 0.1866, 0.2119]` — slightly off 20% each because the edges come from the raw-sample pool while the labels are applied to the interpolated bin-centre values.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
```

iii. Step 5 decision 10 (global, not per-session bin edges) plus the output-balance sanity check; Step 9 reports the realised quintile fractions as consistent "by construction".

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated at the same trial `centers` as the neural bins — identical mechanism to running speed, no extra alignment step.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Same justification as 5-d; panel 4 of the `--show-processing` figure overlays the raw pupil trace, the interpolated bin-centre values and the discretised bins for visual verification.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `/intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order (codes 0–3).

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

iii. Step 5 mapping: "Trial outcome flags `hit`, `miss`, `false_alarm`, `correct_reject` → `output[4]` (`trial_outcome`); Single categorical value per trial, repeated across all time bins in that trial to keep output arrays uniformly time-varying; `Trials.from_nwb`, `Trial._get_trial_data`; Aborted and auto-rewarded trials excluded before conversion." Step 3 records the task definition: "Trial structure consists of GO and CATCH trial types, which combine with behavior to yield HIT, MISS, FALSE ALARM, and CORRECT REJECTION outcomes."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single integer code is broadcast across every time bin of the trial so that the output matrix is uniformly `(5, n_timepoints)`. The exactly-one-flag invariant is asserted at trial-construction time (hard failure, not a fallback class). Realised distribution: hit 18.2%, miss 69.2%, false alarm 1.0%, correct reject 11.5% — miss-dominated because the passive sessions (where the mouse cannot respond) are retained.

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

iii. Step 5 decision 11: "**Trial outcome will be repeated across time bins**: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form." Step 12 discusses the imbalance: "`trial_outcome` is imbalanced (`hit 18.2%`, `miss 69.2%`, `false_alarm 1.0%`, `correct_reject 11.5%`) and is a four-class static label, making it harder than the paper's qualitative two-class hit/miss analyses."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is mostly *exclude-or-fail* rather than impute:
- **Missing eye tracking** (3 files): whole session excluded and recorded in `metadata['excluded_sessions']` with a reason string.
- **Blinks**: set to NaN and dropped, then bridged by linear interpolation (never assigned a sentinel bin).
- **NaN boolean trial flags**: `np.nan_to_num(..., nan=0.0).astype(bool)` treats missing flags as False.
- **Outside the recorded range**: edge-value (constant) extrapolation of running/pupil rather than NaN.
- **Any remaining non-finite running/pupil value**: raises immediately (fail loud).
- **Trial with ≠1 outcome flag**: raises immediately.
- **Omitted stimulus flashes**: mapped to the `gray` class, deliberately not a separate class.
- **Degenerate sessions**: `<2` valid trials or `<2` finite pupil samples excluded; `<2` eligible sessions overall aborts the run.
- **Zero-length bin grid**: a trial shorter than one bin still yields one bin.
- **Genuinely empty neural bins/trials**: left as zeros (3,947 all-zero trials, 4.68%), investigated and kept.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
```
```python
if starts.size == 0:
    starts = np.array([start_time], dtype=np.float64)
```
```python
out = np.interp(query_times, t, v, left=v[0], right=v[-1])
```
```python
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
```
```python
"excluded_sessions": [
    {"experiment_id": p.experiment_id, "session_type": p.session_type, "reason": p.excluded_reason}
    for p in excluded
],
```

iii. Step 5 decision 4 (drop sessions rather than "fabricating pupil values"); Step 10 issue log for the all-zero trials: "direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation"; Step 10 fix log for the omitted-image class. The comment in `unique_nonempty_images` records the omission policy inline.

## 9-a. What are the most time-consuming steps of the code?

i. The script prints timing for both passes and per session. On the full run: preview pass 65.0 s, conversion pass 686.0 s, total 757.7 s (~12.6 min). Within the conversion pass the cost is dominated by (1) the per-bin Python loop that sums event magnitudes (7.2 M iterations: 84,313 trials × ~86 bins), and (2) reading the full `(140k, n_roi)` event-detection matrices into memory. Sessions with ~500–600 neurons take 4–6 s each vs 1–2 s for small ones. Every NWB file is also opened and parsed twice.

ii.
```python
preview_start = time.perf_counter()
...
log(f"Preview pass completed in {time.perf_counter() - preview_start:.1f}s")
...
log(f"Conversion pass completed in {time.perf_counter() - convert_start:.1f}s")
```
```python
stats = {"experiment_id": ..., "trial_count": ..., "neuron_count": ...,
         "elapsed_sec": int(round(time.perf_counter() - session_start))}
```

iii. Step 6: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known" and "Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU." Step 7 estimated and Step 9 confirmed the full run at ~12.6 min, under the 15-minute budget, which is why no further optimisation was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain:
- `for b in range(T)` in `convert_session` — the dominant one. It could be replaced by a single `np.add.reduceat` (or a cumulative sum differenced at `frame_starts`/`frame_ends`) over the whole trial, or even over the whole session at once, with no extra memory.
- `for spec in preview.trial_specs` — the outer per-trial loop; bin grids and `searchsorted` calls for all trials could be computed in one batch.
- `for idx in row_idx` in `make_image_series` — per-presentation loop building the image/change series; a single `np.searchsorted` of `centers` into the presentation boundaries would label all bins at once.
- `for idx in range(raw_count)` in `build_trial_specs` — the boolean arrays are already vectorised, so the row loop only assembles dataclasses and could be replaced by `np.argmax` over the stacked outcome flags plus a validity check.

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
```

iii. The AI explicitly acknowledged the main one in Step 6 ("per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU") and chose not to change it because the measured full-run time fit inside the instructions' 15-minute budget. It did add other speedups: "Preview pass avoids loading neural event matrices", "Conversion uses direct dataset reads and processes one session at a time to cap peak memory", "Sample-mode preview now short-circuits once 2 eligible sessions are found."

## 9-c. What processing does the code repeat multiple times?

i. The largest repetition is the **two full passes over every NWB file**. `read_trials()`, `read_task_presentations()` (including the sort over all presentation rows), and the running / eye-tracking reads and blink-masking are all executed once in `session_preview()` and then again in `convert_session()`; only `TrialSpec` objects and the pooled running/pupil samples are carried over. `collect_time_window_values()` additionally re-walks all trial windows just to build the percentile pools. In `plot_processing_summary()` the bin grid, image series, and both interpolations are recomputed for a trial that was just processed in the main loop. `build_bin_centers()` is also called once per trial in the loop and again inside the plotting routine.

ii.
```python
# pass 1
def session_preview(path, bin_size_sec):
    with h5py.File(path, "r") as f:
        trials = read_trials(f)
        presentations = read_task_presentations(f)
        running_times = np.asarray(f["/processing/running/speed/timestamps"], ...)
        ...
# pass 2
def convert_session(preview, ...):
    with h5py.File(preview.path, "r") as f:
        ...
        trials = read_trials(f)              # only consumed by the plotting routine
        presentations = read_task_presentations(f)
```

iii. Step 6: "Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known." The design constraint is real — global percentile edges and the global image vocabulary must be known before any trial can be encoded — but the re-reading of the small tables (trials, presentations, running, pupil) is an avoidable consequence, since only the neural matrices are large enough to justify deferring.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- `pupil_area_to_diameter()` applies `2*sqrt(A/π)` to every eye-tracking sample of every session. Since the pupil output is discretised by **percentiles** and the transform is strictly monotonic, the bin labels would be bit-identical without it — it is pure cost (though it does make the intermediate quantity interpretable, and the notes justify it on that basis).
- `trials = read_trials(f)` inside `convert_session()` is used **only** by `plot_processing_summary()`, yet it is executed for all 281 sessions even when `--show-processing` is off.
- `load_neural_events()` builds and returns the filtered `event_rois` array, which the caller discards (`event_data, _ = load_neural_events(f)`).
- `TrialSpec.change_time`, `.is_go`, `.is_catch` are computed for all 84,313 trials but only read by the plotting code.
- `read_trials()` pulls `change_frame`, and `read_task_presentations()` pulls `is_sham_change`, `trials_id` and `active`, none of which are ever used.
- `SessionPreview.ophys_session_id` is parsed for every file and never used.
- The full native-rate running and pupil sample pools for all 281 sessions are held in memory only to compute eight percentile values; subsampling would be equivalent.
- An empty `np.empty((0, T))` array is allocated and pickled for each of the 84,313 trials, since the task has no decoder inputs.
- Neural data are stored as dense float32 despite being sparse event magnitudes, giving a 4.3 GB pickle.

ii.
```python
event_data, _ = load_neural_events(f)          # rois discarded
...
trials = read_trials(f)                        # used only by plot_processing_summary
```
```python
input_trial = np.empty((0, T), dtype=np.float32)
```
```python
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)   # monotonic; irrelevant to percentile bins
```

iii. The AI did not flag any of these as waste. It justified the diameter conversion on interpretability/faithfulness grounds (Step 5 mapping: "convert to equivalent diameter `2*sqrt(area/pi)`", grounded in the whitepaper's ellipse-fit description), and the empty input arrays on format grounds (Step 5 decision 12: "**No decoder inputs**: `input_names` will be empty and every `input` trial entry will be an empty 2D array with the same time dimension as the corresponding trial"). None of these items affects correctness; they are cost only.
