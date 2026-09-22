# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK object model at runtime. It tried `pynwb` first, which crashed on these files in this environment (hdmf/namespace error), and fell back to reading the NWB/HDF5 files directly with `h5py`. Data discovery is a filesystem glob of every `behavior_ophys_experiment_*.nwb` file in `/app/data/.../behavior_ophys_experiments` (284 files). No filtering by `project_code` and no use of `ophys_experiment_table.csv` for discovery (that CSV is only read in `--sample` mode, to sort files by cell count so the sample uses high-neuron sessions). Every field is read from a hard-coded HDF5 path that mirrors the SDK's serialization: trials from `/intervals/trials`, stimulus flashes from `/intervals/*_presentations`, neural events from `/processing/ophys/event_detection`, running from `/processing/running/speed`, pupil from `/acquisition/EyeTracking/*`, ophys timebase from `/processing/ophys/dff/traces/timestamps`. Each file is opened **twice**: once in a lightweight "preview" pass (metadata, trial specs, pooled running/pupil samples, image vocabulary) and once in the conversion pass.

ii.
```python
DATA_ROOT = Path("/app/data/visual-behavior-ophys-1.1.0")
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"

def list_nwb_files() -> List[Path]:
    return sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
```
```python
def read_trials(f: h5py.File) -> Dict[str, np.ndarray]:
    group = f["/intervals/trials"]
    names = ["id", "start_time", "stop_time", "go", "catch", "aborted",
             "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
             "change_time", "change_frame", "initial_image_name", "change_image_name"]
    return read_interval_group(group, names)
```
```python
files = sort_files_for_mode(list_nwb_files(), sample_mode=args.sample)
eligible, excluded = collect_previews(files=files, bin_size_sec=BIN_SIZE_SEC,
                                      sample_mode=args.sample, required_eligible=2)
```

iii. From CONVERSION_NOTES Step 6: *"Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment"* and *"Mirrors SDK semantics already serialized into NWB"*. Step 1 of the notes documents the SDK functions (`BehaviorOphysExperiment.from_nwb`, `CellSpecimens.from_nwb`, `Trials.from_stimulus_file`, `Presentations.from_stimulus_file`, `EyeTrackingTable.from_nwb`, …) that produce exactly these NWB tables, and Step 10 contains an explicit "reference code comparison" table arguing the h5py reads use "the same underlying NWB tables/fields". The two-pass design is justified as "intentional to avoid storing large neural matrices before global percentile/bin definitions are known."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `/general/subject/subject_id` strings read from each NWB file. A subject is registered the first time a file belonging to it is converted; `subject_idx` is the index of that subject for each emitted session. This yields 38 mice.

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

iii. Step 5 mapping table: *"`/general/subject/subject_id` or metadata `mouse_id` → `subjects`, `subject_idx`; String subject identifiers with session-level index mapping."* Step 2 cross-checked the count against the metadata CSVs (38 mice present locally vs 107 in the full release).

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one output session.** The AI did not group imaging planes by `ophys_session_id`. Its own Step 2 exploration recorded that the 284 local files correspond to only **247 unique `ophys_session_id`s**, with "Experiments per local ophys session: range 1-7, mean 1.15", but the conversion still emits one session per file (281 after exclusions). Concretely, mouse 457841's 45 VisualBehaviorMultiscope experiments are 8 real recording sessions × 3–7 simultaneously-imaged planes, and they appear in the output as 45 separate sessions with identical trial structure and identical behavioral/stimulus outputs. The verification log shows this directly: `Subject 457841: 45 sessions` (every other mouse has 4–11).

ii.
```python
for i, preview in enumerate(eligible, start=1):
    ...
    neural_trials, input_trials, output_trials, region_idx_arr, stats = convert_session(...)
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```
```python
ophys_session_id = int(f["/general/metadata"].attrs["ophys_session_id"])   # read, but never used to group
```
```python
brain_region_idx = np.full(
    event_data.shape[1], brain_region_to_idx[preview.brain_region], dtype=np.int64
)   # one region per session, so a session can only ever hold one plane
```

iii. No explicit justification is given for keeping planes separate. The closest statements are Step 2 (*"experiment files are the unit of raw neural recording payload"*) and Step 5 decision 5 (*"Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz) … I will rebin all trials to a common bin width"*), i.e. the AI's answer to multi-plane data was to make the bin size uniform, not to merge planes. The Step 9 consistency table silently equates the two, reporting "Sessions … `281` eligible local experiment files".

## 1-d. How are the data split into trials?

i. Trials come from the SDK trials table serialized at `/intervals/trials`. Every non-aborted, non-auto-rewarded trial (i.e. all Go and Catch trials) becomes one output trial, spanning the full `[start_time, stop_time)` window — variable length, median ~8.0 s → ~86 bins of 100 ms. Each trial is represented by a `TrialSpec` holding start/stop/change times, the outcome index, and go/catch flags. 84,313 trials total.

ii.
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
                           stop_time=float(trials["stop_time"][idx]),
                           change_time=..., outcome_idx=outcome_idx,
                           is_go=bool(go[idx]), is_catch=bool(catch[idx])))
```
```python
def build_bin_centers(start_time, stop_time, bin_size_sec):
    starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
    ends = np.minimum(starts + bin_size_sec, stop_time)
    widths = np.maximum(ends - starts, 1e-6)
    centers = starts + 0.5 * widths
    return starts, ends, centers
```

iii. Step 4 discrepancy table: *"Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly."* Step 5 decision 2: *"Use SDK-valid trials only: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed."* Step 1 documents that the SDK builds trials from consecutive `trial_start` frames in the behavior `trial_log`, so `start_time`→`stop_time` is the SDK's own trial interval.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: `aborted` and `auto_rewarded` are dropped; a trial whose hit/miss/false_alarm/correct_reject flags are not exactly one-hot raises an exception (fail-loud rather than skip — it never fired on this dataset). `change_time` is *not* required to be finite (it is finite for all kept trials in this data, verified). Session-level: a file is excluded if it has no eye-tracking group, if it has fewer than 2 valid trials, or if fewer than 2 finite pupil samples fall inside its trial windows. In the full run only 3 files were excluded, all for `missing_eye_tracking`.

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

iii. Step 5 decision 2 (aborted/auto-rewarded per the task spec) and decision 4: *"Exclude sessions with missing eye tracking: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values."* Step 5 decision 3 explains keeping passive sessions: *"Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs."* The planned sanity checks include "per session, valid converted trial count must equal raw trial count minus aborted minus auto_rewarded", which Step 10 reports as passing against a direct raw-NWB recount.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Not dF/F. The AI uses the **L0 event-detection output**, `/processing/ophys/event_detection/data` (shape `(n_frames, n_rois)`), restricted to ROIs with `valid_roi == True` via the event table's `rois` region pointer. The ophys timebase is taken from `/processing/ophys/dff/traces/timestamps` (same clock and length as the event matrix). 41,871 neurons total.

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

iii. Step 5 decision 1: *"Neural signal = event-detection output, not dF/F: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events."* The trajectory shows the AI grepping the paper and finding the sentence *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f"*. It also deliberately used raw events rather than the SDK's `filtered_events`, noting the latter is a visualization-only half-Gaussian smoothing.

## 2-b. How is the `neural` data processed?

i. Three operations: (1) keep only valid ROIs; (2) for each trial, build a uniform 100 ms bin grid over `[start_time, stop_time)` and **sum** the event magnitudes of all ophys frames whose timestamp falls in each bin; (3) store as `float32` `(n_neurons, n_bins)`. No dF/F, no normalization, no smoothing, no z-scoring, and — a consequence of 1-c — no merging of neurons across imaging planes.

ii.
```python
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

iii. Step 5 mapping table: *"Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions."* Metadata records the rule explicitly: `"binning_rule": "sum event magnitudes within each 100 ms trial bin"`. Step 10 verified the rule against raw NWB with `np.allclose()` for a representative non-zero trial, three all-zero trials, and a minimum-length trial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROI-validity filtering: ROIs with `valid_roi == False` are dropped, replicating the SDK's `exclude_invalid_rois=True` default. No event-rate, SNR, or activity-based neuron exclusion, and no trial exclusion for silent neural data (3,947 trials, 4.68%, contain all-zero event matrices and were deliberately kept).

ii.
```python
valid_roi = np.asarray(
    f["/processing/ophys/image_segmentation/cell_specimen_table/valid_roi"], dtype=bool)
valid_mask = valid_roi[event_rois]
event_data = event_data[:, valid_mask]
```

iii. Step 4: *"SDK filters to `valid_roi` by default … Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid."* On the all-zero trials, Step 10: *"investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is **trial start**. The bin grid starts exactly at the trials-table `start_time` and runs to `stop_time`; ophys frames are assigned to bins by `np.searchsorted` on the ophys timestamps, so bins are half-open `[bin_start, bin_end)` in absolute session time with no gaps or double counting. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable-length trials). All other streams are evaluated on the *same* `centers` array, so alignment between neural and outputs is by construction.

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

iii. Step 5 decision 7: *"Alignment event = trial start: The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`)."* Step 3/4 justify using the ophys clock as the common timebase: *"all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz"*, so *"Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms for every trial and session** (`metadata['time_bin_size'] = 100.0`), i.e. yes, explicit rebinning away from the native ophys rate. Native rates in the local data are ~32.3 ms (31 Hz single-plane) and ~93.2 ms (11 Hz multiscope), so a single-plane bin aggregates 3–4 frames and a multiscope bin 1–2 frames. Behavioral streams are point-sampled at the bin centers rather than averaged within the bin.

ii.
```python
BIN_SIZE_SEC = 0.1
...
"time_bin_size": BIN_SIZE_SEC * 1000.0,
```
```python
starts = np.arange(start_time, stop_time, bin_size_sec, dtype=np.float64)
```

iii. Step 5 decisions 5–6: *"Common time base via uniform rebinned trial bins: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions"*, and *"Tentative common bin size = 100 ms: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams."* Step 10 frames the rebinning as *"intentional and required by the target format, not a mismatch in source processing."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations table** (`/intervals/<image_set>_presentations`), using `image_name`, `start_time`, `stop_time` and `omitted` — not from the trials table's `initial_image_name`/`change_image_name`. All interval groups that contain an `image_name` column are concatenated and sorted by start time (this excludes `natural_movie_one_presentations` and `spontaneous_presentations`, which have no `image_name`).

ii.
```python
def read_task_presentations(f: h5py.File) -> Dict[str, np.ndarray]:
    ...
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

iii. Step 5 mapping table: *"Stimulus presentation `image_name` + presentation timing + omission state → `output[0]` (`image_identity`) … Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods."* This is the actual physical stimulus timeline rather than an inferred per-trial label.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is built as `["gray"] + sorted(unique image names across all eligible sessions)` → 17 classes (`gray` + 16 images), with `gray` = index 0. For each trial, every bin is initialized to `gray`; for each presentation overlapping the trial, bins whose **center** falls in `[flash_start, flash_stop)` are set to that image's code. Omitted flashes are skipped, so they stay `gray`. Result: ~67% of all bins are `gray`, the 16 images ~2% each. (A first version mistakenly added `"omitted"` to the vocabulary; Step 10 found and fixed this and re-ran everything.)

ii.
```python
image_series = np.full(centers.shape, image_to_idx["gray"], dtype=np.int16)
for idx in row_idx:
    start = float(presentations["start_time"][idx]); stop = float(presentations["stop_time"][idx])
    in_window = (centers >= start) & (centers < stop)
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

iii. Step 5 decision 8: *"Image identity will include a `gray` class: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin."* Step 10 on the fix: *"`output_values[0]` mistakenly contained `omitted` as an unused image class: fixed by excluding `omitted` from image vocabulary construction"*, after which the verified range is `[0, 16]`, *"consistent with the paper/whitepaper task description of two eight-image sets."*

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on the identical `centers` array used to define the neural bins of the same trial, so it is aligned by construction — bin *k* of the image series and bin *k* of the neural matrix cover the same absolute time interval. A bin is labelled with an image only if its center lies inside the 250 ms flash.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
...
output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                          running_bins, pupil_bins, outcome_series])
```

iii. Step 10 alignment check: *"stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch"*, supported by the `--show-processing` plots that overlay the raw presentation intervals with the rebinned series. I independently confirmed the pattern in `sample_data.pkl`: flashes appear as 2–3 consecutive labelled bins separated by 5 gray bins (250 ms on / 500 ms off), and an omission shows as a 12-bin gray gap.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the presentations table's `is_change` flag (together with `omitted`), not from the trials table's `change_time`/`go`. In these files `is_change` and `is_sham_change` are mutually exclusive, so catch (sham-change) trials get no positive bins — matching the intent of excluding sham changes.

ii.
```python
is_change = bool(np.nan_to_num(presentations["is_change"][idx], nan=0.0))
if is_change and not omitted:
    change_series[in_window] = 1
```

iii. Step 5 mapping: *"Stimulus presentation `is_change` + presentation timing → `output[1]` (`image_change`); Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0"*, with *"trial `change_time` for sanity checks"*.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Just the binary indicator: zeros everywhere, set to 1 on the bins whose centers fall inside the changed image's 250 ms flash. No smoothing, no widening to the following grey period, and nothing for catch trials. Global fraction of positive bins: 2.63%.

ii. See 4-a; the series is allocated alongside the image series in the same function:
```python
change_series = np.zeros(centers.shape, dtype=np.int16)
```

iii. Step 5 decision 9: *"Image-change target will mark the changed-image presentation, not only a single instant: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. Two categories, `output_values[1] = ["no_change", "change"]`. The threshold is purely the presentation-level `is_change` boolean plus the bin-center-inside-flash test; the positive window is one 250 ms flash (2–3 bins of 100 ms).

ii.
```python
"output_values": [image_values, ["no_change", "change"], RUN_BIN_VALUES, PUPIL_BIN_VALUES, TRIAL_OUTCOME_VALUES],
```
```python
in_window = (centers >= start) & (centers < stop)
...
change_series[in_window] = 1
```

iii. Same as 4-b — the window is the changed flash itself, chosen for robustness under binning. The notes acknowledge the resulting sparsity: *"`image_change` positive bins are sparse (`2.63%`), so balanced accuracy modestly above `0.5` is still meaningful."*

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same `centers` grid as the neural data and the image identity series, so change bins coincide exactly with the bins where image identity switches. I verified this in `sample_data.pkl`: in the trial shown below, identity goes `8 → 7` at bins 30–32 and `image_change` is 1 at exactly bins 30–32.

ii.
```python
img [8 8 8 0 0 0 0 0 8 8 0 ... 8 8 0 0 0 0 0 7 7 7 0 0 ...]
chg [0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 1 1 1 0 0 ...]
```
produced by
```python
image_series, change_series = make_image_series(presentations, row_idx, centers, image_to_idx)
```

iii. Step 10: the change series was one of the arrays reconstructed directly from raw NWB and compared with `np.allclose()` on five spot-checked trials, all matching.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `/processing/running/speed/data` with `/processing/running/speed/timestamps` — the SDK's filtered `running_speed` series (cm/s) on the stimulus timebase.

ii.
```python
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
```

iii. Step 1 documents `RunningSpeed.from_stimulus_file` as the SDK producer of this array and notes that it lives on the stimulus timebase, hence the need to resample onto the ophys-aligned grid. The AI used `speed` rather than `speed_unfiltered`, matching the SDK default.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`) of the native ~60 Hz samples onto the trial's 100 ms bin centers, with constant (edge-value) extrapolation outside the recorded range instead of NaN. The result is then discretized (5-c). Non-finite samples are dropped before interpolating, and the code raises if any interpolated value is still non-finite (this never triggered). Note this is a point sample at the bin center, not a within-bin average.

ii.
```python
def interpolate_series(times, values, query_times):
    finite = np.isfinite(times) & np.isfinite(values)
    ...
    t = np.asarray(times[finite], dtype=np.float64); v = np.asarray(values[finite], dtype=np.float64)
    order = np.argsort(t); t = t[order]; v = v[order]
    out = np.interp(query_times, t, v, left=v[0], right=v[-1])
    return out.astype(np.float32)
```
```python
running_interp = interpolate_series(running_times, running_values, centers)
if np.any(~np.isfinite(running_interp)):
    raise ValueError(f"Non-finite running interpolation in experiment {preview.experiment_id}")
```

iii. Step 4: streams are hardware-synced, so *"Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases."* The hard failure on non-finite values is a deliberate fail-loud guard rather than silent imputation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **global** quintile bins. Bin edges are the 20/40/60/80th percentiles of all running samples pooled across all eligible sessions, computed in the preview pass from the *raw* (native-rate) samples that fall inside valid trial windows — not from the interpolated bin values. `np.digitize` then maps each interpolated value to 0–4. Realised distribution: `[0.199, 0.201, 0.199, 0.200, 0.200]`.

ii.
```python
def compute_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    return np.asarray(np.quantile(values, quantiles), dtype=np.float64)

def discretize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges, right=False)
    return np.clip(bins, 0, len(edges)).astype(np.int16)
```
```python
running_pool = np.concatenate([p.running_values for p in eligible if p.running_values.size > 0])
running_pool = running_pool[np.isfinite(running_pool)]
running_edges = compute_bin_edges(running_pool, 5)
```

iii. Step 5 decision 10: *"Running and pupil bin edges will be global, not per-session: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`."* The planned sanity check *"global running and pupil bins should each contain roughly 20% of included samples by construction"* is reported as met in Step 9.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly at the same `centers` used for the neural bins of the same trial, so alignment is exact by construction; only samples within the trial window contribute.

ii.
```python
starts, ends, centers = build_bin_centers(spec.start_time, spec.stop_time, bin_size_sec)
...
running_interp = interpolate_series(running_times, running_values, centers)
running_bins = discretize_with_edges(running_interp, running_edges)
```

iii. Step 10: *"Temporal alignment was rechecked using the processing plots … running interpolation … synchronized on the same ophys-aligned trial window with no visible lag mismatch."* Panel 3 of `processing_<id>.png` overlays the raw running trace, the rebinned trace, and the discrete bin series on a common time axis.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `/acquisition/EyeTracking/pupil_tracking/area_raw` (the pre-blink-filter pupil ellipse area) with timestamps from `/acquisition/EyeTracking/eye_tracking/timestamps`, plus `/acquisition/EyeTracking/likely_blink/data` to mask blinks. Note the reference used `pupil_width` directly; the AI instead converts area to an equivalent diameter.

ii.
```python
pupil_times = np.asarray(f["/acquisition/EyeTracking/eye_tracking/timestamps"], dtype=np.float64)
pupil_area = np.asarray(f["/acquisition/EyeTracking/pupil_tracking/area_raw"], dtype=np.float64)
likely_blink = np.asarray(f["/acquisition/EyeTracking/likely_blink/data"], dtype=np.float64).astype(bool)
pupil_area = pupil_area.copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. Step 3: *"whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter"* — so the AI inverts that relation to recover a diameter. Step 1 notes the SDK's `EyeTrackingTable` "recomputes likely blinks, and filters blink frames", which the AI replicates manually since it bypasses the SDK.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames → NaN; area → diameter via `d = 2*sqrt(area/pi)`; NaNs dropped and the remaining samples linearly interpolated onto the trial bin centers (so blink gaps are bridged rather than left missing); then global quintile discretization. The same fail-loud non-finite guard applies. Because quantile binning is invariant to monotone transforms, the area→diameter step has no effect on the final bin labels.

ii.
```python
def pupil_area_to_diameter(area: np.ndarray) -> np.ndarray:
    area = np.asarray(area, dtype=np.float64)
    out = np.full(area.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(area) & (area >= 0)
    out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)
    return out.astype(np.float32)
```
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
if np.any(~np.isfinite(pupil_interp)):
    raise ValueError(f"Non-finite pupil interpolation in experiment {preview.experiment_id}")
pupil_bins = discretize_with_edges(pupil_interp, pupil_edges)
```

iii. Step 5 mapping: *"Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins."* Blink masking before interpolation is justified as avoiding blink artifacts propagating into neighbouring samples.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 20/40/60/80th percentiles of all blink-masked pupil-diameter samples inside valid trial windows, pooled across all eligible sessions; `np.digitize` → 0–4. Realised distribution `[0.203, 0.206, 0.193, 0.187, 0.212]` — slightly off uniform because the edges come from raw samples while the stored values are interpolated across blink gaps.

ii.
```python
pupil_pool = np.concatenate([p.pupil_values for p in eligible if p.pupil_values.size > 0])
pupil_pool = pupil_pool[np.isfinite(pupil_pool)]
pupil_edges = compute_bin_edges(pupil_pool, 5)
...
"pupil_bin_edges": pupil_edges.tolist(),
```

iii. Same as 5-c (Step 5 decision 10, global edges for a consistent categorical definition). Edges are stored in metadata for recoverability.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated at the trial's `centers`, so it shares indices with the neural matrix.

ii.
```python
pupil_interp = interpolate_series(pupil_times, pupil_diameter, centers)
```

iii. Step 3/4: eye tracking is hardware-synced to the same 100 kHz sync board as the ophys frames, so interpolation onto the ophys-derived grid is valid. Panel 4 of the processing plots visualises raw vs rebinned vs discretized pupil on a shared axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean trial columns `hit`, `miss`, `false_alarm`, `correct_reject`, read once in the preview pass and stored as an integer index on `TrialSpec`.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [hit[idx], miss[idx], false_alarm[idx], correct_reject[idx]]
if sum(int(x) for x in outcome_flags) != 1:
    raise ValueError(f"Trial {idx} does not have exactly one valid outcome")
outcome_idx = outcome_flags.index(True)
```

iii. Step 1 documents that the SDK's `Trial._get_trial_data()` sets these flags and explicitly prevents auto-rewarded trials from receiving one, which is why the AI can assert exactly-one-hot after removing aborted/auto-rewarded trials. Step 5 mapping: *"Trial outcome flags `hit`, `miss`, `false_alarm`, `correct_reject` → `output[4]`."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The index 0–3 is broadcast to a constant series across all bins of the trial so that every output row is time-varying in shape. Missing values are impossible by construction (the exactly-one-hot assertion). Resulting distribution: hit 0.182, miss 0.692, false_alarm 0.010, correct_reject 0.115 (miss-dominated because passive sessions are included).

ii.
```python
outcome_series = np.full(T, spec.outcome_idx, dtype=np.int16)
output_trial = np.vstack([image_series.astype(np.int16), change_series.astype(np.int16),
                          running_bins, pupil_bins, outcome_series])
```

iii. Step 5 decision 11: *"Trial outcome will be repeated across time bins: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form."*

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, split between graceful handling and deliberate hard failure:
- **Missing eye tracking** (3 files): whole session excluded rather than imputing pupil values; recorded in `metadata['excluded_sessions']` with a reason string.
- **Blinks**: set to NaN and bridged by interpolation.
- **Behavioural samples outside the recorded range**: constant edge extrapolation (`left=v[0], right=v[-1]`) rather than NaN, so no artificial "bin 0" is created.
- **NaNs in boolean trial columns**: `np.nan_to_num(..., nan=0.0).astype(bool)` before use.
- **Degenerate sessions**: excluded if <2 valid trials or <2 finite pupil samples; the run aborts if fewer than 2 sessions survive.
- **Degenerate trial windows**: `build_bin_centers` guarantees at least one bin even for a zero-length window.
- **Unexpected states are fatal, not silent**: ambiguous outcome flags and residual non-finite interpolation raise. There is no per-session try/except, so one bad file would abort the whole run.
- **Not handled**: trials are never clipped to the end of the ophys recording. I checked all 284 files and no kept trial's `stop_time` exceeds the last ophys timestamp, so this is harmless on this dataset, but bins past the end would silently become all-zero neural rather than being dropped.
- **Genuinely empty neural trials** (3,947 trials, 4.68%, all-zero event matrices) are kept deliberately.

ii.
```python
aborted = np.nan_to_num(trials["aborted"], nan=0.0).astype(bool)
```
```python
if not has_eye_tracking:
    excluded_reason = "missing_eye_tracking"
elif len(specs) < 2:
    excluded_reason = "fewer_than_2_valid_trials"
elif np.isfinite(pupil_values_all).sum() < 2:
    excluded_reason = "insufficient_valid_pupil_samples"
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

iii. Step 5 decision 4 justifies dropping no-eye-tracking sessions *"rather than fabricating pupil values"*. Step 10 Check 5 (edge cases) reports verifying the 3 exclusions, confirming no session had <2 valid trials, validating a minimum-length trial against raw NWB, and confirming *"all-zero neural warnings arise from genuine zero-event trials in the raw event-detection matrices rather than off-by-one errors at trial boundaries."*

## 9-a. What are the most time-consuming steps of the code?

i. Measured in the full run: preview pass 65.0 s (284 files, ~0.23 s/file), conversion pass 686.0 s (281 sessions, ~2.4 s/session), total 757.7 s. Per-session cost scales with neuron count (1–2 s for ~10-neuron sessions, 6 s for the 591-neuron session), which points at the two real costs: (a) reading the full-session event matrix and behavioural arrays out of HDF5, and (b) the Python-level per-bin loop `for b in range(T)` that slices and sums `event_data`, executed ~25,000 times per session (≈300 trials × ~86 bins), each slice touching all neurons. Interpolation and image-series construction are comparatively cheap.

ii.
```python
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

iii. Step 6: *"Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU."* Step 7 estimated a conservative upper bound of ~28 min for the full run; the actual 12.6 min came in under it, so no further optimisation was done.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- The per-bin neural loop (above) — replaceable by `np.add.reduceat(event_data, frame_starts)` or a cumulative sum differenced at the bin boundaries, which would eliminate the dominant Python overhead.
- `build_trial_specs`'s `for idx in range(raw_count)` loop over already-vectorised boolean arrays; the filtering and one-hot outcome check could be done with array ops (`np.argmax`, `.sum(axis=1)`).
- `make_image_series`'s `for idx in row_idx` loop, which recomputes a full `(centers >= start) & (centers < stop)` boolean mask over the whole trial per flash (~11 flashes/trial); a single `np.searchsorted` of `centers` into the flash boundaries would do it in one pass.
- `collect_time_window_values`'s per-trial `searchsorted`/concatenate loop in the preview pass.

ii.
```python
for idx in range(raw_count):
    if aborted[idx] or auto_rewarded[idx]:
        continue
```
```python
for idx in row_idx:
    ...
    in_window = (centers >= start) & (centers < stop)
```

iii. Only the first of these is acknowledged in the notes (Step 6, quoted in 9-a), framed as a deliberate CPU-for-memory trade. The others are not discussed; since the total runtime landed inside the 15-minute guideline, the AI stopped optimising.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design reads every NWB file twice and repeats several computations:
- `read_trials`, `read_task_presentations`, and the running/pupil arrays (including blink masking and the area→diameter conversion) are executed once in `session_preview` and again in `convert_session`.
- `build_trial_specs` results *are* cached on the `SessionPreview`, so trial parsing is not repeated — but `read_trials` is still re-read in the conversion pass and then used only for the optional plot.
- `plot_processing_summary` recomputes `build_bin_centers`, `make_image_series`, both interpolations and both discretizations for the plotted trial, duplicating work just done in the conversion loop.
- Running/pupil pooling is performed for every file in the preview, including files later excluded.

ii.
```python
# session_preview
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
...
# convert_session — same reads again
running_times = np.asarray(f["/processing/running/speed/timestamps"], dtype=np.float64)
running_values = np.asarray(f["/processing/running/speed/data"], dtype=np.float64)
pupil_area[likely_blink] = np.nan
pupil_diameter = pupil_area_to_diameter(pupil_area)
```

iii. Step 6: *"Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known."* The preview is kept cheap by never touching the event matrix, which is why the repeat costs only 65 s of the 758 s total.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none of them expensive:
- `pupil_area_to_diameter` applies a strictly monotone `sqrt`, and the pupil output is then quantile-binned — so the transform provably cannot change a single stored value. It is presentational only.
- `read_trials` is called in `convert_session` for every session, but `trials` is used only inside `plot_processing_summary`, i.e. for at most 2 sessions.
- `load_neural_events` returns the filtered ROI ids, which the caller discards (`event_data, _ = load_neural_events(f)`).
- `build_bin_centers` returns `starts` and `ends`, which the plotting path ignores.
- `read_task_presentations` decodes and keeps `is_sham_change`, `trials_id` and `active`, none of which are ever read.
- `TrialSpec` stores `change_time` (never used anywhere) and `trial_idx`/`is_catch` (used only for plot selection / diagnostics); the change indicator is derived entirely from the presentations table.
- `session_preview` computes a per-session `unique_images` list when only the global union is needed, and pools running/pupil samples for sessions that are then excluded.
- `--sample` mode reads the whole `ophys_cells_table.csv` just to rank files by cell count.

ii.
```python
event_data, _ = load_neural_events(f)
```
```python
out[valid] = 2.0 * np.sqrt(area[valid] / np.pi)   # monotone; quantile bins unchanged
```
```python
trials = read_trials(f)          # only consumed by plot_processing_summary
presentations = read_task_presentations(f)
```

iii. The notes do not identify any of these; Step 6's efficiency discussion covers only the two-pass I/O and the per-bin summation. The redundant work is minor relative to HDF5 reads and the per-bin loop, which is presumably why it was never flagged.
