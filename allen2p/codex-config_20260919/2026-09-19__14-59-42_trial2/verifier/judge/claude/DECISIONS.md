# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI **bypasses the AllenSDK object API entirely** and reads the released NWB/HDF5 files directly with `h5py`, using the local release cache at `/app/data/visual-behavior-ophys-1.1.0`. Membership is established from the release metadata CSV `project_metadata/ophys_experiment_table.csv`, intersected with the NWB files actually present on disk (`_file_map()`). Only the exact HDF5 groups needed are read (`processing/ophys/event_detection`, `processing/ophys/image_segmentation/cell_specimen_table`, `intervals/trials`, `intervals/*_presentations`, `processing/running/speed`, `acquisition/EyeTracking`); image templates, ROI masks and projection images are never touched. Files are processed strictly one at a time, and each NWB is opened three times over the run (eye-stream probe, image-vocabulary pass, conversion pass).

ii.
```python
DATA_ROOT = APP_DIR / "data" / "visual-behavior-ophys-1.1.0"
EXPERIMENT_DIR = DATA_ROOT / "behavior_ophys_experiments"
EXPERIMENT_TABLE = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"

EVENT_ROOT = "processing/ophys/event_detection"
CELL_ROOT = "processing/ophys/image_segmentation/cell_specimen_table"
RUN_ROOT = "processing/running/speed"
EYE_ROOT = "acquisition/EyeTracking"
TRIAL_ROOT = "intervals/trials"


def _file_map() -> dict[int, Path]:
    return {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb")
    }


def select_experiments(sample: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Select active, published VisualBehavior experiments with eye data."""
    table = pd.read_csv(EXPERIMENT_TABLE)
    files = _file_map()
    selected = table[
        (table["project_code"] == PROJECT_CODE)
        & (table["behavior_type"] == "active_behavior")
        & (~table["passive"].astype(bool))
        & (table["ophys_experiment_id"].isin(files))
    ].copy()
```
Per-session read (`convert_session`, lines 360-414):
```python
with h5py.File(path, "r") as nwb:
    event_ds = nwb[f"{EVENT_ROOT}/data"]
    ophys_t = np.asarray(nwb[f"{EVENT_ROOT}/timestamps"][:], dtype=np.float64)
    ...
    trials = _trial_table(nwb)
    specs = _trial_specs(trials, ophys_t)
    identity, image_change, n_presentations = _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id)
    run_t = np.asarray(nwb[f"{RUN_ROOT}/timestamps"][:], dtype=np.float64)
    ...
    events = event_ds.astype(np.float32)[:]
```

iii. From CONVERSION_NOTES Step 6/Step 10: "Direct HDF5 reads follow the exact SDK-documented paths while avoiding template/projection data"; "Loading through the full PyNWB/AllenSDK object materializes large image templates and ROI masks that are irrelevant to conversion; the downloaded directory is 247 GB although required time-series datasets are much smaller." The AI defends equivalence with an independent audit script (`cache/raw_sanity_checks.py`) that re-derives membership, trial boundaries and array values from the raw files and compares with `np.allclose`; it reports that all 165 experiment IDs, all 42,470 trial IDs, 37 subjects and 28,821 event channels matched, and that spot-checked neural/behavioral arrays were exactly equal. Full conversion ran in 85.86 s.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiments, sorted numerically (as integers) and stored as strings. `subject_idx` maps each output session to its mouse via a lookup dictionary. 37 mice are retained.

ii.
```python
subjects = sorted(selected["mouse_id"].astype(str).unique(), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[str(mouse)] for mouse in selected["mouse_id"]], dtype=np.int64
),
```

iii. CONVERSION_NOTES Step 5: `mouse_id` gives "stable sorted string mouse IDs and integer lookup per session. 37 subjects in full output." The AI cross-checked 37 mice in the supplied V1.1 metadata against 35 mice in the V1.0 whitepaper Table 1 and documented the difference as a release-version expansion, not an error.

## 1-c. How are the data split into sessions?

i. **One target session = one `VisualBehavior` ophys experiment (NWB file)**. The AI verified in Step 2/Step 4 that the `VisualBehavior` project is single-plane VISp with a strict 1:1 mapping of the 239 experiments to 239 ophys sessions, so no plane-merging is needed. Two session-level curation filters are applied before anything is converted:
- **Passive sessions excluded** (`behavior_type == "active_behavior"` and `~passive`): removes the 71 `OPHYS_2_images_A_passive` / `OPHYS_5_images_B_passive` experiments.
- **Sessions with no eye-tracking stream excluded**: each candidate NWB is probed for `acquisition/EyeTracking/pupil_tracking/area`; experiments 795953296, 806456687 and 833631914 are dropped and recorded in `metadata['excluded_sessions']` with a reason.

Result: 165 sessions from 37 mice (168 active − 3 without eye tracking). Sessions are ordered by `ophys_experiment_id`.

ii.
```python
selected = table[
    (table["project_code"] == PROJECT_CODE)
    & (table["behavior_type"] == "active_behavior")
    & (~table["passive"].astype(bool))
    & (table["ophys_experiment_id"].isin(files))
].copy()
selected = selected.sort_values("ophys_experiment_id")

keep_rows = []
excluded = []
for row in selected.itertuples(index=False):
    experiment_id = int(row.ophys_experiment_id)
    path = files[experiment_id]
    with h5py.File(path, "r") as nwb:
        has_eye = f"{EYE_ROOT}/pupil_tracking/area" in nwb
    if has_eye:
        keep_rows.append(experiment_id)
    else:
        excluded.append({
            "ophys_experiment_id": experiment_id,
            "mouse_id": str(row.mouse_id),
            "reason": "missing eye-tracking stream required for pupil output",
        })
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "`VisualBehavior` is single-plane and experiment/session IDs are one-to-one ... Treat each `VisualBehavior` NWB experiment as one target session." For passive: "Passive rows have nominal go/catch labels but essentially no hits/false alarms (lick spout retracted). Paper says passive viewing 'was not analyzed here' ... Exclude passive sessions; their trial outcomes are not meaningful decoder targets." For eye tracking: "Three sessions with no eye table are excluded" because pupil diameter is a required decoder output and cannot be fabricated. The loss is quantified: "Excluding the three eye-missing sessions removes 276 session-neuron channels and 917 eligible trials but no subject."

## 1-d. How are the data split into trials?

i. Trials come from the SDK-written trials table (`intervals/trials`). A trial is retained if `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Its temporal extent is the **native, variable-length SDK trial window**, taken as the **half-open interval `[start_time, stop_time)`** on the ophys timestamp grid via `np.searchsorted(..., side="left")`. No fixed change-locked window is imposed. Retained trial windows are asserted to be non-overlapping.

ii.
```python
def _trial_specs(trials, ophys_t):
    keep = (
        (trials["go"].astype(bool) | trials["catch"].astype(bool))
        & ~trials["aborted"].astype(bool)
        & ~trials["auto_rewarded"].astype(bool)
    )
    specs = []
    for raw_idx in np.flatnonzero(keep):
        ...
        start = float(trials["start_time"][raw_idx])
        stop = float(trials["stop_time"][raw_idx])
        lo = int(np.searchsorted(ophys_t, start, side="left"))
        hi = int(np.searchsorted(ophys_t, stop, side="left"))
        if hi - lo < 2:
            raise RuntimeError(f"Trial {int(trials['id'][raw_idx])} has only {hi-lo} ophys frames")
        if not (ophys_t[lo] >= start and ophys_t[hi - 1] < stop):
            raise RuntimeError("Half-open trial boundary invariant failed")
```

iii. CONVERSION_NOTES Step 4/Step 5: "Segment each trial at native `[start_time, stop_time)` boundaries; do not force a fixed change-centered window because that would violate experimental trial definitions"; "Half-open indexing prevents double-counting boundary samples." Step 10 reports that all 42,470 trial lengths and half-open boundary inequalities were re-derived from the raw ophys timestamps in an independent audit and matched. Resulting trial durations: median 8.02 s, range 7.02-12.56 s; 217-389 ophys frames.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level curation is exactly the task-mandated set plus structural invariants:
- Keep go and catch; drop aborted and auto-rewarded (the explicit instruction).
- Assert each retained trial has **exactly one** of `hit / miss / false_alarm / correct_reject` set; otherwise abort the run.
- Require ≥2 ophys frames in the trial window (`hi - lo >= 2`), otherwise abort.
- Require ≥2 retained trials per session, otherwise abort that session.
- **No activity-based trial rejection.** 1,729 trials (4.07%) contain no non-zero L0 event in any neuron and trigger validator warnings; the AI verified against raw NWB that these are genuine sparse slices and retained them.
- **No pupil-gap-based trial rejection**, despite a sensitivity analysis showing pupil source gaps up to ~64 s.
- Note that filtering by `(go|catch)` is empirically identical to the human reference's `~aborted & ~auto_rewarded & change_time.notna()` (verified: 145 == 145 trials on experiment 844395446, 0 disagreements).

ii.
```python
outcome_flags = np.array([bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool)
if outcome_flags.sum() != 1:
    raise RuntimeError(f"Trial {int(trials['id'][raw_idx])} has {outcome_flags.sum()} outcome flags")
...
specs = _trial_specs(trials, ophys_t)
if len(specs) < 2:
    raise RuntimeError(f"Experiment {experiment_id} has fewer than two retained trials")
...
used_indices = np.concatenate([np.arange(spec["lo"], spec["hi"], dtype=np.int64) for spec in specs])
if len(np.unique(used_indices)) != len(used_indices):
    raise RuntimeError("Retained trial ophys slices overlap")
```

iii. CONVERSION_NOTES Step 4/Step 10: "Keep go/catch only; outcome classes are hit, miss, false alarm, correct reject"; "Aborted trials result from premature licking before change ... Auto-rewarded/free-reward trials bias choice and are not categorized hit/miss." On the all-zero trials: "They are genuine raw L0-event slices, not conversion omissions ... Removing them would add an unsupported post-publication activity filter and break exact go/catch curation, so they are retained." On pupil gaps: "no paper/SDK rule supports an arbitrary maximum-gap trial rejection; imposing 1 s would remove 1,587 trials and 5 s would remove 281. Retain all eligible trials, record source missingness/max gap in session metadata."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The **released FastLZero (L0) detected calcium events**: `processing/ophys/event_detection/data` (time × ROI) with `.../timestamps` and `.../rois`. dF/F is explicitly *not* used. Cell identity is recovered by indexing `cell_specimen_table/cell_specimen_id` with the event `rois` index.

ii.
```python
EVENT_ROOT = "processing/ophys/event_detection"
...
event_ds = nwb[f"{EVENT_ROOT}/data"]
ophys_t = np.asarray(nwb[f"{EVENT_ROOT}/timestamps"][:], dtype=np.float64)
if event_ds.shape[0] != len(ophys_t):
    raise RuntimeError("Event/timestamp length mismatch")
n_neurons = int(event_ds.shape[1])

cell_ids = np.asarray(nwb[f"{CELL_ROOT}/cell_specimen_id"][:], dtype=np.int64)
valid_rois = np.asarray(nwb[f"{CELL_ROOT}/valid_roi"][:], dtype=bool)
event_rois = np.asarray(nwb[f"{EVENT_ROOT}/rois"][:], dtype=np.int64)
if not valid_rois.all() or len(event_rois) != n_neurons:
    raise RuntimeError("Unexpected invalid/missing event ROI in published NWB")
event_cell_ids = cell_ids[event_rois]
```

iii. The AI's Step 1 notes initially favoured dF/F as "the conservative primary stream pending confirmation", then reversed after reading the methods. Step 3/Step 4: "The paper used released detected calcium events for all neural analyses ... These released events remove slow GCaMP decay and should not be recomputed"; "Use released **raw L0 event magnitude** (`event_detection/data`), not recomputed dF/F and not visualization-smoothed events." This directly tracks `methods.txt` line 208 ("For all analysis of neural data we used the detected calcium events") and line 179 ("We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f").

## 2-b. How is the `neural` data processed?

i. **Essentially no processing.** Events are read straight from HDF5 with an in-read cast to `float32`, checked for finiteness, then sliced per trial and transposed to (n_neurons, n_timepoints) with `np.ascontiguousarray`. No smoothing, no normalization, no z-scoring, no baseline correction, no rebinning. The AI explicitly rejected the SDK's `filtered_events` (causal half-Gaussian, scale 2/31 s) that `Events.from_nwb` can produce. Because there is only one plane per session, no cross-plane merging is needed. `brain_regions` is hard-coded to `["VISp"]` with an all-zero `brain_region_idx` (verified: all 239 `VisualBehavior` experiments target VISp).

ii.
```python
# Read only the scientifically selected neural dataset; HDF5 casts while
# reading, halving peak memory relative to a float64 ndarray conversion.
events = event_ds.astype(np.float32)[:]
if not np.isfinite(events).all():
    raise RuntimeError("Neural event data contain NaN/Inf")
...
for spec in specs:
    lo, hi = spec["lo"], spec["hi"]
    neural = np.ascontiguousarray(events[lo:hi, :].T, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5 decision 2: "Use released raw L0 events, because this is the neural stream used by the supplied paper and tutorials state it avoids contamination from prolonged calcium decay." Step 4: "not visualization-smoothed events". Step 10 reference-comparison table: "Same released arrays and timestamps ... preserves values, as proven by raw `np.allclose`." The AI noted (Step 9/10) that this yields 1,729 all-zero trials and chose to keep them rather than apply an activity filter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No additional filtering.** The AI verified in Step 2 that every stored ROI in the released files already has `valid_roi == True` (invalid ROIs were removed before publication), and asserts this at run time. It also asserts the event ROI count equals the event matrix width, so trace order matches the cell-specimen table. No activity/SNR/event-rate threshold is invented. All 28,821 session-neuron channels are retained.

ii.
```python
if not valid_rois.all() or len(event_rois) != n_neurons:
    raise RuntimeError("Unexpected invalid/missing event ROI in published NWB")
```

iii. CONVERSION_NOTES Step 4 curation notes: "Published valid ROIs exclude unions, duplicates (>70% overlap), motion-border objects, likely dendrites ... SDK default is to retain valid ROIs only. No new activity threshold is described for the paper's decoder, so no post-publication neuron filter should be invented." Step 2 confirmed "The project metadata cell counts exactly match the NWB dF/F channel counts."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: the first ophys frame at or after `start_time`, running up to (but excluding) the first frame at or after `stop_time`. Slice indices come from `np.searchsorted(ophys_t, ..., side="left")` on the event timestamps, which *are* the master clock (no resampling of the neural data ever occurs — all other streams are brought to this grid). The metadata records `temporal_alignment_event = "SDK trial start on the synchronized ophys timestamp grid (first ophys frame at or after start_time)"`, `off_start = 0.0`, `off_end = None` (variable duration). The half-open invariant is asserted for every trial.

ii.
```python
lo = int(np.searchsorted(ophys_t, start, side="left"))
hi = int(np.searchsorted(ophys_t, stop, side="left"))
...
if not (ophys_t[lo] >= start and ophys_t[hi - 1] < stop):
    raise RuntimeError("Half-open trial boundary invariant failed")
...
"temporal_alignment_event": (
    "SDK trial start on the synchronized ophys timestamp grid "
    "(first ophys frame at or after start_time)"
),
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 5 decision 4: "Keep variable-length SDK trial boundaries rather than a fixed change-centered window. Alignment origin is trial start (first ophys frame at/after it); `off_start=0`, `off_end=None` because duration varies." Step 3/4: all streams were recorded on one 100-kHz sync board, so released timestamps are directly comparable and "no inferred clock correction is warranted"; the explicit instruction "Temporally align based on ophys timestamp" makes the ophys frames the master grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning.** Data stay on the native single-plane ophys frame grid, median interval ≈ 32.31 ms (≈30.95 Hz). `metadata['time_bin_size']` is the **median across sessions of each session's median inter-frame interval**, in ms (each session also stores its own `median_ophys_interval_ms`). The plotting mode includes a frame-interval histogram panel as an audit.

ii.
```python
median_bin_ms = float(np.median([x["median_ophys_interval_ms"] for x in session_info]))
...
"time_bin_size": median_bin_ms,
"time_bin_size_units": "ms",
"neural_sampling": "Native single-plane ophys frames, nominally 31 Hz",
```

iii. CONVERSION_NOTES Step 4/5: "Explicit decoder instruction to align on ophys timestamps takes priority: retain native ophys samples and map all outputs to them. Report nominal 32.31-ms bins"; "tiny acquisition-clock variation is retained rather than resampling neural events away from their measured frames." The AI noted the paper interpolates event-triggered traces to a common 30-Hz grid but judged that the explicit "align on ophys timestamp" instruction overrides it.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations interval table**, not from the trials table: for each `*_presentations` group that has both an `image_name` and an `active` column (which excludes `natural_movie_one_presentations` and `spontaneous_presentations`), the AI uses `image_name`, `start_time`, `stop_time` and `active`. The global 16-image vocabulary is assembled in a separate pre-pass over every selected NWB and asserted to be exactly 16 names.

ii.
```python
def _image_presentation_groups(nwb):
    groups = []
    for name, group in nwb["intervals"].items():
        if not isinstance(group, h5py.Group) or not name.endswith("_presentations"):
            continue
        if "image_name" not in group or "active" not in group:
            continue
        active = np.asarray(group["active"][:], dtype=bool)
        if active.any():
            groups.append(group)
    ...

def collect_image_names(selected):
    names: set[str] = set()
    for path in selected["nwb_path"]:
        with h5py.File(path, "r") as nwb:
            for group in _image_presentation_groups(nwb):
                active = np.asarray(group["active"][:], dtype=bool)
                for name in np.asarray(_decode_strings(group["image_name"][:]))[active]:
                    if name and name != "omitted":
                        names.add(str(name))
    if len(names) != 16:
        raise RuntimeError(f"Expected 16 natural-image identities ... found {len(names)}")
    return sorted(names)
```

iii. CONVERSION_NOTES Step 1: "Stimulus presentations are typically 250 ms; omitted presentations are represented explicitly ... Image identity must therefore be constructed from the presentation table on the ophys grid rather than inferred only from per-trial initial/change names." Step 4: "Map the 16 images only during their start/stop intervals and use an explicit `gray` category everywhere else, including omissions."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A session-length integer array is initialized to **0 = "gray"**. For every *active*, non-omitted image presentation, the ophys frames in `[start_time, stop_time)` are set to that image's global code (1-16, lexically sorted image names). The ISI (500 ms gray), omitted flashes, and any non-task period therefore remain class 0. The result is a **17-class** time-varying variable in which gray occupies 66.9% of frames and each image ≈2.0-2.1%. `output_values[0] = ["gray", im000, im031, ...]`.

ii.
```python
def _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id):
    identity = np.zeros(len(ophys_t), dtype=np.int16)  # 0 is gray
    change = np.zeros(len(ophys_t), dtype=np.int16)
    for group in _image_presentation_groups(nwb):
        ...
        for name, start, stop, is_active, changed in zip(names, starts, stops, active, is_change):
            if not is_active:
                continue
            lo = int(np.searchsorted(ophys_t, start, side="left"))
            hi = int(np.searchsorted(ophys_t, stop, side="left"))
            lo, hi = max(0, lo), min(len(ophys_t), hi)
            if hi <= lo or name == "omitted":
                continue
            identity[lo:hi] = image_to_id[name]
```
```python
image_to_id = {name: idx + 1 for idx, name in enumerate(image_names)}
"output_values": [["gray", *image_names], ...]
```

iii. CONVERSION_NOTES Step 5 decision 6: "Use a `gray` class because the output is required at every neural sample and no image is displayed for two-thirds of the cadence. Omitted presentations are gray, not a seventeenth image." Step 3: "Natural images are displayed for 250 ms followed by 500 ms gray. Omitted flashes continue the gray screen." The instruction's phrase "of the image presented during the non-grey screen" is read as licensing an explicit gray label outside the flash.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is built on the **full-session ophys timestamp grid** first, then sliced with the *same* `[lo, hi)` indices as the neural matrix, guaranteeing column-for-column alignment by construction. Presentation boundaries use the same `searchsorted(..., side="left")` convention as trial boundaries. A per-trial shape assertion enforces equal length.

ii.
```python
decoder_output = np.vstack((
    identity[lo:hi],
    image_change[lo:hi],
    running_bin[lo:hi],
    pupil_bin[lo:hi],
    outcome,
)).astype(np.int16, copy=False)
if neural.shape[1] != decoder_output.shape[1]:
    raise RuntimeError("Per-trial temporal dimension mismatch")
```

iii. CONVERSION_NOTES Step 5: "Ophys timestamps define columns ... stimulus intervals use synchronized start/stop searches." Step 7 plot review: "In the go trial, the change row begins exactly at synchronized changed-image onset ... trial slicing and change markers show no offset"; Step 10 independently re-derived identity for spot-checked trials from the raw presentation table and compared with `np.allclose`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the presentation table's **`is_change` flag** (plus `start_time`/`stop_time`/`active` of the same rows). It is *not* derived from the trials table `change_time`/`go` columns. `is_change` is False for catch-trial sham changes (`is_sham_change`), so catch trials automatically get an all-zero change row (verified in the source: 130 `is_change` vs 20 `is_sham_change`, zero overlap).

ii.
```python
is_change = np.asarray(group["is_change"][:], dtype=float)
...
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. CONVERSION_NOTES Step 5 mapping table: "`is_change` ... This is the immediate post-change image window, is false on catch sham changes, and matches the paper's change-image decoding unit better than an unobservable single onset sample." Step 10: "Every catch sham had zero image-change samples and every go trial had a changed-image interval" (checked over all 42,470 trials).

## 4-b. What processing is involved in computing `output` *Image change*?

i. A session-length zero array is set to 1 across the ophys frames of the **changed-image presentation only**, i.e. the 250 ms flash `[start_time, stop_time)`. The following 500 ms gray interval is **not** marked. Resulting time-weighted distribution: `no_change` 0.974, `change` 0.026. (Auto-rewarded trials also carry `is_change` flashes, but their trial windows are excluded from segmentation and the AI asserts retained windows do not overlap, so those frames never enter the output.)

ii.
```python
change = np.zeros(len(ophys_t), dtype=np.int16)
...
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
```

iii. CONVERSION_NOTES Step 5 decision 7: "Mark the 250-ms changed-image interval. A one-frame pulse at display onset precedes much of the cortical response and conflicts with the reference paper's image-interval decoder; the chosen definition remains strictly 'right after' the identity change."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — it is natively binary. The only numeric step is the guard `np.isfinite(changed) and changed > 0.5`, which handles `is_change` being stored as a float with NaNs in the HDF5 table. `output_values[1] = ["no_change", "change"]`, and `validate_converted` asserts the row lies in [0, 1].

ii.
```python
is_change = np.asarray(group["is_change"][:], dtype=float)
...
if np.isfinite(changed) and changed > 0.5:
    change[lo:hi] = 1
...
expected_ranges = ((0, 16), (0, 1), (0, 4), (0, 4), (0, 3))
for row, (low, high) in zip(decoder_output, expected_ranges):
    if row.min() < low or row.max() > high:
        raise RuntimeError("Output value outside declared categories")
```

iii. Not discussed as a "threshold" in the notes; the metadata states `"image_change_encoding": "1 throughout the 250-ms changed-image presentation, 0 otherwise; catch sham changes remain 0"`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identical mechanism to image identity: the change indicator is built on the full-session ophys grid in the same `_stimulus_on_ophys_grid` pass, then sliced with the same `[lo, hi)` indices as the neural matrix.

ii.
```python
identity, image_change, n_presentations = _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id)
...
decoder_output = np.vstack((identity[lo:hi], image_change[lo:hi], ...))
```

iii. Same justification as 3-c. Step 7/Step 12 plot review confirmed "the change row begins exactly at synchronized changed-image onset and remains high only for that image window" and that "catch sham markers do not become changes."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. The **released, already-filtered running speed** in cm/s: `processing/running/speed/data` with `processing/running/speed/timestamps`. The raw/unfiltered encoder channels are deliberately not used and no new filtering is applied.

ii.
```python
RUN_ROOT = "processing/running/speed"
...
run_t = np.asarray(nwb[f"{RUN_ROOT}/timestamps"][:], dtype=np.float64)
run_source = np.asarray(nwb[f"{RUN_ROOT}/data"][:], dtype=np.float64)
running = _interp_finite(ophys_t, run_t, run_source)
```

iii. CONVERSION_NOTES Step 1/Step 4: "Running speed loaded through `.running_speed` is the released filtered signal in cm/s. No new encoder filtering should be applied when loading NWB"; "Do not re-filter; use released `processing/running/speed`, which preserves release processing despite historical wording" (the AI noted and resolved the whitepaper's "10-Hz lowpass" wording vs. the code's 3rd-order Butterworth `Wn=4, fs=60`).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation onto the ophys timestamp grid via a defensive helper that (a) keeps only samples with finite time *and* value, (b) sorts by time, (c) drops duplicate timestamps, (d) errors out if fewer than two usable samples remain. `np.interp` clamps (rather than NaN-fills) outside the source range, so the interpolated trace is finite everywhere. The interpolated trace is then discretized (see 5-c). No smoothing, clipping or absolute-value transform is applied.

ii.
```python
def _interp_finite(target_t, source_t, source_x):
    """Linearly interpolate finite source samples, rejecting unusable streams."""
    valid = np.isfinite(source_t) & np.isfinite(source_x)
    if valid.sum() < 2:
        raise RuntimeError("Fewer than two finite samples in behavioral stream")
    source_t = np.asarray(source_t[valid], dtype=np.float64)
    source_x = np.asarray(source_x[valid], dtype=np.float64)
    order = np.argsort(source_t, kind="stable")
    source_t, source_x = source_t[order], source_x[order]
    unique = np.r_[True, np.diff(source_t) > 0]
    return np.interp(target_t, source_t[unique], source_x[unique]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Linearly interpolate released filtered speed to ophys timestamps"; Step 3 notes the paper itself linearly interpolates behavior to a common grid, so linear interpolation is reference-consistent. Step 7 plot review: "Released running samples and ophys-grid interpolation visually overlap."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. **Per-session** quintiles. The 20/40/60/80th percentiles are computed over **only the samples that will actually be exported** (the union of retained trial index ranges), then `np.digitize(..., right=False)` maps the whole session trace to bins 0-4. The helper errors out if any retained value is non-finite or if the four thresholds are not strictly increasing. Achieved distribution is exactly [.200, .200, .200, .200, .200]. Thresholds are stored per session in `metadata['session_info'][i]['running_quintile_thresholds_cm_per_s']`.

ii.
```python
def _quantile_bins(values, used_indices):
    """Discretize a continuous session stream using retained-sample quintiles."""
    retained = np.asarray(values[used_indices], dtype=np.float64)
    if not np.isfinite(retained).all():
        raise RuntimeError("Non-finite retained continuous values before discretization")
    thresholds = np.percentile(retained, [20, 40, 60, 80])
    if np.any(np.diff(thresholds) <= 0):
        raise RuntimeError(f"Non-distinct quintile thresholds: {thresholds}")
    bins = np.digitize(values, thresholds, right=False).astype(np.int16)
    return bins, thresholds.astype(np.float64)

used_indices = np.concatenate([np.arange(spec["lo"], spec["hi"], dtype=np.int64) for spec in specs])
running_bin, running_thresholds = _quantile_bins(running, used_indices)
```

iii. CONVERSION_NOTES Step 5 decision 8: "Threshold per session using only samples that will be converted. This yields comparable within-session behavioral state quintiles and avoids pooling pupil pixel scales across rigs/animals. Equal values use `np.digitize(..., right=False)`; thresholds and achieved fractions are saved per session." Step 12 explicitly rejected the alternative: "Pooling running/pupil percentiles across rigs would add scale artifacts."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The interpolation target is `ophys_t` itself, so the binned trace is defined on the neural grid for the whole session; the per-trial row is then the same `[lo, hi)` slice used for the neural matrix. Alignment relies on the released hardware-synchronized timestamps (single 100-kHz sync board), with no additional clock correction.

ii.
```python
running = _interp_finite(ophys_t, run_t, run_source)
...
running_bin, running_thresholds = _quantile_bins(running, used_indices)
...
decoder_output = np.vstack((identity[lo:hi], image_change[lo:hi], running_bin[lo:hi], ...))
```

iii. CONVERSION_NOTES Step 3: "Temporal synchronization of all data-streams ... was achieved by recording all experimental clocks on a single NI PCI-6612 digital IO board"; "released synchronized timestamps should be used; no inferred clock correction is warranted." Step 10 independently re-ran `np.interp` from the raw NWB onto selected ophys samples and compared bins with `np.allclose`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The **processed, blink-masked pupil area** `acquisition/EyeTracking/pupil_tracking/area`, with timestamps `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is derived as `2 * sqrt(area / π)`. The SDK stores `pupil_area = π · max(pupil_width, pupil_height)²` with width/height as *semi*-axes, so this expression exactly recovers the major-axis diameter used by the whitepaper. Because the SDK's `filter_on_blinks` already writes NaN into `pupil_area` where `likely_blink` is True, blink rejection is inherited (e.g. 2.6% NaN in experiment 844395446). Sessions with no eye-tracking group at all were removed in `select_experiments`.

ii.
```python
eye_t = np.asarray(nwb[f"{EYE_ROOT}/eye_tracking/timestamps"][:], dtype=np.float64)
pupil_area = np.asarray(nwb[f"{EYE_ROOT}/pupil_tracking/area"][:], dtype=np.float64)
with np.errstate(invalid="ignore"):
    pupil_diameter_source = 2.0 * np.sqrt(pupil_area / np.pi)
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
pupil_missing, pupil_max_gap = _nan_gap_stats(eye_t, pupil_diameter_source)
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "Tutorial labels `pupil_width` as diameter for plotting. Height exceeds width in ~18.2% of valid frames ... Whitepaper defines diameter as the longest pupil ellipse axis and area from that circularized axis. Derive diameter as `2*sqrt(pupil_area/pi)` (equivalent to twice the longest half-axis), using processed/blink-masked area. Factor 2 is immaterial to percentile bins but definition is exact." Step 3: "Missing fits or pupil/eye area z-score >3, plus two adjacent frames on each side, are flagged likely blinks and processed fields are NaN."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical to running speed: area → diameter, then `_interp_finite` drops the blink-NaN samples and linearly interpolates the remaining finite samples onto the ophys grid (so gaps are bridged, not filled with NaN), then per-session quintile discretization. Raw/unprocessed pupil fields are deliberately not used to backfill blinks. Per-session missing fraction and longest contiguous missing gap are computed and stored as provenance.

ii.
```python
def _nan_gap_stats(timestamps, values):
    """Return missing fraction and longest consecutive missing duration."""
    bad = ~np.isfinite(values)
    missing_fraction = float(bad.mean())
    padded = np.r_[False, bad, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if not len(edges):
        return missing_fraction, 0.0
    lengths = edges[1::2] - edges[::2]
    dt = float(np.median(np.diff(timestamps)))
    return missing_fraction, float(lengths.max() * dt)
...
"pupil_source_missing_fraction": pupil_missing,
"pupil_source_longest_missing_gap_s": pupil_max_gap,
```

iii. CONVERSION_NOTES Step 4: "Preserve reference blink rejection, then interpolate processed diameter only to the ophys grid; do not restore raw outlier values." Step 5 decision 9: "Use only processed area, then interpolate across its NaNs. A sensitivity audit found source gaps up to ~64 s, but no paper/SDK rule supports an arbitrary maximum-gap trial rejection ... Retain all eligible trials, record source missingness/max gap in session metadata, and visualize interpolation. This avoids silently changing experimental trial curation."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly the same `_quantile_bins` call as running speed: per-session 20/40/60/80th percentiles computed over retained-trial samples only, `np.digitize(..., right=False)` → bins 0-4, achieved distribution exactly [.200 × 5]. Thresholds stored as `pupil_quintile_thresholds_pixels` per session.

ii.
```python
pupil_bin, pupil_thresholds = _quantile_bins(pupil, used_indices)
...
"pupil_quintile_thresholds_pixels": pupil_thresholds.tolist(),
"behavior_discretization": (
    "Per-session 20/40/60/80 percentiles over retained trial ophys samples"
),
```

iii. CONVERSION_NOTES Step 5 decision 8 (shared with running): per-session thresholds "avoid pooling pupil pixel scales across rigs/animals." Pupil diameter is in camera pixels whose scale depends on the rig/animal/session, so cross-session pooling would mix units of different meaning.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same construction as running speed — interpolated to `ophys_t` for the whole session, binned, then sliced with the shared `[lo, hi)` trial indices.

ii.
```python
pupil = _interp_finite(ophys_t, eye_t, pupil_diameter_source)
...
decoder_output = np.vstack((identity[lo:hi], image_change[lo:hi], running_bin[lo:hi], pupil_bin[lo:hi], outcome))
```

iii. Same hardware-synchronization argument as 5-d; Step 7 plot panel 4 ("Blink-masked pupil + interpolation") was reviewed to confirm the interpolation tracks the finite source samples and "does not restore visible outliers."

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. The code asserts that exactly one is set for every retained trial (no `'other'` fallback exists — a violation aborts the run).

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcome_flags = np.array([bool(trials[name][raw_idx]) for name in OUTCOME_COLUMNS], dtype=bool)
if outcome_flags.sum() != 1:
    raise RuntimeError(f"Trial {int(trials['id'][raw_idx])} has {outcome_flags.sum()} outcome flags")
...
"outcome": int(np.flatnonzero(outcome_flags)[0]),
```

iii. CONVERSION_NOTES Step 4: "SDK makes go/catch/aborted/auto-rewarded mutually exclusive; outcome flags partition go/catch. Active data exactly satisfy hit+miss+FA+CR = go+catch"; "Go comprises hit or miss; catch comprises false alarm or correct rejection."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The one-hot flag is converted to an integer code 0-3 and **broadcast across all `T` time bins** of the trial so that all five outputs share a single `(5, T)` int16 array. `validate_converted` asserts the outcome row has exactly one unique value per trial (i.e. it really is static). `output_values[4] = ["hit", "miss", "false_alarm", "correct_reject"]`. Trial-level distribution: hit .3195, miss .5551, false_alarm .0192, correct_reject .1063.

ii.
```python
outcome = np.full(hi - lo, spec["outcome"], dtype=np.int16)
decoder_output = np.vstack((identity[lo:hi], image_change[lo:hi], running_bin[lo:hi], pupil_bin[lo:hi], outcome)).astype(np.int16, copy=False)
...
if np.unique(decoder_output[4]).size != 1:
    raise RuntimeError("Trial outcome is not static")
...
"outcome_static_per_trial": True,
"outcome_storage": "Static value repeated across time to share a (5,T) output array",
```

iii. CONVERSION_NOTES Step 5 decision 10: "`[hit, miss, false_alarm, correct_reject]`; broadcast static labels across time solely for homogeneous output shape. Trial-level class counts remain separately reported so duration weighting is transparent." Step 12 explicitly declined to change this ("labeling outcome only after the change would make ... outcome easier, but would violate the requested SDK-defined full-trial segmentation and static-per-trial outcome").

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is deliberately **fail-fast** rather than fail-soft, with one class of exception:
- **Missing eye-tracking stream** (3 experiments): handled gracefully — the session is dropped at selection time and recorded, with a reason, in `metadata['excluded_sessions']`.
- **Blink/outlier NaNs in pupil** (mean 3.65%, session range 0.14-29.6%): dropped and bridged by linear interpolation. Gaps of up to ~64 s are interpolated across without rejecting the affected trials; the AI quantified that a 1 s max-gap rule would drop 1,587 trials and a 5 s rule 281, and chose to keep them, storing missing fraction and longest gap per session.
- **Behavioral samples outside the ophys range**: `np.interp` clamps to the nearest endpoint rather than emitting NaN, so no "missing" bin category is ever needed.
- **Duplicate / unsorted / non-finite behavioral timestamps**: filtered and sorted inside `_interp_finite`.
- **All-zero L0 event trials** (1,729 / 42,470 = 4.07%): kept, verified against raw NWB, and documented as a warning explanation rather than "fixed".
- **Everything else raises `RuntimeError`**: event/timestamp length mismatch, invalid ROIs, non-finite neural values, non-finite retained behavioral values, tied quintile thresholds, unknown image names, an image vocabulary ≠ 16, trials with <2 frames, sessions with <2 trials, overlapping trial windows, half-open boundary violations, wrong output shapes/dtypes/ranges, non-static outcome. There is **no `try/except`** anywhere — a single bad session aborts the whole conversion.

ii.
```python
if has_eye: keep_rows.append(experiment_id)
else: excluded.append({... "reason": "missing eye-tracking stream required for pupil output"})
...
valid = np.isfinite(source_t) & np.isfinite(source_x)
if valid.sum() < 2:
    raise RuntimeError("Fewer than two finite samples in behavioral stream")
...
if not np.isfinite(events).all():
    raise RuntimeError("Neural event data contain NaN/Inf")
...
if hi - lo < 2:
    raise RuntimeError(f"Trial {int(trials['id'][raw_idx])} has only {hi-lo} ophys frames")
if len(specs) < 2:
    raise RuntimeError(f"Experiment {experiment_id} has fewer than two retained trials")
```

iii. CONVERSION_NOTES Step 5 decision 9 and Step 10 "Warning Resolution": the AI's stated principle is that no undocumented, invented curation rule should silently remove data — anomalies are either explicitly excluded with a recorded reason (missing eye stream) or retained with quantified provenance (pupil gaps, silent-event trials). Step 10 Check 5 lists the edge cases explicitly tested: "minimum/maximum trial frame counts (217/389), low-neuron sessions (6 cells), genuinely silent event trials, first/last trials, timestamps exactly adjacent to half-open bounds, catch sham changes, omissions-as-gray, static outcome broadcasting, missing pupil samples, and all four outcomes."

## 9-a. What are the most time-consuming steps of the code?

i. Measured: **85.86 s total for 165 sessions**, of which **8.37 s is the pickle write** of a 7.816 GiB file; per-session conversion is printed (e.g. 1.54-1.74 s for the two 426/556-neuron sample sessions). The dominant cost is **HDF5 I/O** — reading the full session event matrix (`events = event_ds.astype(np.float32)[:]`, ~140k frames × up to 666 ROIs) plus the presentation/running/eye tables. Two whole-corpus pre-passes also open every one of the 239 candidate NWBs: the eye-stream probe in `select_experiments` (cheap, metadata only) and `collect_image_names` (reads every `image_name` array, ~4,800 strings per file, decoded one-by-one in Python). Serialization and the final `validate_converted` sweep over all 42,470 trials are the remaining named costs.

ii.
```python
started = time.perf_counter()
with h5py.File(path, "r") as nwb:
    ...
    events = event_ds.astype(np.float32)[:]
...
elapsed = time.perf_counter() - started
print(f"  experiment {experiment_id}: {n_neurons} neurons, {len(specs)} trials, "
      f"{sum(x.shape[1] for x in neural_trials):,} samples in {elapsed:.2f}s", flush=True)
...
write_start = time.perf_counter()
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
write_seconds = time.perf_counter() - write_start
```

iii. CONVERSION_NOTES Step 6: "Loading through the full PyNWB/AllenSDK object materializes large image templates and ROI masks that are irrelevant to conversion"; "Direct HDF5 reads follow the exact SDK-documented paths while avoiding template/projection data"; "HDF5 casts event data to float32 during read"; "Sequential file processing avoids random I/O contention on very large NWBs; timing is printed per session and for pickle serialization." Step 7 estimated <6 min and Step 9 reported the actual 85.86 s, both well inside the 15-minute budget.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most of the hot paths are already vectorized (`np.interp`, `np.percentile`, `np.digitize`, `np.searchsorted`, array slicing). The remaining Python-level loops are:
- **`_stimulus_on_ophys_grid`: a per-presentation loop over ~4,800 rows per session**, doing two scalar `np.searchsorted` calls and two slice assignments each — this is the largest avoidable loop and could be replaced with two vectorized `np.searchsorted` calls plus `np.repeat`/`np.add.reduceat`-style index construction.
- `_decode_strings`: a per-element Python list comprehension decoding every presentation name (called twice per file — once in `collect_image_names`, once in `_stimulus_on_ophys_grid`).
- `_trial_specs`: a per-trial loop (~260 iterations/session) building dicts and doing scalar searchsorted.
- The per-trial loop in `convert_session` that slices/transposes/stacks each trial.
- The per-file loops in `select_experiments` and `collect_image_names`, and `validate_converted`'s nested per-trial sweep.

ii.
```python
for name, start, stop, is_active, changed in zip(names, starts, stops, active, is_change):
    if not is_active:
        continue
    lo = int(np.searchsorted(ophys_t, start, side="left"))
    hi = int(np.searchsorted(ophys_t, stop, side="left"))
    ...
    identity[lo:hi] = image_to_id[name]
```
```python
for raw_idx in np.flatnonzero(keep):
    ...
    lo = int(np.searchsorted(ophys_t, start, side="left"))
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "Timestamp mapping uses `np.searchsorted`; continuous alignment uses vectorized `np.interp`; discretization uses vectorized percentiles/`np.digitize`." The AI identified "Per-sample Python loops would be prohibitive for ~10 million retained timepoints" and avoided those, but did not flag or vectorize the remaining per-presentation loop — presumably because the measured 85.86 s runtime already sat far below the 15-minute optimization threshold in the instructions.

## 9-c. What processing does the code repeat multiple times?

i. **Each NWB is opened three times** across the run: (1) the eye-stream existence probe in `select_experiments`, (2) the image-vocabulary pass in `collect_image_names`, (3) the actual conversion in `convert_session`. Consequently the stimulus-presentation `image_name` array and the `active` mask are read, byte-decoded and scanned **twice** per file, and `_image_presentation_groups` (which reads and reduces the full `active` column for every interval group) is executed twice per file. `np.diff(ophys_t)` is computed twice per session in plot mode. `validate_converted` re-walks every trial after conversion, repeating shape/dtype/range checks already asserted inline in `convert_session`.

ii.
```python
def build_dataset(sample, show_processing):
    selected, excluded = select_experiments(sample=sample)   # pass 1: open every NWB
    image_names = collect_image_names(selected)              # pass 2: open every NWB
    ...
    for session, (_, row) in enumerate(selected.iterrows()):
        converted = convert_session(...)                     # pass 3: open every NWB
```
```python
def collect_image_names(selected):
    for path in selected["nwb_path"]:
        with h5py.File(path, "r") as nwb:
            for group in _image_presentation_groups(nwb):
                active = np.asarray(group["active"][:], dtype=bool)
                for name in np.asarray(_decode_strings(group["image_name"][:]))[active]:
```

iii. The notes do not name this as repetition, but the design intent is visible: a **global, deterministic 16-image vocabulary and a global integer code map must exist before any session is converted**, so that `image_to_id` is identical across sessions and independent of processing order (the human reference instead collects names after the fact and re-codes at assembly time). The eye probe is likewise a pre-pass so that `metadata['excluded_sessions']` and the session count are known up front. CONVERSION_NOTES Step 5 planned sanity check: "Membership: selected experiment IDs equal metadata query (`VisualBehavior`, active, eye-equipped); exact session/subject counts."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items are computed but never consumed by the decoder:
- **Whole-session output construction**: `identity`, `image_change`, `running_bin` and `pupil_bin` are built for all ~140k session frames although only the retained trial frames (~262 × 258 ≈ 68k, roughly half) are exported. Likewise the entire event matrix is read into memory even though only retained frames are sliced out.
- **`_nan_gap_stats`** (missing fraction and longest gap) — pure provenance, never used to filter or alter anything.
- **Metadata payloads**: `cell_specimen_ids`, `trial_ids`, per-session quintile thresholds, `n_active_image_presentations`, `median_ophys_interval_ms`, `outcome_trial_counts`, `conversion_seconds` — useful for auditing, invisible to `train_decoder.py`.
- **Broadcasting the static trial outcome across all T columns**, which stores ~262 identical int16 values per trial instead of one (required by the shared `(5, T)` layout, but redundant information).
- **`np.isfinite(events).all()`** over the entire session event matrix, and the full post-hoc `validate_converted` sweep, duplicating inline assertions.
- **`--show-processing` plotting** (8 panels × 2 sessions) — diagnostic only.
- The `input` field is an empty `(0, T)` array per trial, so `input_names`/`input` carry no information (this one is mandated by the task's "No inputs for this task").

ii.
```python
identity, image_change, n_presentations = _stimulus_on_ophys_grid(nwb, ophys_t, image_to_id)   # full session
running = _interp_finite(ophys_t, run_t, run_source)                                            # full session
pupil_missing, pupil_max_gap = _nan_gap_stats(eye_t, pupil_diameter_source)                     # provenance only
events = event_ds.astype(np.float32)[:]                                                         # full session
if not np.isfinite(events).all():
    raise RuntimeError("Neural event data contain NaN/Inf")
...
outcome = np.full(hi - lo, spec["outcome"], dtype=np.int16)      # static value repeated T times
...
info = {..., "cell_specimen_ids": event_cell_ids, "trial_ids": np.asarray(trial_ids, dtype=np.int64),
        "running_quintile_thresholds_cm_per_s": running_thresholds.tolist(), ...}
```

iii. CONVERSION_NOTES Step 5 decision 11: "Load only needed HDF5 datasets directly, one session at a time; copy retained slices so discarded continuous arrays are released" — the whole-session intermediates are accepted as the price of vectorization and are freed when the file handle closes. Decision 10 justifies the outcome broadcast: "broadcast static labels across time solely for homogeneous output shape." Decision 5 justifies the empty input: "Represent the task's explicit 'No inputs' as `(0,T)` arrays, not placeholders or leakage from requested outputs." The extra metadata exists to support the Step 10 independent audit ("Running/pupil percentiles were recomputed from original retained raw-session samples rather than copied from conversion thresholds").
