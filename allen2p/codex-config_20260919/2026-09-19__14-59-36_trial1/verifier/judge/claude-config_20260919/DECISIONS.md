# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK cache API. It reads the released local export directly: the project metadata CSV `data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv` is used as the master listing, filtered to exact `project_code == "VisualBehavior"` (239 experiments, 37 mice — the 45 locally present `VisualBehaviorMultiscope` files are deliberately ignored). Each retained row is mapped to its NWB file `behavior_ophys_experiments/behavior_ophys_experiment_<eid>.nwb`, existence of every file is asserted up front, and each file is then opened with `h5py` and read group-by-group (`intervals/trials`, `intervals/<stimulus presentations>`, `processing/ophys/event_detection`, `processing/running/speed`, `acquisition/EyeTracking/pupil_tracking`, `processing/ophys/image_segmentation/cell_specimen_table`, `general/subject/subject_id`). Sessions are processed strictly one at a time in a single sequential pass, sorted by `ophys_experiment_id`.

ii.
```python
DATA_ROOT = APP_ROOT / "data" / "visual-behavior-ophys-1.1.0"
NWB_ROOT = DATA_ROOT / "behavior_ophys_experiments"
EXPERIMENT_TABLE = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"

def eligible_experiments(sample: bool) -> pd.DataFrame:
    table = pd.read_csv(EXPERIMENT_TABLE)
    selected = table[
        table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool)
        & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)
    ].copy()
    selected["nwb_path"] = selected["ophys_experiment_id"].map(
        lambda eid: NWB_ROOT / f"behavior_ophys_experiment_{int(eid)}.nwb")
    missing_files = selected.loc[~selected["nwb_path"].map(Path.exists)]
    if not missing_files.empty:
        raise FileNotFoundError(...)
    selected.sort_values("ophys_experiment_id", inplace=True)
```
```python
with h5py.File(path, "r") as nwb:
    raw_subject = nwb["general/subject/subject_id"][()]
    trials = nwb["intervals/trials"]
    event_group = nwb["processing/ophys/event_detection"]
    running = nwb["processing/running/speed"]
    pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
    presentations = find_image_presentations(nwb["intervals"])
```

iii. From CONVERSION_NOTES Step 1/4/5: "The SDK documentation recommends NWB access through `BehaviorOphysExperiment`; provided data may already be a derived local export". The AI mapped every SDK accessor it would have used onto the underlying NWB dataset ("Same released event, timestamp, running, eye, trial, stimulus, and valid-cell datasets. Direct HDF access avoids loading projections/templates and is value-identical in raw checks") and justified direct access on efficiency grounds — one bounding `read_direct` per session instead of materializing whole-session float64 matrices, ROI masks and image templates. It then closed the risk of diverging from the SDK by running an independent audit script (`cache/review_conversion.py`) that re-derives every stream from raw NWB and compares with `np.allclose`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the `mouse_id` values of the metadata table, kept as strings and sorted numerically (`key=int`). The mouse identity read from the CSV is cross-checked against `general/subject/subject_id` inside each NWB, and a mismatch aborts the conversion. `subject_idx` is the index of each session's mouse into that sorted list. Result: 37 subjects, identical to the reference's 37.

ii.
```python
raw_subject = nwb["general/subject/subject_id"][()]
raw_subject = raw_subject.decode() if isinstance(raw_subject, bytes) else str(raw_subject)
if raw_subject != str(int(row["mouse_id"])):
    raise ValueError(f"Mouse mismatch in experiment {eid}: {raw_subject} vs {row['mouse_id']}")
...
subjects = sorted(set(session_subjects), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int32)
```

iii. Step 5 mapping table: "`/general/subject/subject_id` / metadata `mouse_id` → `subjects`, `subject_idx` … Direct NWB and CSV IDs must agree." The AI treats the metadata table as the discovery mechanism but the NWB as authoritative, so a metadata/file mismatch cannot silently mislabel a session. Step 4 documents that all 37 mice survive the session filters, so no subject is lost.

## 1-c. How are the data split into sessions?

i. One converted session = one `ophys_experiment_id` (one NWB file). The AI verified in Step 2 that the single-plane `VisualBehavior` project has exactly one imaging plane per `ophys_session_id`, so experiment and session are 1:1 and no cross-plane merging is needed. `ophys_session_id` and `behavior_session_id` are nevertheless recorded per session in `metadata['session_info']`. Sessions are ordered by ascending `ophys_experiment_id` (not by acquisition date). Each experiment is kept as an independent session even when the same longitudinal cells recur across days.

ii.
```python
for session_num, (_, row) in enumerate(selected.iterrows(), start=1):
    eid = int(row["ophys_experiment_id"])
    converted, plot_context = convert_session(row, make_plot=make_plot)
    neural_sessions.append(converted["neural"])
    ...
info = {
    "ophys_experiment_id": eid,
    "ophys_session_id": int(row["ophys_session_id"]),
    "behavior_session_id": int(row["behavior_session_id"]),
    ...
}
```

iii. Step 5 Key Decision 11: "Each experiment is a target-format session even when longitudinal cell IDs recur across days. This matches the single-plane project's one experiment per recording session and avoids inventing cross-day time concatenation." Step 2 records "Local NWBs comprise all 239 experiments with exact `project_code == 'VisualBehavior'`", and the experiment/session 1:1 relationship was measured from the metadata.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` dynamic table. Eligible rows are `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. For each eligible row the ophys frames satisfying the **half-open** interval `start_time <= t < stop_time` are taken, via `searchsorted(..., side="left")` on the event-detection timestamps. Trials are therefore variable length (217–389 frames, ~7–12.5 s). The code asserts no trial is empty and that consecutive eligible trials do not share frames. All retained frames are concatenated once per session into `target_t` (with per-trial offsets) so that every derived stream can be computed vectorized and then re-split.

ii.
```python
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
aborted = trials["aborted"][:].astype(bool)
auto_rewarded = trials["auto_rewarded"][:].astype(bool)
keep = (go | catch) & ~aborted & ~auto_rewarded
raw_rows = np.flatnonzero(keep)
if len(raw_rows) < 2:
    raise ValueError(f"Experiment {eid} has fewer than two eligible trials")
...
starts = trials["start_time"][:][raw_rows]
stops = trials["stop_time"][:][raw_rows]
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
if np.any(right <= left):
    raise ValueError(f"Experiment {eid}: empty aligned trial")
if np.any(left[1:] < right[:-1]):
    raise ValueError(f"Experiment {eid}: overlapping eligible trial frame slices")
lengths = right - left
offsets = np.concatenate(([0], np.cumsum(lengths)))
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
```

iii. Step 5 Key Decision 2: "Keep `(go OR catch) AND NOT aborted AND NOT auto_rewarded`; verify exactly one valid outcome. Preserve native SDK trial boundaries and use half-open `[start_time, stop_time)` slices, preventing shared boundary frames." This is a literal implementation of the instruction ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"). Step 4 sanity-checks the resulting cohort against the whitepaper: "Eligible cohort: go 87.464%, catch 12.536%, mean change-start 4.230 s" vs the whitepaper's expected 87.5%/12.5% and ~4.2 s. Step 10 edge-case audit confirmed for *all* trials that the first included frame is `>= start`, the preceding frame is `< start`, the last included frame is `< stop`, and the next frame is `>= stop`.

## 1-e. How are trials filtered based on quality controls?

i. Three layers:
- **Trial level**: `(go|catch) & ~aborted & ~auto_rewarded`, plus a hard assertion that each eligible trial has exactly one of `hit/miss/false_alarm/correct_reject` set.
- **Session level**: passive replay sessions (`OPHYS_2`, `OPHYS_5`; `passive == True`) are dropped — 71 of 239; three further active experiments (795953296, 806456687, 833631914) are dropped because they contain no eye-tracking samples at all and therefore cannot supply the required pupil output; sessions with fewer than two eligible trials raise (none do).
- **Integrity level**: non-monotonic ophys timestamps, empty/overlapping trial slices, non-finite event data, invalid stored ROIs or cell/column count mismatches all raise rather than being silently skipped.

Final cohort: 165 sessions / 37 mice / 28,821 session-neurons / 42,470 trials. (Reference: 236 sessions / 71,242 trials, because it retains passive sessions.)

ii.
```python
selected = table[
    table["project_code"].eq("VisualBehavior")
    & ~table["passive"].astype(bool)
    & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)
].copy()
...
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
    raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
...
if len(raw_rows) < 2:
    raise ValueError(f"Experiment {eid} has fewer than two eligible trials")
```

iii. Step 3 Trial curation rules: "Premature licks abort/restart trials; free/auto-reward trials bias choice." Step 4 resolution on passive sessions: "Paper says passive viewing 'was not analyzed'; behavioral outcomes require active task performance … Keep 168 active sessions initially; exclude three more only because pupil output is entirely absent, leaving 165." Step 2 quantifies why: "Passive replay trials carry labels copied from an earlier active stimulus sequence rather than contemporaneous behavioral outcomes." Step 5 Key Decision 1 notes the explicit cost: "Excluding the three eye-absent experiments removes 276 session-neurons and 917 trials but avoids fabricated required labels."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The unfiltered FastLZero (L0) **detected calcium event magnitude** matrix, `processing/ophys/event_detection/data`, shape `(n_ophys_frames, n_valid_cells)`, with its companion `processing/ophys/event_detection/timestamps`. dF/F is explicitly *not* used, and the SDK's `filtered_events` (causally smoothed, for visualization) is also explicitly not used.

ii.
```python
event_group = nwb["processing/ophys/event_detection"]
event_data = event_group["data"]
ophys_t = event_group["timestamps"][:]
if event_data.shape[0] != len(ophys_t):
    raise ValueError(f"Experiment {eid}: event/timestamp length mismatch")
```
Metadata records: `"neural_signal": "unfiltered FastLZero L0 calcium event magnitude"`.

iii. Step 3: "The paper used **unfiltered L0-detected calcium event magnitude traces** for all neural analysis and decoding". This tracks `methods.txt`: "For all analysis of neural data we used the detected calcium events as described in Garrett et al." and "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f." Step 4 resolution: "Use raw/unfiltered `event_detection/data`, not `filtered_events` visualization and not recomputed dF/F." Step 5 Key Decision 4 adds that events "already incorporate the whitepaper's movie correction, segmentation, demixing, neuropil subtraction, dF/F, and event-detection pipeline."

## 2-b. How is the `neural` data processed?

i. Essentially none. A single bounding HDF5 block `[min(left), max(right))` is read per session directly into a `float32` buffer via `read_direct`; per trial, the rows for that trial are transposed to `(n_neurons, n_timepoints)` and copied. No normalization, smoothing, z-scoring, rebinning or cross-plane merging is applied (single-plane project, so no merging is needed). Finiteness of the whole block is asserted.

ii.
```python
block_left, block_right = int(left.min()), int(right.max())
event_block = np.empty((block_right - block_left, event_data.shape[1]), dtype=np.float32)
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
if not np.all(np.isfinite(event_block)):
    raise ValueError(f"Experiment {eid}: event data contain NaN/Inf")
...
neural = event_block[a - block_left : b - block_left].T.copy()
```

iii. Step 6: "One bounding HDF5 read per session via `read_direct` converts event data straight to float32 … Trial matrices are copied from the in-memory task block, after which excluded raw frames can be released." Step 5 Key Decision 9: "Neural/input float32 … process one session at a time and avoid retaining full raw session matrices after slicing." The underlying scientific position is Step 4's: the released event traces are already the paper's analysis representation, so any further transformation would diverge from the reference processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering is applied. The AI relies entirely on the release's ROI curation, and *verifies* it: it asserts every row of `cell_specimen_table` has `valid_roi == True` and that the cell-table length equals the number of event columns. It also records `cell_specimen_ids` per session for traceability. It explicitly declines to add any signal-based cell filter, and explicitly declines to drop the 1,729 trials (4.07%) whose event matrices are all zero.

ii.
```python
cell_table = nwb["processing/ophys/image_segmentation/cell_specimen_table"]
if "valid_roi" in cell_table and not np.all(cell_table["valid_roi"][:]):
    raise ValueError(f"Experiment {eid}: unexpected invalid stored ROI")
if len(cell_table["id"]) != event_data.shape[1]:
    raise ValueError(f"Experiment {eid}: cell/event column mismatch")
```

iii. Step 3 Neuron curation rules: "NWB dF/F/event matrices already contain the valid released ROIs, so no ad hoc signal-based cell filter should be added", following the SDK's `CellSpecimens(exclude_invalid_rois=True)` default and the whitepaper's ROI classification/demixing QC. Step 10 Check 1 on the all-zero warnings: removing them "would be less faithful than retaining them" because it would require "(a) deleting scientifically valid quiet trials … biasing the dataset toward active neurons, (b) fabricating events, or (c) switching away from the paper's L0 event representation."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instruction is "temporally align based on ophys timestamp". The AI treats `event_detection/timestamps` as the master clock and never equates sample indices across streams. Each trial is the set of ophys frames in `[start_time, stop_time)`, i.e. the alignment event is trial start; `metadata['off_start'] = 0.0`, `off_end = None` (variable length). Every other stream (running, pupil, stimulus) is resampled/looked-up **onto** these frame times, never the reverse.

ii.
```python
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
...
"temporal_alignment_event": (
    "Native ophys frame timestamps within each SDK trial; trials begin "
    "at the first ophys timestamp >= trial start_time."),
"off_start": 0.0,
"off_end": None,
"trial_interval_convention": "start_time <= ophys_timestamp < stop_time",
```

iii. Step 5 Key Decision 3: "The event trace's ophys timestamps are authoritative. No index-based alignment across streams is permitted." Step 3 notes all clocks were hardware-synced on one NI board, so timestamps are directly comparable. The choice of the full native trial window (rather than a fixed window around `change_time`) is what allows image identity/change to be genuinely time-varying. Step 10 Check 5 verified the half-open boundary behaviour on every trial in the cohort.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native ophys frame resolution, **no rebinning, no resampling, no interpolation of the neural data**. Single-plane imaging at ~31 Hz; measured median inter-frame interval 32.31 ms (session medians 32.300–32.320 ms). `metadata['time_bin_size']` is the median over sessions of each session's median `diff(ophys_t)`, in ms. Every session's per-session median dt is also stored in `session_info`.

ii.
```python
if np.any(np.diff(ophys_t) <= 0):
    raise ValueError(f"Experiment {eid}: non-monotonic ophys timestamps")
...
"ophys_median_dt_s": float(np.median(np.diff(ophys_t))),
...
median_dt_ms = float(np.median([x["ophys_median_dt_s"] for x in session_info]) * 1000.0)
data["metadata"]["time_bin_size"] = median_dt_ms
```

iii. Step 3: "Whitepaper single-plane imaging is nominally 31 Hz; the measured NWB interval of 32.31 ms is consistent." Step 4: "Task specifically requests ophys timestamps: keep neural native; interpolate behavior to each selected ophys frame." Step 5 Key Decision 3 addresses the format requirement that bins be the same size everywhere: "Native single-plane dt is effectively common (median 32.31 ms, max session-median deviation 0.01 ms), satisfying common time-bin size while preserving the explicitly requested ophys alignment." The AI notes the paper itself interpolated to a 30 Hz grid but declines to do so because the instruction pins alignment to ophys timestamps.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentations** dynamic table (the one interval table other than `trials` that has an `image_name` column), using `start_time`, `stop_time`, `image_name`, `active`, and `omitted`. It is *not* derived from the trials table's `initial_image_name` / `change_image_name`. The label is per ophys frame: the identity of the image actually on screen at that frame, or an explicit `gray` class when no image is on screen (inter-stimulus gray, omitted flashes, inactive stimulus blocks).

ii.
```python
def find_image_presentations(intervals):
    candidates = [g for name, g in intervals.items()
                  if name != "trials" and isinstance(g, h5py.Group) and "image_name" in g]
    if len(candidates) != 1:
        raise ValueError(...)
    return candidates[0]

def presentation_outputs(group, target_t):
    starts = group["start_time"][:]; stops = group["stop_time"][:]
    names = decode_strings(group["image_name"][:])
    active = group["active"][:].astype(bool)
    omitted = np.nan_to_num(group["omitted"][:], nan=0.0).astype(bool)
    is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)
```

iii. Step 1: "`BehaviorSession.stimulus_presentations` … Provides image identity, exact start/end times, omitted and change flags, activity block, and trial mapping"; and "Stimulus output should be formed from the active change-detection stimulus presentations. Omitted flashes are gray periods and should not be mislabeled as image presentations." Step 5 mapping: "Initialize gray (0); during each active, non-omitted `[start, stop)` interval assign one of 16 globally sorted image codes 1–16 … Gray includes normal inter-stimulus gray and omitted flashes." The AI's position is that this is literally what was on screen at each ophys frame, and it sanity-checked it against the 250 ms-on / 500 ms-off protocol: "about 33% image-on frames as expected from 250 ms on/500 ms off" (measured full-cohort gray fraction 0.669).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A fixed global 17-value code list `['gray', 'im000', ..., 'im106']` (gray = 0, the 16 natural images = 1–16 in sorted order) is hardcoded as `IMAGE_VALUES`/`IMAGE_TO_CODE`. Per session, each target ophys frame is mapped to the last presentation whose `start_time <= t`; the frame is labeled with that presentation's image code only if the frame is also before its `stop_time`, and the presentation is `active`, not `omitted`, and carries a recognized non-gray image name. Everything else stays 0 (gray). The result is asserted to contain only codes in `range(17)`.

ii.
```python
IMAGE_VALUES = ["gray", "im000", "im031", ..., "im106"]
IMAGE_TO_CODE = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
...
idx = np.searchsorted(starts, target_t, side="right") - 1
nonnegative = idx >= 0
safe_idx = np.maximum(idx, 0)
known_image = np.fromiter(
    (name in IMAGE_TO_CODE and name != "gray" for name in names[safe_idx]),
    dtype=bool, count=len(target_t))
shown = (nonnegative & np.isfinite(stops[safe_idx]) & (target_t < stops[safe_idx])
         & active[safe_idx] & ~omitted[safe_idx] & known_image)
image = np.zeros(len(target_t), dtype=np.int16)
if np.any(shown):
    image[shown] = np.fromiter((IMAGE_TO_CODE[name] for name in names[safe_idx[shown]]),
                               dtype=np.int16, count=np.count_nonzero(shown))
...
if set(np.unique(image_codes)) - set(range(len(IMAGE_VALUES))):
    raise AssertionError("Unknown image output code")
```

iii. Step 5 Key Decision 5: "Define global codes `['gray', 'im000', …, 'im106']`. A session uses gray plus its 8-image set; global coding keeps meanings constant across sessions." Step 2 enumerates the 16 image names measured from the data and notes "`omitted` represents gray, not an image identity". The `active` mask exists to exclude the end-of-session fingerprint/movie and spontaneous blocks from being labeled as task images.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identically, by construction: image codes are computed on `target_t`, which *is* the concatenation of the exact ophys frame timestamps used to slice the neural matrix. The per-trial split uses the same `offsets` array, so `output[0]` and `neural` share the same time axis frame-for-frame, with no interpolation or nearest-neighbour approximation of the label.

ii.
```python
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
image_codes, change_codes = presentation_outputs(presentations, target_t)
...
for j, (a, b) in enumerate(zip(left, right)):
    lo, hi = int(offsets[j]), int(offsets[j + 1])
    neural = event_block[a - block_left : b - block_left].T.copy()
    output = np.empty((5, b - a), dtype=np.int16)
    output[0] = image_codes[lo:hi]
...
if sum(x.shape[1] for x in neural_trials) != len(target_t):
    raise AssertionError("Trial frame count mismatch")
```

iii. Step 5 Key Decision 3 ("No index-based alignment across streams is permitted") and the Step 10 comparison table: "timestamp `searchsorted` half-open slicing … Every boundary independently checked." The Step 7 plot review states "images occupy 250-ms flashes separated by 500-ms gray; the change label coincides exactly with the changed image flash", and the Step 10 independent audit reconstructed the image row from raw presentations for 9 trials across 3 sessions with `np.allclose`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of the same stimulus presentations table, intersected with the `shown` mask from 3-b (so a change frame is necessarily a frame on which the changed image is actually on screen). Catch (sham-change) trials are 0 throughout because the SDK sets `is_change` only when the image identity genuinely changed.

ii.
```python
is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)
...
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. Step 1 identifies the SDK definition: "`is_change_event` … Defines change as first non-omitted presentation of a new image (ignores omitted flashes and the session's first stimulus)." Step 5 mapping: "Value 1 for ophys frames during the changed-image presentation interval, else 0 … catch sham changes remain 0 because identity did not change." I verified this empirically on three sessions: the number of trials containing at least one change frame exactly equals the number of go trials, and zero catch trials contain a change frame.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The indicator is 1 for every ophys frame inside the changed image's on-screen presentation interval — i.e. the ~250 ms flash, which is 7–8 ophys frames at 31 Hz — and 0 everywhere else, including the 500 ms gray that follows the change flash. It is not a one-frame impulse and not the whole post-change period. Cohort-wide this yields 2.57% positive frames (vs. the reference's 7.65% under its 750 ms window). Codes are asserted to be a subset of `{0, 1}`.

ii.
```python
change = (shown & is_change[safe_idx]).astype(np.int16)
...
if set(np.unique(change_codes)) - {0, 1}:
    raise AssertionError("Unknown change output code")
...
"image_change_definition": (
    "1 during the on-screen changed-image presentation interval, otherwise 0"),
```

iii. Step 5 Key Decision 6: "Label every ophys sample falling in the changed image's on-screen interval. A one-frame impulse would underrepresent the defined changed presentation and be less consistent with the presentation table/paper's post-presentation decoder window." Step 9 consistency check cross-validates the resulting rate against first principles: "go 87.5%; one changed 250-ms flash/trial → no-change 0.974, change 0.026."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is needed — the variable is natively binary. The two categories are named `['no_change', 'change']` in `output_values[1]`, matching codes 0 and 1.

ii.
```python
"output_names": ["image_identity", "image_change", "running_speed_quintile",
                 "pupil_diameter_quintile", "trial_outcome"],
"output_values": [IMAGE_VALUES, ["no_change", "change"], QUINTILE_NAMES,
                  QUINTILE_NAMES, OUTCOME_NAMES],
```

iii. The instruction defines the variable as binary ("Have value of 1 right after a change in image identity, otherwise 0"), so the only decision was the *width* of the positive window, documented in 4-b. Step 10 edge-case audit additionally confirmed "change implies a non-gray image", i.e. no positive frame lands on a gray frame.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Exactly as image identity: computed on `target_t`, split by the same `offsets`, written into `output[1]` of the same `(5, n_timepoints)` array whose time axis is asserted equal to `neural.shape[1]`.

ii.
```python
image_codes, change_codes = presentation_outputs(presentations, target_t)
...
output[1] = change_codes[lo:hi]
...
if not all(n.shape[1] == i.shape[1] == o.shape[1]
           for n, i, o in zip(neural_trials, input_trials, output_trials)):
    raise AssertionError("Within-trial time dimension mismatch")
```

iii. Same rationale as 3-c. The `--show-processing` plots (`processing_775614751.png`, `processing_788490510.png`) overlay the change step function on the L0 event raster for a retained trial; Step 7 review: "the change label coincides exactly with the changed image flash … No anomalies or temporal offsets were seen."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. The released, already-processed running speed stream `processing/running/speed` (fields `data` in cm/s and `timestamps`), i.e. the wrap-corrected, transient-rejected, low-pass-filtered signal. The unfiltered `speed_unfiltered` variant is explicitly not used, and the speed is not re-derived from the raw encoder voltage.

ii.
```python
running = nwb["processing/running/speed"]
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed")
```

iii. Step 1: "`get_running_df` … Corrects encoder wraps/outliers and applies the released low-pass running-speed filter". Step 4 records a discrepancy the AI chose not to chase: "SDK prose/whitepaper call it 10-Hz low-pass; current code constructs 3rd-order Butterworth with `Wn=4, fs=60` … Resolution: Do not attempt to resolve/reapply legacy cutoff wording; load released `processing/running/speed`." Step 5 notes "negative filter excursions are retained before rank binning", i.e. the small negative values the filter produces are deliberately not clipped.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the ~60 Hz behavior clock onto the retained ophys frame times only (`target_t`, not the whole session), after dropping non-finite samples and de-duplicating/sorting timestamps if they are not strictly increasing. `np.interp` is used, so values outside the behavior recording are clamped to the nearest endpoint rather than becoming NaN. The interpolated trace is then percentile-coded (see 5-c).

ii.
```python
def interpolate_finite(source_t, source_x, target_t, label):
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if np.count_nonzero(good) < 2:
        raise ValueError(f"Insufficient finite {label} samples")
    t = source_t[good]; x = source_x[good]
    if np.any(np.diff(t) <= 0):
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
        unique = np.concatenate(([True], np.diff(t) > 0))
        t, x = t[unique], x[unique]
    return np.interp(target_t, t, x)
```

iii. Step 3/4: behavior is sampled at ~60 Hz on a different grid from the ~31 Hz ophys clock, so "Behavior/eye are nominally 30 Hz and therefore require timestamp-based interpolation rather than sample-index matching." Interpolating onto `target_t` only (rather than all ~145k session frames) is one of the Step 6 documented speed-ups: "Vectorized timestamp alignment, presentation lookup, interpolation, percentile coding, and outcome construction operate once per session."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **within-session** quintiles. For each session independently, the 0/20/40/60/80/100th percentiles of the interpolated running speed over that session's retained eligible-trial frames become the bin edges; each frame is coded 0–4 by `searchsorted` on the four interior edges. The code asserts each session's codes span 0..4, and the six edges are stored per session in `metadata['session_info'][i]['running_quintile_edges_cm_s']`. This differs from the reference, which computes a single set of percentile edges **globally across all sessions**. Both yield ~20% per class overall; the verification log shows exactly 0.200 per class.

ii.
```python
def quintile_codes(values):
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Quintile source must be a nonempty finite 1-D array")
    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    codes = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
    return codes, edges
...
running_codes, running_edges = quintile_codes(running_aligned)
if np.min(running_codes) != 0 or np.max(running_codes) != 4:
    raise AssertionError("Running quintiles do not span 0..4")
```

iii. Step 5 Key Decision 7: "Compute within session and only over frames actually retained in eligible trials. This yields five percentile-defined categories in the converted analysis population and avoids treating session-specific pupil camera scale as biologically meaningful. Edges are stored in metadata." Note that the stated rationale is a pupil-scale argument applied to both behavioral variables; the notes give no running-specific justification, and do not discuss what within-session binning does in sessions where the mouse barely moves.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto `target_t` — the retained ophys frame timestamps — so it shares the neural time axis exactly, then split by the same `offsets` into `output[2]`.

ii.
```python
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed")
running_codes, running_edges = quintile_codes(running_aligned)
...
output[2] = running_codes[lo:hi]
```

iii. Step 3: "All clocks were sampled by one synchronization board; released timestamps are already brought into a common clock", so timestamp interpolation onto ophys frames is valid. Step 7's processing plots show the raw interpolated cm/s trace, the quintile thresholds as horizontal lines, and the resulting step-function code on a shared time axis: "running and pupil traces are smooth after timestamp interpolation; quintile transitions occur at the plotted thresholds."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (plus its `timestamps`) — the released, **blink-masked** pupil ellipse area. Diameter is recovered analytically as `2*sqrt(area/pi)`. `area_raw` is explicitly rejected. The AI derived (and I confirmed in the data) that the released `area` equals `pi * max(width, height)**2` where `width`/`height` are ellipse *radii*, so `2*sqrt(area/pi) == 2*max(width,height)` is exactly the ellipse **major-axis diameter**. In the file I checked, `area` is NaN on 4.99% of samples vs 1.46% for `area_raw`, confirming the blink/outlier mask is already applied to `area`.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil_area = pupil["area"][:]
pupil_diameter = np.full(pupil_area.shape, np.nan, dtype=np.float64)
nonnegative_area = np.isfinite(pupil_area) & (pupil_area >= 0)
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
...
"pupil_definition": (
    "blink-masked major-axis diameter = 2*sqrt(released pupil_area/pi), in pixels"),
```

iii. Step 3: "The whitepaper defines pupil diameter as the ellipse major axis and the released blink-masked pupil area as the area of a circle whose diameter is that major axis. Thus diameter can be recovered exactly as `2*sqrt(pupil_area/pi)`; likely blinks/outlier fits are NaN." Step 4 resolution: "Diameter is `2*sqrt(pupil_area/pi)` (equivalently twice major radius), retaining blink masking before interpolation." Step 5: "Do not use `area_raw`, which would reintroduce invalid blink fits."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Same pipeline as running speed: convert area → major-axis diameter, discard non-finite (blink-masked and negative-area) samples, linearly interpolate the remaining samples onto `target_t` with endpoint clamping, then percentile-code. Blink gaps are therefore bridged by interpolation rather than being labeled as missing. Step 2 measured the blink/mask burden: "all 165 sessions with eye tracking contain blink-masked NaNs (median missing fraction 2.94%, maximum 29.64%)."

ii.
```python
pupil_aligned = interpolate_finite(
    pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter")
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
```

iii. Step 5 mapping: "`diameter=2*sqrt(area/pi)`; discard nonfinite source points, linear timestamp interpolation to ophys frames, then session-specific quintiles across eligible frames … Interpolation is necessary because every remaining session has short masked blink/outlier gaps." Step 5 Key Decision 8: "Use the released blink mask; interpolate only from finite blink-filtered diameter values, including endpoint hold through `np.interp`."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five **within-session** quintiles over the session's retained eligible-trial frames, codes 0–4, edges stored per session in `session_info[i]['pupil_quintile_edges_px']`, span asserted to be 0..4. The reference instead uses one global set of percentile edges across all sessions. Measured distribution: exactly 0.200 per class.

ii.
```python
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
if np.min(pupil_codes) != 0 or np.max(pupil_codes) != 4:
    raise AssertionError("Pupil quintiles do not span 0..4")
...
"quintile_definition": (
    "within-session 20/40/60/80 percentiles over retained eligible-trial ophys frames"),
```

iii. Step 5 Key Decision 7: within-session binning "avoids treating session-specific pupil camera scale as biologically meaningful." Pupil is measured in camera pixels, and camera/eye positioning varies between sessions and mice (the reference's own global edges span 5.3–240 px), so a global pixel threshold partly encodes rig geometry rather than arousal.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated onto `target_t` (the retained ophys frames), split by `offsets` into `output[3]`, sharing the neural time axis exactly.

ii.
```python
pupil_aligned = interpolate_finite(pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter")
...
output[3] = pupil_codes[lo:hi]
```

iii. Step 3 hardware-sync argument, as for running. Step 7 plot review: the raw interpolated diameter, the quintile thresholds, and the resulting code are plotted on the shared trial time axis with no offsets observed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, read for the eligible rows only. Exclusivity (exactly one True per eligible trial) is asserted rather than assumed — there is no `'other'` fallback class.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
    raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
outcomes = np.argmax(outcome_flags[raw_rows], axis=1).astype(np.int16)
```

iii. Step 1: "Provides trial bounds and mutually exclusive go/catch/aborted/auto-rewarded flags plus hit/miss/false-alarm/correct-reject outcomes." Step 4: "All 43,387 eligible rows have exactly one outcome; hit rate 36.43%, FA rate 15.17% … Whitepaper definitions match → Map the four flags directly to outcome classes." Step 3 Trial curation rules: "the task yields go/catch trials and hit/miss/false-alarm/correct-reject outcomes … requiring exactly one of the four valid outcomes."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` over the four flags gives an integer code 0–3 in the fixed order `hit, miss, false_alarm, correct_reject`; the code is then broadcast constant across every time bin of the trial so the output array stays rectangular `(5, n_timepoints)`. Cohort distribution (per frame): hit 0.316, miss 0.559, false alarm 0.018, correct reject 0.107.

ii.
```python
output = np.empty((5, b - a), dtype=np.int16)
...
output[4].fill(outcomes[j])
```
```python
"output_values": [..., OUTCOME_NAMES],
"outcome_trial_counts": {name: int(np.count_nonzero(outcomes == idx))
                         for idx, name in enumerate(OUTCOME_NAMES)},
```

iii. Step 5 mapping: "Map mutually exclusive flag to 0/1/2/3; repeat the static code over the trial's time axis … Repetition permits static and time-varying outputs in one rectangular `(5,time)` array." This follows the instruction's "Trial outcome. Static per-trial." while satisfying the format's preference for time-varying arrays. Step 10 edge-case audit "confirmed all static outcomes are constant" within each trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's posture is **fail loudly for the unexpected, exclude explicitly for the known-unfixable**:
- **Entirely missing eye tracking** (3 active experiments: 795953296, 806456687, 833631914): the experiments are excluded up front by an explicit hardcoded ID set, because a required output could otherwise only be fabricated. Cost documented: 276 session-neurons, 917 trials.
- **Blink-masked / negative-area pupil samples**: dropped from the interpolation source, so the ophys-grid value is a linear bridge across the gap; `np.interp` clamps (holds the endpoint) outside the recorded range rather than producing NaN. Same for non-finite running samples.
- **Non-monotonic / duplicated behavior timestamps**: sorted stably and de-duplicated before interpolation.
- **NaN in `omitted`/`is_change`**: coerced to False via `np.nan_to_num`.
- **Everything else raises**: non-monotonic ophys timestamps, event/timestamp length mismatch, NaN/Inf in event data, empty or overlapping trial slices, non-exclusive outcomes, <2 eligible trials, mouse-ID mismatch between CSV and NWB, invalid stored ROI, cell/event column mismatch, unknown image or change code, quintiles not spanning 0..4, and a final full-cohort totals assertion.
- **All-zero neural trials** (1,729 trials, 4.07%, across 62 sessions) are knowingly retained, after verifying against raw NWB that they are genuinely zero.

ii.
```python
MISSING_EYE_EXPERIMENTS = {795953296, 806456687, 833631914}
...
nonnegative_area = np.isfinite(pupil_area) & (pupil_area >= 0)
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
...
omitted = np.nan_to_num(group["omitted"][:], nan=0.0).astype(bool)
is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)
...
if not sample:
    expected = (165, 42470, 28821, 11192974, 37)
    observed = (len(neural_sessions), total_trials, total_neurons, total_frames, len(subjects))
    if observed != expected:
        raise AssertionError(f"Full-cohort totals changed: observed {observed}, expected {expected}")
```

iii. Step 5 Key Decision 1: excluding the eye-absent experiments "avoids fabricated required labels". Key Decision 8 covers blink handling. Step 10 Check 1 on all-zero trials: "This cannot be 'fixed' without either (a) deleting scientifically valid quiet trials, violating the requested go/catch inclusion and biasing the dataset toward active neurons, (b) fabricating events, or (c) switching away from the paper's L0 event representation. The warnings are therefore legitimate, irreducible diagnostics for sparse detected events." The pervasive assertions are presented in Step 6 as regression guards: "Data-level assertions catch mapping regressions before serialization."

## 9-a. What are the most time-consuming steps of the code?

i. Source I/O dominates. Full conversion of 165 sessions took 66.69 s total: 58.6 s to read/align/slice and 8.09 s to pickle 7.81 GiB. Per session that is ~0.35 s, essentially all of it the bounding `read_direct` of the L0 event block plus the `[:]` reads of running/pupil/presentation/trial columns. The script prints per-session timing, cumulative elapsed, and a rolling ETA on every session, and `session_info` stores `conversion_seconds` per session.

ii.
```python
t0 = time.perf_counter()
...
"conversion_seconds": float(time.perf_counter() - t0),
...
elapsed = time.perf_counter() - conversion_start
mean_s = elapsed / session_num
eta = mean_s * (len(selected) - session_num)
print(f"[{session_num:3d}/{len(selected)}] {eid}: ... session {...:.2f}s, elapsed {elapsed:.1f}s, ETA {eta:.1f}s")
```

iii. Step 6: "Loading complete 140k-frame float64 event matrices would transiently double memory and process spontaneous/movie periods that are never retained. Reading each trial independently would cause tens of thousands of small HDF5 reads." Step 7 estimated 2–4 min for the full run; Step 9 reports the actual 66.69 s, "substantially faster than the conservative 2–4 minute estimate" — comfortably inside the instruction's 15-minute budget.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Only one Python loop over data remains — `for j, (a, b) in enumerate(zip(left, right))` — and it does nothing but slice and copy already-computed arrays into the per-trial lists. Every expensive operation is already vectorized once per session: timestamp alignment (`np.searchsorted` on whole arrays of starts/stops), the presentation lookup for all frames at once, both interpolations, both quintile codings, and the outcome `argmax`. The two remaining `np.fromiter` generator passes inside `presentation_outputs` (`known_image`, and the code lookup on `shown` frames) are per-frame Python iteration and are the only genuinely vectorizable remnant — they could be replaced with a precomputed integer code array indexed by `safe_idx`. The outer `selected.iterrows()` session loop is sequential and could be parallelized, but the AI argued against it.

ii.
```python
known_image = np.fromiter(
    (name in IMAGE_TO_CODE and name != "gray" for name in names[safe_idx]),
    dtype=bool, count=len(target_t))          # per-frame Python iteration
...
for j, (a, b) in enumerate(zip(left, right)):  # slicing/copying only
    lo, hi = int(offsets[j]), int(offsets[j + 1])
    neural = event_block[a - block_left : b - block_left].T.copy()
    output = np.empty((5, b - a), dtype=np.int16)
    output[0] = image_codes[lo:hi]
```

iii. Step 6 Code speedups: "Vectorized timestamp alignment, presentation lookup, interpolation, percentile coding, and outcome construction operate once per session." Against parallelism: "Parallel full-file reads would compete for storage bandwidth and multiply peak memory … Sequential session processing bounds raw-read memory while accumulating only required float32 output." Given a 67 s total runtime, neither remaining loop is worth optimizing.

## 9-c. What processing does the code repeat multiple times?

i. Nothing substantive. Each NWB is opened exactly once, in a single pass; there is no second pass over sessions. Because the quintile edges are computed **within** each session, the code never needs the reference's two-stage structure (hold all trials in memory, compute global edges, then re-walk every trial to discretize) — discretization happens in the same pass as extraction. The only repeated work is trivial: `trials[name][:]` is read per outcome column, and `np.median(np.diff(ophys_t))` is computed per session and then re-medianed at the end.

ii.
```python
for session_num, (_, row) in enumerate(selected.iterrows(), start=1):
    converted, plot_context = convert_session(row, make_plot=make_plot)   # single pass
...
running_codes, running_edges = quintile_codes(running_aligned)   # discretized in-pass
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
```

iii. Step 6: "One bounding HDF5 read per session … Trial matrices are copied from the in-memory task block, after which excluded raw frames can be released." Step 5 Key Decision 9: "process one session at a time and avoid retaining full raw session matrices after slicing" — the single-pass design is a direct consequence of the within-session quintile choice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little, and what there is was a deliberate trade:
- The bounding block read `[min(left), max(right))` pulls in every ophys frame between the first and last eligible trial, including frames belonging to aborted/auto-rewarded trials and inter-trial gaps that are never copied into any output. This is a knowing choice of one large sequential read over tens of thousands of small ones.
- `interpolate_finite` is called on the full running/pupil source arrays (sorting/de-duplication checks over the whole ~150k-sample stream) although only `target_t` is needed.
- Pupil diameter is computed for every eye frame in the session, not only those near retained trials.
- Purely diagnostic quantities are computed and stored per session in `metadata['session_info']`: `image_on_fraction`, `image_change_fraction`, `outcome_trial_counts`, `cell_specimen_ids`, `trial_ids`, `conversion_seconds`, quintile edges.
- The static trial outcome is materialized as a full-length row (`output[4].fill(...)`) rather than stored once per trial — required by the rectangular target format, but redundant information.
- Neural events are stored dense `float32` even though L0 event matrices are overwhelmingly zero (4% of trials are entirely zero), giving a 7.81 GiB pickle where a sparse representation would be far smaller.

ii.
```python
block_left, block_right = int(left.min()), int(right.max())
event_block = np.empty((block_right - block_left, event_data.shape[1]), dtype=np.float32)
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
...
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
...
"image_on_fraction": float(np.mean(image_codes != 0)),
"image_change_fraction": float(np.mean(change_codes)),
"cell_specimen_ids": cell_specimen_ids,
```

iii. Step 6 lists the alternative and why it was rejected: "Reading each trial independently would cause tens of thousands of small HDF5 reads"; "Keeping transposed views would pin bounding source blocks containing excluded trial intervals" — hence the explicit `.T.copy()`. Step 5 mapping: `session_info` "Supports reproducibility and raw spot checks." Step 5 also enumerates the source variables deliberately *not* read at all (dF/F, corrected/demixed/neuropil fluorescence, filtered events, lick/reward times, response latency, eye position/angle, motion correction, projections/ROI masks, natural movie and spontaneous blocks), so the bulk of the NWB is never touched.
