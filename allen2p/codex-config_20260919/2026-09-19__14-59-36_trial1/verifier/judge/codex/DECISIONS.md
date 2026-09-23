# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the local Allen release export rather than the AllenSDK cache. It reads `/app/data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`, filters to exact-project `VisualBehavior`, removes passive sessions and three experiments with no eye-tracking data, constructs NWB paths, and opens each selected NWB directly with `h5py`. Inside each NWB it reads the trials table, event-detection data, running-speed stream, eye-tracking stream, and stimulus-presentation table.

ii.
```python
def eligible_experiments(sample: bool) -> pd.DataFrame:
    table = pd.read_csv(EXPERIMENT_TABLE)
    selected = table[
        table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool)
        & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)
    ].copy()
    selected["nwb_path"] = selected["ophys_experiment_id"].map(
        lambda eid: NWB_ROOT / f"behavior_ophys_experiment_{int(eid)}.nwb"
    )
```

```python
with h5py.File(path, "r") as nwb:
    trials = nwb["intervals/trials"]
    event_group = nwb["processing/ophys/event_detection"]
    running = nwb["processing/running/speed"]
    pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
```

iii. In `CONVERSION_NOTES.md`, the AI says the local NWBs are the authoritative release export, that passive replay sessions should be excluded because the paper “was not analyzed” for passive viewing, and that the three eye-absent experiments cannot provide the required pupil output without fabrication. It also says direct NWB access was chosen for speed and memory efficiency while preserving released semantics.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs. For each NWB, the code reads `general/subject/subject_id`, stores that string per converted session, then builds a sorted unique subject list and a `subject_idx` array.

ii.
```python
raw_subject = nwb["general/subject/subject_id"][()]
raw_subject = raw_subject.decode() if isinstance(raw_subject, bytes) else str(raw_subject)
...
session_subjects.append(converted["subject"])
...
subjects = sorted(set(session_subjects), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int32)
```

iii. The notes say the NWB subject ID and metadata `mouse_id` must agree, and that subject IDs should be stored as string mouse IDs with session indices pointing into that sorted global list.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id` is treated as one output session. The code does not group multiple experiments by shared `ophys_session_id`; instead, `ophys_session_id` is kept only as metadata inside each converted session.

ii.
```python
for session_num, (_, row) in enumerate(selected.iterrows(), start=1):
    eid = int(row["ophys_experiment_id"])
    converted, plot_context = convert_session(row, make_plot=make_plot)
```

```python
info = {
    "ophys_experiment_id": eid,
    "ophys_session_id": int(row["ophys_session_id"]),
    "behavior_session_id": int(row["behavior_session_id"]),
    ...
}
```

iii. In the notes, the AI argues that the exact-project active dataset is single-plane, so “each experiment is a target-format session,” and that this avoids inventing cross-plane or cross-day concatenation.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI keeps go and catch trials that are not aborted or auto-rewarded, then segments each trial using the half-open interval `[start_time, stop_time)` on the ophys/event timestamp grid. `np.searchsorted` gives left/right frame indices, and each trial becomes one variable-length neural/output slice.

ii.
```python
trials = nwb["intervals/trials"]
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
aborted = trials["aborted"][:].astype(bool)
auto_rewarded = trials["auto_rewarded"][:].astype(bool)
keep = (go | catch) & ~aborted & ~auto_rewarded
raw_rows = np.flatnonzero(keep)
```

```python
starts = trials["start_time"][:][raw_rows]
stops = trials["stop_time"][:][raw_rows]
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
...
for j, (a, b) in enumerate(zip(left, right)):
    neural = event_block[a - block_left : b - block_left].T.copy()
```

iii. The notes say the AI intentionally preserves native trial bounds and uses half-open slices so adjacent trials do not share boundary frames.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filtering keeps only `(go OR catch) AND NOT aborted AND NOT auto_rewarded`, requires at least two eligible trials in a session, requires exactly one of the four valid outcome flags per retained trial, and raises an error if any aligned trial is empty or if aligned trial slices overlap. At the dataset level, the AI also excludes passive sessions and three sessions with no eye data.

ii.
```python
keep = (go | catch) & ~aborted & ~auto_rewarded
raw_rows = np.flatnonzero(keep)
if len(raw_rows) < 2:
    raise ValueError(f"Experiment {eid} has fewer than two eligible trials")
```

```python
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
    raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
...
if np.any(right <= left):
    raise ValueError(f"Experiment {eid}: empty aligned trial")
if np.any(left[1:] < right[:-1]):
    raise ValueError(f"Experiment {eid}: overlapping eligible trial frame slices")
```

iii. The notes justify these choices by saying passive sessions do not provide contemporaneous behavioral outcomes, eye-absent sessions cannot provide the required pupil labels, and retained trials should have a single well-defined hit/miss/false-alarm/correct-reject outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB event-detection matrix `processing/ophys/event_detection/data`, using `processing/ophys/event_detection/timestamps` as the neural time base. It does not use dF/F.

ii.
```python
event_group = nwb["processing/ophys/event_detection"]
event_data = event_group["data"]
ophys_t = event_group["timestamps"][:]
```

iii. The notes say the paper’s neural analyses and decoders used unfiltered FastLZero L0 calcium event magnitudes, so the conversion should use those event traces on the native ophys grid rather than dF/F.

## 2-b. How is the `neural` data processed?

i. The code reads one contiguous block of the event matrix spanning from the first kept trial frame to the last kept trial frame, casts it to `float32`, then slices each trial and transposes it to `(n_neurons, n_timepoints)`. It does not apply additional normalization, smoothing, or per-neuron filtering.

ii.
```python
block_left, block_right = int(left.min()), int(right.max())
event_block = np.empty(
    (block_right - block_left, event_data.shape[1]), dtype=np.float32
)
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
```

```python
for j, (a, b) in enumerate(zip(left, right)):
    neural = event_block[a - block_left : b - block_left].T.copy()
    neural_trials.append(neural)
```

iii. The notes say the event traces are already the released, QC’d signal of interest, and that the bounding-block read avoids loading unnecessary full-session matrices or doing thousands of tiny HDF5 reads.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional signal-based neuron filtering is applied. The code trusts the released NWB cell table, but it explicitly checks that all stored ROIs are already marked `valid_roi` and that the cell table length matches the number of event columns.

ii.
```python
cell_table = nwb["processing/ophys/image_segmentation/cell_specimen_table"]
if "valid_roi" in cell_table and not np.all(cell_table["valid_roi"][:]):
    raise ValueError(f"Experiment {eid}: unexpected invalid stored ROI")
if len(cell_table["id"]) != event_data.shape[1]:
    raise ValueError(f"Experiment {eid}: cell/event column mismatch")
```

iii. The notes say the released NWB matrices already contain the valid post-QC ROIs, so no ad hoc activity-based neuron filter should be added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start on the ophys clock. For each retained trial, the first included sample is the first event/ophys timestamp `>= start_time`, and the trial ends at the first timestamp `>= stop_time`.

ii.
```python
starts = trials["start_time"][:][raw_rows]
stops = trials["stop_time"][:][raw_rows]
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
```

iii. The metadata string and the notes both say the event trace’s native ophys timestamps are the master clock, and that the trial interval convention is `start_time <= ophys_timestamp < stop_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the native ophys/event sample rate, about 32.31 ms per frame. No temporal rebinning is applied to the neural signal; only behavioral streams are interpolated to those native timestamps.

ii.
```python
median_dt_ms = float(np.median([x["ophys_median_dt_s"] for x in session_info]) * 1000.0)
...
"time_bin_size": median_dt_ms,
```

```python
"behavior_alignment": "linear timestamp interpolation to native ophys frames",
"neural_signal": "unfiltered FastLZero L0 calcium event magnitude",
```

iii. The notes state that the task explicitly requires ophys timestamps as the target grid, and that the local data have a stable single-plane dt of about 32.31 ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table in the NWB `intervals` group, specifically the `image_name`, `start_time`, `stop_time`, `active`, and `omitted` columns. It is not derived from the trial table’s `initial_image_name` and `change_image_name`.

ii.
```python
def find_image_presentations(intervals: h5py.Group) -> h5py.Group:
    candidates = [
        group
        for name, group in intervals.items()
        if name != "trials" and isinstance(group, h5py.Group) and "image_name" in group
    ]
```

```python
starts = group["start_time"][:]
stops = group["stop_time"][:]
names = decode_strings(group["image_name"][:])
active = group["active"][:].astype(bool)
omitted = np.nan_to_num(group["omitted"][:], nan=0.0).astype(bool)
```

iii. The notes say active, non-omitted stimulus presentations are the authoritative source for image identity, and that omitted flashes and gray periods should not be mislabeled as image presentations.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI uses a fixed global label set `["gray", "im000", ..., "im106"]`. For each retained ophys timestamp, it finds the most recent presentation start, checks whether the frame is still before that presentation’s stop time and whether the presentation is active and non-omitted, and assigns the corresponding image code. Frames outside active image presentations are assigned code `0` for `"gray"`.

ii.
```python
IMAGE_VALUES = [
    "gray",
    "im000",
    "im031",
    ...
    "im106",
]
IMAGE_TO_CODE = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
```

```python
idx = np.searchsorted(starts, target_t, side="right") - 1
...
shown = (
    nonnegative
    & np.isfinite(stops[safe_idx])
    & (target_t < stops[safe_idx])
    & active[safe_idx]
    & ~omitted[safe_idx]
    & known_image
)
image = np.zeros(len(target_t), dtype=np.int16)
if np.any(shown):
    image[shown] = np.fromiter(
        (IMAGE_TO_CODE[name] for name in names[safe_idx[shown]]),
        dtype=np.int16,
        count=np.count_nonzero(shown),
    )
```

iii. The notes justify this by saying global coding keeps meanings constant across sessions and that `"gray"` should represent inter-stimulus gray and omitted-flash periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same concatenated `target_t` ophys timestamps used for the neural slices, and each trial receives the matching time slice of that framewise image-code array.

ii.
```python
image_codes, change_codes = presentation_outputs(presentations, target_t)
...
output[0] = image_codes[lo:hi]
```

iii. The notes say the ophys timestamps are the single master clock and that all streams must be aligned by timestamps rather than by matching sample indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table, using the presentation-level `is_change` flag together with the same presentation timing/activity/omission fields used for image identity.

ii.
```python
is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)
...
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. The notes say the changed-image presentation interval is the relevant “right after a change” period and that catch/sham trials should remain zero because there is no true image-identity change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. After locating the active presentation at each retained frame, the code outputs `1` exactly when that active, non-omitted presentation has `is_change=True`; otherwise it outputs `0`. This yields a short positive interval during the changed image’s on-screen flash.

ii.
```python
shown = (
    nonnegative
    & np.isfinite(stops[safe_idx])
    & (target_t < stops[safe_idx])
    & active[safe_idx]
    & ~omitted[safe_idx]
    & known_image
)
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. In the notes, the AI explicitly says it chose the changed presentation interval rather than a one-frame impulse, because it considered that more consistent with the presentation table and the paper’s post-presentation decoder window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no continuous thresholding step. The code uses a direct binary encoding: `0` for `no_change` and `1` for `change`.

ii.
```python
change = (shown & is_change[safe_idx]).astype(np.int16)
...
"output_values": [
    IMAGE_VALUES,
    ["no_change", "change"],
    QUINTILE_NAMES,
    QUINTILE_NAMES,
    OUTCOME_NAMES,
],
```

iii. The notes describe image change as a binary category defined directly from the presentation table, not as a thresholded continuous variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change labels are computed on the same `target_t` frame times as the neural data and then sliced per trial with the same `lo:hi` offsets.

ii.
```python
image_codes, change_codes = presentation_outputs(presentations, target_t)
...
output[1] = change_codes[lo:hi]
```

iii. The notes repeatedly state that all behavioral/stimulus streams are aligned to native ophys timestamps and that clock alignment must be timestamp-based.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB processed running stream `processing/running/speed`, specifically its `timestamps` and `data` datasets.

ii.
```python
running = nwb["processing/running/speed"]
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed"
)
```

iii. The notes say the released processed running-speed stream should be used directly rather than re-deriving speed from the wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates finite running-speed samples to the retained ophys timestamps `target_t`. It then computes quintile codes within each session from the aligned running values over all retained eligible-trial frames.

ii.
```python
def interpolate_finite(
    source_t: np.ndarray, source_x: np.ndarray, target_t: np.ndarray, label: str
) -> np.ndarray:
    good = np.isfinite(source_t) & np.isfinite(source_x)
    ...
    return np.interp(target_t, t, x)
```

```python
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed"
)
running_codes, running_edges = quintile_codes(running_aligned)
```

iii. The notes justify this by saying timestamp interpolation is required because behavior and ophys use different native sample grids, and by arguing that within-session quintiles avoid turning session-specific scale differences into nominally shared decoder categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five categories using the session-specific 0/20/40/60/80/100% quantiles of the aligned running values from retained eligible-trial frames. The categories are integer codes `0` through `4`.

ii.
```python
def quintile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    codes = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
    return codes, edges
```

iii. The notes say the threshold edges are stored in session metadata and that the goal was to obtain five percentile-defined categories in the retained analysis population for each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto the same concatenated `target_t` ophys timestamps used for neural data, then each trial reuses the corresponding `lo:hi` slice when building `output[2]`.

ii.
```python
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed"
)
...
output[2] = running_codes[lo:hi]
```

iii. The notes say “behavior_alignment” is linear timestamp interpolation to native ophys frames, which makes running and neural data share the same frame indices after alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking table `acquisition/EyeTracking/pupil_tracking`, using `area` and `timestamps`. The AI does not use SDK `pupil_width`.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil_area = pupil["area"][:]
pupil_diameter = np.full(pupil_area.shape, np.nan, dtype=np.float64)
nonnegative_area = np.isfinite(pupil_area) & (pupil_area >= 0)
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
```

iii. The notes say the released pupil area is a circularized measure derived from the major axis, so diameter can be recovered exactly as `2*sqrt(area/pi)`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts finite nonnegative pupil area to diameter, linearly interpolates finite diameter samples to `target_t`, and then discretizes those aligned values into within-session quintiles. Sessions with no eye-tracking stream are excluded before conversion.

ii.
```python
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
pupil_aligned = interpolate_finite(
    pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter"
)
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
```

iii. The notes justify this by saying blink/outlier-masked samples should be excluded from the interpolation source, and that sessions with entirely missing eye data cannot provide the required pupil output without fabrication.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five categories using the session-specific 0/20/40/60/80/100% quantiles of the aligned pupil-diameter values from retained eligible-trial frames. The categories are integer codes `0` through `4`.

ii.
```python
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
...
"pupil_quintile_edges_px": pupil_edges.tolist(),
```

iii. The notes say these session-specific edges are stored in metadata and were chosen to produce percentile-balanced classes for each retained session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same `target_t` ophys timestamps used for the neural slices, and each trial uses the corresponding `lo:hi` slice for `output[3]`.

ii.
```python
pupil_aligned = interpolate_finite(
    pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter"
)
...
output[3] = pupil_codes[lo:hi]
```

iii. The notes make the same alignment argument as for running speed: all non-neural streams should be timestamp-interpolated to native ophys frames before trial slicing.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean outcome columns in the trial table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
```

iii. The notes say these four fields are the canonical active-task outcomes once aborted and auto-rewarded trials have been excluded.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. For each retained trial, the code stacks the four boolean outcome columns, requires that exactly one is true, takes `argmax` to obtain a categorical code `0` to `3`, and fills that code across every time bin of the trial in `output[4]`.

ii.
```python
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
    raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
outcomes = np.argmax(outcome_flags[raw_rows], axis=1).astype(np.int16)
```

```python
output[4].fill(outcomes[j])
```

iii. The notes say this exclusivity check prevents ambiguous outcomes, and that repeating the static trial-outcome code across time lets all outputs share a single rectangular `(5, time)` format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is strict rather than permissive. Entire experiments with no eye-tracking data are excluded before conversion. For running and pupil interpolation, only finite source samples are used, and `np.interp` provides interpolation plus endpoint hold instead of leaving NaNs. The code raises hard errors for mouse-ID mismatches, non-monotonic timestamps, empty or overlapping trial slices, non-finite event data, and non-exclusive trial outcomes. It also rejects sessions with fewer than two eligible trials.

ii.
```python
MISSING_EYE_EXPERIMENTS = {795953296, 806456687, 833631914}
...
selected = table[
    table["project_code"].eq("VisualBehavior")
    & ~table["passive"].astype(bool)
    & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)
].copy()
```

```python
good = np.isfinite(source_t) & np.isfinite(source_x)
if np.count_nonzero(good) < 2:
    raise ValueError(f"Insufficient finite {label} samples")
...
return np.interp(target_t, t, x)
```

```python
if np.any(right <= left):
    raise ValueError(f"Experiment {eid}: empty aligned trial")
if np.any(left[1:] < right[:-1]):
    raise ValueError(f"Experiment {eid}: overlapping eligible trial frame slices")
if not np.all(np.isfinite(event_block)):
    raise ValueError(f"Experiment {eid}: event data contain NaN/Inf")
```

iii. The notes say this avoids fabricating required labels, uses the released blink mask implicitly through finite-value filtering, and catches mapping regressions early with explicit assertions and sanity checks.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify disk I/O on large NWB event matrices as the dominant cost, especially reading neural-event blocks and eventually serializing the large pickle. It explicitly optimized against repeated small HDF5 reads and against loading whole full-session float64 matrices.

ii.
```python
event_block = np.empty(
    (block_right - block_left, event_data.shape[1]), dtype=np.float32
)
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
...
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 6 of the notes, the AI says loading full event matrices and reading each trial independently would dominate runtime and memory, so it uses one bounding HDF5 read per session and sequential processing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s decision was to vectorize most session-level work already: timestamp alignment, interpolation, presentation lookup, percentile coding, and outcome construction are done once per session. The remaining obvious non-vectorized loop is the per-trial loop that copies out trial slices and assembles output matrices.

ii.
```python
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
running_aligned = interpolate_finite(...)
running_codes, running_edges = quintile_codes(running_aligned)
pupil_aligned = interpolate_finite(...)
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
image_codes, change_codes = presentation_outputs(presentations, target_t)
```

```python
for j, (a, b) in enumerate(zip(left, right)):
    lo, hi = int(offsets[j]), int(offsets[j + 1])
    neural = event_block[a - block_left : b - block_left].T.copy()
    output = np.empty((5, b - a), dtype=np.int16)
```

iii. The notes say the code intentionally moved work out of the per-trial loop so the loop only does the final slicing/copying that is hard to eliminate cleanly.

## 9-c. What processing does the code repeat multiple times?

i. The code does not repeat raw loading or recompute framewise labels per trial. Within each session it computes running, pupil, image, and change arrays once on the concatenated `target_t`, then reuses those arrays for every trial slice. The same alignment/interpolation/quintile/assertion pipeline is repeated independently for each session.

ii.
```python
running_codes, running_edges = quintile_codes(running_aligned)
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
image_codes, change_codes = presentation_outputs(presentations, target_t)
```

```python
for j, (a, b) in enumerate(zip(left, right)):
    lo, hi = int(offsets[j]), int(offsets[j + 1])
    output[0] = image_codes[lo:hi]
    output[1] = change_codes[lo:hi]
    output[2] = running_codes[lo:hi]
    output[3] = pupil_codes[lo:hi]
```

iii. The notes explicitly say the pipeline avoids repeated per-trial HDF5 reads and repeated per-trial output reconstruction by doing one session-level pass and then reusing the precomputed framewise arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs extra validation and bookkeeping that the downstream decoder does not need: optional plot generation, construction of a large `info`/`session_info` metadata structure, multiple assertions and summary statistics, and one bounding event-block read that may include non-retained gaps between eligible trials before those extra frames are discarded when trial slices are copied.

ii.
```python
if make_plot and j == 0:
    plot_context = {
        "experiment_id": eid,
        "trial_id": int(trial_ids[j]),
        ...
    }
...
if plot_context is not None:
    plot_path = APP_ROOT / f"processing_{eid}.png"
    plot_processing(plot_context, plot_path)
```

```python
info = {
    "ophys_experiment_id": eid,
    "ophys_session_id": int(row["ophys_session_id"]),
    ...
    "running_quintile_edges_cm_s": running_edges.tolist(),
    "pupil_quintile_edges_px": pupil_edges.tolist(),
    "image_on_fraction": float(np.mean(image_codes != 0)),
    "image_change_fraction": float(np.mean(change_codes)),
    ...
}
```

```python
block_left, block_right = int(left.min()), int(right.max())
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
```

iii. The notes describe these as deliberate tradeoffs for sanity checking and I/O efficiency: metadata and plots support validation, while bounding block reads avoid many tiny disk reads even though they include some unused frames.
