# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from NWB files using `h5py`, rather than using the Allen SDK's Python API. It reads a CSV experiment table from the local data directory to discover experiments, constructs NWB file paths, and opens each NWB file to extract neural, behavioral, and trial data. It filters the experiment table to `project_code == 'VisualBehavior'`, excludes passive replay sessions, and excludes 3 experiments with missing eye tracking data.

ii.
```python
table = pd.read_csv(EXPERIMENT_TABLE)
selected = table[
    table["project_code"].eq("VisualBehavior")
    & ~table["passive"].astype(bool)
    & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)
].copy()
selected["nwb_path"] = selected["ophys_experiment_id"].map(
    lambda eid: NWB_ROOT / f"behavior_ophys_experiment_{int(eid)}.nwb"
)
# ...
with h5py.File(path, "r") as nwb:
    # extract all data from NWB
```

iii. The AI chose direct NWB/h5py access because the data is stored locally as NWB files, avoiding the overhead of the Allen SDK's cache management. It documented in CONVERSION_NOTES.md that passive sessions were excluded because they "cannot provide contemporaneous trial outcomes" and the paper states passive viewing "was not analyzed." The 3 eye-absent experiments were excluded because pupil diameter is a required output.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment table CSV. Unique mouse IDs are collected from session subjects and sorted numerically.

ii.
```python
session_subjects.append(converted["subject"])
# ...
subjects = sorted(set(session_subjects), key=int)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int32)
```

iii. The `mouse_id` field is verified against the NWB's internal `subject_id` field within each session conversion to ensure consistency.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane/NWB file) is treated as one session. Since the `VisualBehavior` project uses single-plane imaging, each experiment corresponds to one session. The AI processes 165 sessions (168 active minus 3 without eye data).

ii.
```python
for session_num, (_, row) in enumerate(selected.iterrows(), start=1):
    eid = int(row["ophys_experiment_id"])
    converted, plot_context = convert_session(row, make_plot=make_plot)
    neural_sessions.append(converted["neural"])
```

iii. The AI documented that "Each experiment is a target-format session even when longitudinal cell IDs recur across days. This matches the single-plane project's one experiment per recording session."

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. The AI selects trials where `(go | catch) & ~aborted & ~auto_rewarded`. Each trial spans from `start_time` to `stop_time` using ophys frame indices, producing variable-length trials.

ii.
```python
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
aborted = trials["aborted"][:].astype(bool)
auto_rewarded = trials["auto_rewarded"][:].astype(bool)
keep = (go | catch) & ~aborted & ~auto_rewarded
raw_rows = np.flatnonzero(keep)
# ...
starts = trials["start_time"][:][raw_rows]
stops = trials["stop_time"][:][raw_rows]
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
```

iii. The AI follows the instruction specification to include Go and Catch trials while excluding Aborted and Auto-rewarded. It verifies that each eligible trial has exactly one mutually exclusive outcome.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch, (2) not aborted, (3) not auto-rewarded, (4) must have exactly one valid outcome, (5) must have at least 2 eligible trials per experiment. Additionally, 3 experiments missing eye tracking data are excluded entirely. Passive replay sessions are excluded.

ii.
```python
keep = (go | catch) & ~aborted & ~auto_rewarded
raw_rows = np.flatnonzero(keep)
if len(raw_rows) < 2:
    raise ValueError(f"Experiment {eid} has fewer than two eligible trials")
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
    raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
```

iii. The AI extensively validates trial data integrity, checking for exclusive outcomes and sufficient trial counts. It raises exceptions rather than silently skipping bad data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` in the NWB files — the L0-detected calcium event magnitudes (FastLZero deconvolution), not dF/F traces.

ii.
```python
event_group = nwb["processing/ophys/event_detection"]
event_data = event_group["data"]
ophys_t = event_group["timestamps"][:]
# ...
event_block = np.empty(
    (block_right - block_left, event_data.shape[1]), dtype=np.float32
)
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
```

iii. The AI chose L0 events because "the paper used unfiltered L0-detected calcium event magnitude traces for all neural analysis and decoding." This was documented in CONVERSION_NOTES.md Steps 3 and 4 with specific references to the paper's methodology.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The raw L0 event magnitudes are read directly from the NWB and sliced into trial segments. The data is cast to float32. Each experiment has a single imaging plane, so no merging across planes is needed.

ii.
```python
neural = event_block[a - block_left : b - block_left].T.copy()
neural_trials.append(neural)
```

iii. The AI stated the event data "already incorporates the whitepaper's movie correction, segmentation, demixing, neuropil subtraction, dF/F, and event-detection pipeline."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell filtering is applied. The AI accepts all cells present in the NWB's event_detection matrix, verifying that the stored `valid_roi` flags are all True.

ii.
```python
cell_table = nwb["processing/ophys/image_segmentation/cell_specimen_table"]
if "valid_roi" in cell_table and not np.all(cell_table["valid_roi"][:]):
    raise ValueError(f"Experiment {eid}: unexpected invalid stored ROI")
```

iii. The AI documented: "NWB cell table length equals event columns and every stored valid_roi=True" and "Accept NWB-valid cells; add no second filter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. For each trial, the ophys frames between `start_time` and `stop_time` are identified using `np.searchsorted` with `side="left"`, extracting all frames where `start_time <= ophys_timestamp < stop_time`.

ii.
```python
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
# ...
neural = event_block[a - block_left : b - block_left].T.copy()
```

iii. The AI uses the native ophys frame timestamps as the master temporal grid, as explicitly requested in the instructions ("Temporally align based on ophys timestamp").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Data is kept at the native ophys frame rate (~31 Hz, ~32.31 ms per frame). The time bin size is computed as the median across all sessions' median inter-frame intervals.

ii.
```python
median_dt_ms = float(np.median([x["ophys_median_dt_s"] for x in session_info]) * 1000.0)
```

iii. The AI documented: "Native single-plane dt is effectively common (median 32.31 ms, max session-median deviation 0.01 ms), satisfying common time-bin size while preserving the explicitly requested ophys alignment."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation table in the NWB (`intervals/` group excluding `trials`), using `image_name`, `start_time`, `stop_time`, `active`, `omitted`, and `is_change` fields. This provides per-flash image identity rather than per-trial identity.

ii.
```python
presentations = find_image_presentations(nwb["intervals"])
image_codes, change_codes = presentation_outputs(presentations, target_t)
# ...
def find_image_presentations(intervals: h5py.Group) -> h5py.Group:
    candidates = [
        group
        for name, group in intervals.items()
        if name != "trials" and isinstance(group, h5py.Group) and "image_name" in group
    ]
```

iii. The AI used the stimulus presentation table to get precise per-frame image identity, including distinguishing between image-on periods and gray (inter-stimulus) periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys frame, the AI determines which presentation interval it falls in using `np.searchsorted`. It assigns code 0 ("gray") when no active, non-omitted image is being shown, and the appropriate image code (1-16) when an image is on screen. There are 17 total categories: gray plus 16 natural images.

ii.
```python
idx = np.searchsorted(starts, target_t, side="right") - 1
nonnegative = idx >= 0
safe_idx = np.maximum(idx, 0)
known_image = np.fromiter(
    (name in IMAGE_TO_CODE and name != "gray" for name in names[safe_idx]),
    dtype=bool, count=len(target_t),
)
shown = (
    nonnegative & np.isfinite(stops[safe_idx])
    & (target_t < stops[safe_idx])
    & active[safe_idx] & ~omitted[safe_idx] & known_image
)
image = np.zeros(len(target_t), dtype=np.int16)
if np.any(shown):
    image[shown] = np.fromiter(
        (IMAGE_TO_CODE[name] for name in names[safe_idx[shown]]),
        dtype=np.int16, count=np.count_nonzero(shown),
    )
```

iii. The AI defined a global set of 17 image codes (`IMAGE_VALUES`) including "gray" as index 0. Image identity varies within a trial at the individual flash level (250ms on / 500ms off), so gray periods between flashes are explicitly represented.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed for exactly the same ophys timestamps as the neural data (`target_t`), which is the concatenation of all trial ophys frames. This ensures frame-level alignment.

ii.
```python
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
image_codes, change_codes = presentation_outputs(presentations, target_t)
# ...
output[0] = image_codes[lo:hi]
```

iii. By computing image codes on the same `target_t` array used for neural data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentation table, combined with the presentation `start_time` and `stop_time`.

ii.
```python
is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)
# ...
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. The AI uses the presentation-level `is_change` flag from the NWB, which marks presentations where the image identity changed from the previous one.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each ophys frame, if it falls within a presentation interval that is active, non-omitted, shows a known image, AND has `is_change=True`, then image_change=1. Otherwise image_change=0. The change indicator is 1 for the entire duration of the changed-image flash (~250ms).

ii.
```python
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. The AI documented: "Label every ophys sample falling in the changed image's on-screen interval."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed.

ii. N/A — the variable is naturally binary.

iii. The `is_change` flag is already binary in the source data.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed on the same `target_t` ophys timestamps, then sliced per trial.

ii.
```python
output[1] = change_codes[lo:hi]
```

iii. Same alignment mechanism as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB, which provides the released filtered running speed in cm/s with associated timestamps.

ii.
```python
running = nwb["processing/running/speed"]
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed"
)
```

iii. The AI uses the SDK's released processed (filtered) running speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys trial timestamps using `np.interp` (via the `interpolate_finite` helper which filters out non-finite values first). It is then discretized into 5 quintile bins using within-session percentiles (20/40/60/80th percentiles).

ii.
```python
def interpolate_finite(source_t, source_x, target_t, label):
    good = np.isfinite(source_t) & np.isfinite(source_x)
    t = source_t[good]
    x = source_x[good]
    return np.interp(target_t, t, x)

running_codes, running_edges = quintile_codes(running_aligned)
# ...
def quintile_codes(values):
    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    codes = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
    return codes, edges
```

iii. The AI documented: "Compute within session and only over frames actually retained in eligible trials."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles) computed within each session. Bin edges are at the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the session's retained trial data. The `np.searchsorted` with `side="right"` maps values to bins 0-4.

ii.
```python
edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
codes = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
```

iii. Within-session discretization "yields five percentile-defined categories in the converted analysis population and avoids treating session-specific pupil camera scale as biologically meaningful."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly to the trial ophys timestamps (`target_t`), which is the same timebase as the neural data.

ii.
```python
running_aligned = interpolate_finite(
    running["timestamps"][:], running["data"][:], target_t, "running speed"
)
```

iii. Timestamp-based interpolation to the ophys grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area) in the NWB, converted to diameter via `2*sqrt(area/pi)`.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil_area = pupil["area"][:]
pupil_diameter = np.full(pupil_area.shape, np.nan, dtype=np.float64)
nonnegative_area = np.isfinite(pupil_area) & (pupil_area >= 0)
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
```

iii. The AI documented: "Whitepaper defines pupil diameter as the ellipse major axis and the released blink-masked pupil area as the area of a circle whose diameter is that major axis. Thus diameter can be recovered exactly as `2*sqrt(pupil_area/pi)`."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter. Non-finite and negative area values are set to NaN (blink masking). The diameter is then linearly interpolated to ophys trial timestamps and discretized into 5 within-session quintile bins.

ii.
```python
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
pupil_aligned = interpolate_finite(
    pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter"
)
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
```

iii. The AI noted that "interpolation is necessary because every remaining session has short masked blink/outlier gaps."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — within-session quintile binning using `np.quantile` at 20/40/60/80 percentiles, producing 5 categories (0-4).

ii. Same `quintile_codes` function as running speed.

iii. Within-session quintiles ensure each session has balanced bin counts regardless of absolute pupil size differences across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated directly to trial ophys timestamps.

ii.
```python
pupil_aligned = interpolate_finite(
    pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter"
)
```

iii. Same timestamp-based interpolation approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
    raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
outcomes = np.argmax(outcome_flags[raw_rows], axis=1).astype(np.int16)
```

iii. The AI verifies mutual exclusivity of outcomes before mapping to integer codes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcome flags are stacked into a matrix. `np.argmax` maps the one-hot encoding to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The static per-trial code is then broadcast across all time bins of the trial.

ii.
```python
outcomes = np.argmax(outcome_flags[raw_rows], axis=1).astype(np.int16)
# ...
output[4].fill(outcomes[j])
```

iii. The outcome is static per trial but repeated across all frames to form a time-varying vector in the output array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 experiments with no eye data are excluded entirely.
- **Blink-masked pupil data**: Non-finite/negative pupil area values are set to NaN; only finite values are used for interpolation (via `interpolate_finite`). `np.interp` holds endpoint values for extrapolation.
- **Non-finite running data**: Filtered out before interpolation via the same `interpolate_finite` function.
- **Non-monotonic timestamps**: If timestamps are not sorted, the code sorts and deduplicates them.
- **Session validation**: Experiments with fewer than 2 eligible trials raise an error.
- **Data integrity checks**: The code includes extensive assertions (NaN/Inf checks, frame count mismatches, output code range checks).

ii.
```python
MISSING_EYE_EXPERIMENTS = {795953296, 806456687, 833631914}
# ...
def interpolate_finite(source_t, source_x, target_t, label):
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if np.count_nonzero(good) < 2:
        raise ValueError(f"Insufficient finite {label} samples")
    t = source_t[good]
    x = source_x[good]
    if np.any(np.diff(t) <= 0):
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
        unique = np.concatenate(([True], np.diff(t) > 0))
        t, x = t[unique], x[unique]
    return np.interp(target_t, t, x)
```

iii. The AI chose to exclude experiments with entirely missing eye data rather than fabricating values, and to interpolate through blink gaps rather than assigning a default bin.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading the NWB files via `h5py`, particularly reading the event detection data matrix. The AI reads a contiguous block of frames spanning all trial times, then slices per-trial.

ii.
```python
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
```

iii. I/O from large NWB files is the primary bottleneck. The AI optimizes by reading one contiguous block rather than per-trial reads.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that constructs output arrays could potentially be vectorized, but it is already lightweight since the heavy computation (interpolation, quintile binning) is done vectorized across all trial frames at once before the loop.

ii.
```python
for j, (a, b) in enumerate(zip(left, right)):
    lo, hi = int(offsets[j]), int(offsets[j + 1])
    neural = event_block[a - block_left : b - block_left].T.copy()
    output = np.empty((5, b - a), dtype=np.int16)
    output[0] = image_codes[lo:hi]
    # ...
```

iii. The loop is simple slicing and does not repeat expensive computation.

## 9-c. What processing does the code repeat multiple times?

i. The `interpolate_finite` function is called separately for running speed and pupil diameter — each performs similar finite-value filtering and interpolation. However, these operate on different source data, so the repetition is necessary.

ii. N/A — no unnecessary repetition identified.

iii. The code processes each session independently in a single pass with no redundant computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores extensive per-session metadata in `session_info` (cell specimen IDs, trial IDs, quintile edges, image-on fractions, etc.) that is not directly used by the decoder but serves documentation/debugging purposes. The code also computes and stores `plot_context` for visualization when `--show-processing` is enabled.

ii.
```python
info = {
    "ophys_experiment_id": eid,
    "ophys_session_id": int(row["ophys_session_id"]),
    # ... extensive metadata ...
    "cell_specimen_ids": cell_specimen_ids,
    "running_quintile_edges_cm_s": running_edges.tolist(),
    "pupil_quintile_edges_px": pupil_edges.tolist(),
    "image_on_fraction": float(np.mean(image_codes != 0)),
    "image_change_fraction": float(np.mean(change_codes)),
}
```

iii. This metadata is useful for debugging and reproducibility but adds to pickle file size and is not consumed by the decoder.
