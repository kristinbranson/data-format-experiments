# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from NWB files on disk using `h5py`, bypassing the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads the experiment metadata CSV to identify available experiments, then opens each NWB file directly to extract neural, behavioral, and trial data. It filters to `active_behavior` experiments that have eye tracking data and required data streams.

ii.
```python
metadata = pd.read_csv(METADATA_PATH).set_index("ophys_experiment_id", drop=False)
paths = sorted(NWB_ROOT.glob("behavior_ophys_experiment_*.nwb"))
available = {experiment_id_from_path(path): path for path in paths}
# ...
for eid in candidate_ids:
    row = metadata.loc[eid]
    if row["behavior_type"] != "active_behavior":
        excluded.append(...)
        continue
    path = available[eid]
    with h5py.File(path, "r") as nwb:
        if "EyeTracking" not in nwb["acquisition"]:
            excluded.append(...)
            continue
```

iii. The AI chose direct h5py reads to avoid loading unnecessary data (masks, projections, pandas overhead) that the AllenSDK would materialize, improving efficiency. The metadata CSV provides experiment selection criteria. Only active behavior experiments with eye tracking are selected, which is justified by the paper's exclusion of passive viewing and the need for pupil diameter as a decoder output.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV. Each subject is assigned a string ID.

ii.
```python
subjects = sorted({result.subject for result in results}, key=int)
subject_lookup = {name: i for i, name in enumerate(subjects)}
# In process_session:
subject=str(int(meta["mouse_id"]))
```

iii. The `mouse_id` field from the metadata table uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each NWB file (one per imaging plane/experiment) is treated as a separate session. Multi-plane sessions (where multiple imaging planes were recorded simultaneously in one ophys session) are NOT grouped together — each plane becomes its own session.

ii.
```python
for position, (eid, path) in enumerate(selected):
    result = process_session(eid=eid, path=path, meta=metadata.loc[eid], ...)
    results.append(result)
# ...
data = {
    "neural": [result.neural for result in results],
    # ...
}
```

iii. The AI documented: "Use each imaging plane/experiment as a decoder session, consistent with the paper's imaging-plane sampling unit." The CONVERSION_NOTES state the 284 experiment files correspond to 247 unique ophys sessions, with the difference being seven multiscope sessions. The AI treats each imaging plane as a separate session.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. Only go and catch trials are retained (filtering by `go | catch` flags). Aborted and auto-rewarded trials are excluded via an assertion check. Trial windows span from `start_time` to `stop_time` (native variable-length). Trials are then further filtered by pupil data validity (removing trials that overlap long pupil gaps).

ii.
```python
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
aborted = trials["aborted"][:].astype(bool)
auto_rewarded = trials["auto_rewarded"][:].astype(bool)
eligible = go | catch
if np.any(eligible & (aborted | auto_rewarded)):
    raise AssertionError(...)
# ...
kept_indices = candidate_indices[~invalid_trials]
```

iii. The AI uses the `go | catch` boolean flags rather than filtering on `~aborted & ~auto_rewarded`. This is logically equivalent since go/catch and aborted/auto_rewarded are mutually exclusive. The assertion verifies this mutual exclusivity.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) only go and catch trials are retained (excluding aborted and auto-rewarded), and (2) trials that overlap long pupil-invalid intervals (gaps >30 eye-camera frames that cannot be interpolated) are excluded. Sessions with <2 valid trials after filtering would raise an error. Additionally, entire sessions are excluded if they are passive viewing or lack eye tracking data.

ii.
```python
# Pupil gap detection and trial filtering:
pupil_filled, short_runs, residual_runs = fill_short_internal_gaps(
    pupil_raw, eye_t, MAX_PUPIL_GAP_FRAMES
)
invalid_trials = trials_overlapping_invalid_runs(
    candidate_starts, candidate_stops, eye_t, residual_runs
)
kept_indices = candidate_indices[~invalid_trials]
if len(kept_indices) < 2:
    raise RuntimeError(...)
```

iii. The AI justified pupil-based trial filtering: "Gaps >1 s are invalid periods; trials overlapping them are excluded rather than fabricating long stretches." This ensures pupil diameter output values are based on real data, not long interpolations. The reference solution instead maps NaN pupil values to bin 0 without excluding trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` — the detected calcium event magnitudes from FastLZero deconvolution — NOT from dF/F traces.

ii.
```python
ophys_t = nwb["processing/ophys/event_detection/timestamps"][:].astype(np.float64)
event_dataset = nwb["processing/ophys/event_detection/data"]
events = read_float32_dataset(event_dataset)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
```

iii. The AI documented: "Use released raw detected calcium event magnitudes. DFF is less appropriate because the paper explicitly used inferred events; SDK `filtered_events` adds smoothing intended for visualization only." The CONVERSION_NOTES cite the paper's methods describing that detected events remove slow GCaMP6f decay.

## 2-b. How is the `neural` data processed?

i. The raw event magnitudes are linearly interpolated from their native ophys timestamps to a 30 Hz grid (bin centers within each trial). The interpolation is done across all neurons simultaneously using a vectorized matrix interpolation function. Data is stored as float32.

ii.
```python
def interpolate_matrix(source_t, source_values, target_t):
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    result = source_values[left] * (1.0 - weight[:, None]) + source_values[right] * weight[:, None]
    return np.asarray(result, dtype=np.float32)
# ...
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
```

iii. The AI justified 30 Hz interpolation: "the paper linearly interpolated both events and running to 30 Hz" for event-triggered analyses. This creates a common temporal grid across all sessions regardless of native frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only released valid ROIs are included (verified by assertion). No additional activity-based filtering is applied. The AI verifies `valid_roi == True` for all cells. Sessions without eye tracking are excluded (affecting which neural data enters the dataset).

ii.
```python
cell_ids = nwb["processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id"][:]
valid_rois = nwb["processing/ophys/image_segmentation/cell_specimen_table/valid_roi"][:]
if len(cell_ids) != nneurons or not np.all(valid_rois):
    raise AssertionError(f"{eid}: event/cell table mismatch or invalid ROI present")
```

iii. The AI stated: "The whitepaper reports extensive experiment QC; the NWBs already passed release QC." No additional threshold-based filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. A 30 Hz grid of bin centers is generated within each trial's `[start_time, stop_time)` window, and neural event data is linearly interpolated to these bin centers. The first bin center is at `start_time + 0.5/30` seconds.

ii.
```python
def trial_centers(start: float, stop: float) -> np.ndarray:
    n_bins = int(np.floor((stop - start) * TARGET_HZ + 1.0e-9))
    if n_bins < 1:
        return np.empty(0, dtype=np.float64)
    return start + (np.arange(n_bins, dtype=np.float64) + 0.5) * BIN_SIZE_S

per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
target_t = np.concatenate(per_trial_t)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
```

iii. The AI used bin centers (offset by half a bin) rather than bin edges, ensuring all target times fall within the source data range. The trial start on the ophys clock serves as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to 30 Hz (33.333 ms bins), regardless of the native ophys frame rate. The native rate varies across sessions (~31 Hz for single-plane, ~11 Hz for multi-plane). Linear interpolation is used to resample to the common 30 Hz grid.

ii.
```python
TARGET_HZ = 30.0
BIN_SIZE_S = 1.0 / TARGET_HZ
# ...
"time_bin_size": 1000.0 / TARGET_HZ,  # 33.333 ms
```

iii. The AI justified: "target bins must match across sessions and the paper linearly interpolated both events and running to 30 Hz." This ensures consistent temporal resolution across sessions with different native frame rates. The paper's event-triggered analyses used 30 Hz interpolation.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation table in the NWB file (`intervals/<presentation_table>`), specifically the `image_name`, `start_time`, `stop_time`, `active`, and `omitted` fields. This is different from the trial-level `initial_image_name`/`change_image_name` approach.

ii.
```python
presentations = presentation_group(nwb)
presentation_starts = presentations["start_time"][:].astype(np.float64)
presentation_stops = presentations["stop_time"][:].astype(np.float64)
active = presentations["active"][:].astype(bool)
omitted = np.nan_to_num(presentations["omitted"][:], nan=0.0).astype(bool)
image_names = decode_strings(presentations["image_name"][:])
```

iii. The AI used the presentation table to get precise timing of when each image is physically on screen, including inter-stimulus gray periods and omitted presentations. This provides finer temporal resolution than the trial-level image fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 30 Hz bin center, the code finds which presentation interval it falls within. If the bin falls during an active, non-omitted presentation, it gets that image's code. Otherwise, it is labeled as "gray" (code 0). The 17 global image categories include gray plus 16 named images.

ii.
```python
IMAGE_VALUES = ["gray", "im000", "im031", ..., "im106"]
IMAGE_TO_CODE = {name: i for i, name in enumerate(IMAGE_VALUES)}
# ...
image_codes = np.zeros(len(target_t), dtype=np.int8)  # default gray
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
valid_pidx = pidx >= 0
shown = np.zeros(len(target_t), dtype=bool)
shown[valid_pidx] = (
    active[pidx[valid_pidx]]
    & ~omitted[pidx[valid_pidx]]
    & (target_t[valid_pidx] < presentation_stops[pidx[valid_pidx]])
)
image_codes[shown_indices] = np.asarray(
    [IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]], dtype=np.int8
)
```

iii. The AI included "gray" as a class to represent inter-stimulus intervals and omitted presentations, reasoning that during these periods no image is physically on screen. The 17 classes (gray + 16 images) provide a complete representation of the visual stimulus at each timepoint.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30 Hz bin centers as the neural data, using presentation timing from the synchronized ophys clock. Both use the same `target_t` array.

ii.
```python
# Same target_t used for neural and image identity
target_t = np.concatenate(per_trial_t)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
# ...
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
```

iii. Using the same time grid ensures perfect alignment between neural activity and image identity labels.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the trial `change_time` and `go` flag from the NWB trials table. Only true image changes (go trials) are marked; catch/sham changes are not.

ii.
```python
change_times_all = trials["change_time"][:].astype(np.float64)
for trial_position, raw_index in enumerate(kept_indices):
    if not go[raw_index]:
        continue
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
    change_codes[lo + local_index] = 1
```

iii. The AI marks change only for go trials where a true image identity change occurs. Catch trials have sham changes (same image repeated) so image change is 0.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary variable. For each go trial, the single 30 Hz bin at or immediately after `change_time` is set to 1; all other bins are 0. This is a single-bin pulse rather than a sustained window.

ii.
```python
change_codes = np.zeros(len(target_t), dtype=np.int8)
# ... (only one bin set to 1 per go trial)
change_codes[lo + local_index] = 1
```

iii. The AI chose a single-bin pulse because "image change" marks the instantaneous moment of change. Cross-validation is performed against the presentation table's `is_change` annotations.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii. See 4-b above.

iii. The instruction specifies "binary variable" so no discretization is required beyond the binary coding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same 30 Hz bin centers as the neural data. The change onset is found using `np.searchsorted` on the trial's portion of `target_t`.

ii. See 4-a code snippet.

iii. Same alignment approach as all other output variables — shared `target_t` time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file — the processed (filtered) running speed in cm/s.

ii.
```python
running_t = nwb["processing/running/speed/timestamps"][:].astype(np.float64)
running = nwb["processing/running/speed/data"][:].astype(np.float64)
```

iii. The AI uses the already-processed running speed from the NWB, which has been through the SDK's low-pass Butterworth filter.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz bin centers, then discretized into 5 quintile bins. Quintile edges are computed **per-session** (not globally), using only retained trial timepoints. NaN handling is strict — nonfinite values raise an error.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)
running_codes, running_edges = quantile_codes(running_aligned)
# ...
def quantile_codes(values):
    edges = np.quantile(values.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int8)
    return codes, edges
```

iii. Session-wise quintiles ensure balanced class distributions within each session, accounting for different behavioral states across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using session-wise quintile boundaries (20th, 40th, 60th, 80th percentiles). `np.searchsorted` with `side="right"` assigns codes 0-4.

ii.
```python
edges = np.quantile(values.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
codes = np.searchsorted(edges, values, side="right").astype(np.int8)
```

iii. The AI uses 4 internal edges (quintile boundaries) and `searchsorted` to produce 5 bins. This differs from the reference's `np.digitize` with global percentile edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz `target_t` grid used for neural data, ensuring temporal alignment.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)
```

iii. Using the common time grid guarantees alignment between neural and behavioral data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `acquisition/EyeTracking/pupil_tracking/height`, along with eye tracking timestamps. The diameter is computed as `2 * max(width, height)`.

ii.
```python
eye_t = nwb["acquisition/EyeTracking/eye_tracking/timestamps"][:].astype(np.float64)
pupil_width = nwb["acquisition/EyeTracking/pupil_tracking/width"][:].astype(np.float64)
pupil_height = nwb["acquisition/EyeTracking/pupil_tracking/height"][:].astype(np.float64)
pupil_raw = 2.0 * np.maximum(pupil_width, pupil_height)
```

iii. The AI justified using `2*max(width,height)`: "Whitepaper says major ellipse axis reflects pupil diameter." The width and height are ellipse half-axes, so doubling the major axis gives the diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing involves: (1) computing diameter from ellipse half-axes, (2) filling short internal NaN gaps (<=30 eye frames, ~1 second) by linear interpolation, (3) identifying residual long/edge gaps, (4) excluding trials that overlap residual gaps, (5) interpolating the cleaned signal to 30 Hz bin centers, (6) session-wise quintile discretization.

ii.
```python
pupil_filled, short_runs, residual_runs = fill_short_internal_gaps(
    pupil_raw, eye_t, MAX_PUPIL_GAP_FRAMES
)
invalid_trials = trials_overlapping_invalid_runs(
    candidate_starts, candidate_stops, eye_t, residual_runs
)
kept_indices = candidate_indices[~invalid_trials]
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
```

iii. The AI's approach to pupil processing is more sophisticated than the reference: it handles blink gaps explicitly by filling short gaps and excluding trials with long gaps, rather than simply mapping NaN to bin 0. Session-wise quintiles account for per-animal/per-rig calibration differences.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: session-wise quintile discretization into 5 bins (codes 0-4).

ii.
```python
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
```

iii. Session-wise quantiles ensure balanced bins and handle cross-session calibration differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 30 Hz `target_t` time grid used for neural data, after blink gap handling and trial exclusion.

ii.
```python
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
```

iii. Same temporal alignment approach as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
outcome_flags = np.vstack([trials[name][:].astype(bool) for name in OUTCOME_COLUMNS])
if np.any(outcome_flags[:, eligible].sum(axis=0) != 1):
    raise AssertionError(f"{eid}: retained outcomes are not mutually exclusive/exhaustive")
```

iii. The AI verifies mutual exclusivity with an assertion, ensuring exactly one outcome flag is set per trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The index of the set outcome flag (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) is used as the integer code. This code is repeated across all time bins within the trial.

ii.
```python
outcome = int(np.flatnonzero(outcome_flags[:, raw_index])[0])
output_trial[4].fill(outcome)
```

iii. Using `np.flatnonzero` and indexing into the stacked flags produces the outcome integer directly.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data through several mechanisms:
- **Missing eye tracking**: Entire sessions lacking the EyeTracking group are excluded.
- **Pupil blink gaps**: Short gaps (<=30 eye frames) are linearly interpolated; trials overlapping longer gaps are excluded.
- **Missing metadata**: Experiments without metadata entries are excluded.
- **Passive sessions**: Excluded entirely.
- **Nonfinite values**: Neural, running, and pupil data are checked for finiteness with hard errors (rather than silent fill).
- **All-zero neural trials**: Accepted as genuine sparse event silence; not filtered.

ii.
```python
# Missing eye tracking -> exclude session
if "EyeTracking" not in nwb["acquisition"]:
    excluded.append({"ophys_experiment_id": eid, "reason": "missing eye tracking"})
    continue
# Nonfinite check
if not np.all(np.isfinite(events)):
    raise ValueError(f"{eid}: event data contain NaN/Inf")
if not np.all(np.isfinite(pupil_aligned)):
    raise ValueError(f"{eid}: nonfinite pupil after valid-trial selection")
```

iii. The AI takes a strict approach: rather than silently filling missing data (like mapping NaN to bin 0), it either excludes the affected data or raises errors. This prevents downstream issues from fabricated values.

## 9-a. What are the most time-consuming steps of the code?

i. Reading the NWB files from disk is the primary bottleneck, particularly reading the event detection data matrix (`processing/ophys/event_detection/data`), which contains the full-session neural activity for all neurons. The AI mitigates this with direct `h5py` access and `read_direct` for float32 reads.

ii.
```python
def read_float32_dataset(dataset: h5py.Dataset) -> np.ndarray:
    destination = np.empty(dataset.shape, dtype=np.float32)
    dataset.read_direct(destination)
    return destination
```

iii. The CONVERSION_NOTES report full conversion completed in ~88 seconds for 199 sessions, with most time in HDF5 reads.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for constructing change codes iterates over each kept trial to set the change bin. This could be vectorized using array operations on the offsets and change times. The image code assignment also involves a list comprehension per shown index.

ii.
```python
# Per-trial loop for change codes
for trial_position, raw_index in enumerate(kept_indices):
    if not go[raw_index]:
        continue
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
    change_codes[lo + local_index] = 1
```

iii. The loop is not a major bottleneck since it only iterates over go trials, but it could theoretically be replaced with vectorized searchsorted operations.

## 9-c. What processing does the code repeat multiple times?

i. The code does not repeat significant processing. Each NWB file is read once, and all trial data is extracted in a single pass. The `quantile_codes` function is called twice per session (once for running, once for pupil), but these are independent computations on different data.

ii. N/A

iii. The code is well-structured with a single-pass architecture.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive metadata per session (cell specimen IDs, removed trial IDs, pupil gap statistics, outcome counts, native frame rate, etc.) that is unlikely to be used by the downstream decoder. The diagnostic data for `--show-processing` mode also captures additional arrays, though this is gated behind a flag.

ii.
```python
info = {
    "ophys_experiment_id": int(eid),
    "cell_specimen_ids": cell_ids.tolist(),
    "removed_raw_trial_ids": trials["id"][:][removed_indices].astype(int).tolist(),
    "retained_raw_trial_ids": trials["id"][:][kept_indices].astype(int).tolist(),
    "short_pupil_gaps_interpolated": int(len(short_runs)),
    # ... many more fields
}
```

iii. This metadata is useful for auditing but is not consumed by the decoder. It adds some overhead to pickle serialization.
