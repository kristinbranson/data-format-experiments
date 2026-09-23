# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads the local release directly from `/app/data/visual-behavior-ophys-1.1.0/` using `pandas` for the experiment metadata CSV and `h5py` for each NWB. It enumerates NWB files, matches them to metadata rows by `ophys_experiment_id`, and only keeps experiments that are active behavior, have eye tracking, and contain several required datasets.

ii.
```python
metadata = pd.read_csv(METADATA_PATH).set_index("ophys_experiment_id", drop=False)
paths = sorted(NWB_ROOT.glob("behavior_ophys_experiment_*.nwb"))
available = {experiment_id_from_path(path): path for path in paths}

for eid in candidate_ids:
    row = metadata.loc[eid]
    if row["behavior_type"] != "active_behavior":
        ...
    with h5py.File(path, "r") as nwb:
        if "EyeTracking" not in nwb["acquisition"]:
            ...
        required = [
            "processing/ophys/event_detection/data",
            "processing/ophys/event_detection/timestamps",
            "processing/running/speed/data",
            "processing/running/speed/timestamps",
            "intervals/trials",
        ]
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI justifies this as a direct-NWB implementation of the Allen release, with extra cohort restrictions because pupil diameter is a required decoder output and because it wanted to exclude passive viewing up front.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the experiment metadata. The final `subjects` list is the sorted unique mouse IDs, and each retained session stores the corresponding `mouse_id` string.

ii.
```python
subjects = sorted({result.subject for result in results}, key=int)
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
subject=str(int(meta["mouse_id"]))
...
"subject_idx": np.asarray([subject_lookup[result.subject] for result in results], dtype=np.int32),
```

iii. Step 5 of `CONVERSION_NOTES.md` says subject identity should come directly from metadata `mouse_id`, with all retained mice preserved.

## 1-c. How are the data split into sessions?

i. The AI treats each retained NWB experiment, i.e. each `ophys_experiment_id` / imaging plane, as one decoder session. It does not group multiple planes that share an `ophys_session_id`.

ii.
```python
for position, (eid, path) in enumerate(selected):
    result = process_session(
        eid=eid,
        path=path,
        meta=metadata.loc[eid],
        make_diagnostic=args.show_processing and position < 2,
    )
...
"session_definition": "one released ophys experiment/imaging plane",
```

iii. The justification in Step 4 and Step 5 is that each NWB contains a distinct neural population and is therefore the natural session unit for the target format, even when multiple planes were recorded simultaneously.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI keeps `go` or `catch` rows, then defines each retained trial by its native `start_time` and `stop_time`, but converts that interval into a set of evenly spaced 30 Hz bin centers rather than native ophys frames.

ii.
```python
trials = nwb["intervals/trials"]
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
eligible = go | catch
...
starts_all = trials["start_time"][:].astype(np.float64)
stops_all = trials["stop_time"][:].astype(np.float64)
candidate_indices = np.flatnonzero(eligible)
...
per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
```

iii. In Step 5, the AI says it is trusting the synchronized NWB trial table and converting each variable-duration trial into complete 1/30 s bins so that all sessions share a common bin size.

## 1-e. How are trials filtered based on quality controls?

i. The code keeps only `go | catch` trials and asserts that such trials are not aborted or auto-rewarded. It then removes any trial overlapping a long or edge pupil-invalid interval, requires at least two retained trials per experiment, and requires each retained trial to have at least two 30 Hz bins.

ii.
```python
eligible = go | catch
if np.any(eligible & (aborted | auto_rewarded)):
    raise AssertionError(f"{eid}: eligible trial is aborted/auto-rewarded")
...
invalid_trials = trials_overlapping_invalid_runs(
    candidate_starts, candidate_stops, eye_t, residual_runs
)
kept_indices = candidate_indices[~invalid_trials]
...
if len(kept_indices) < 2:
    raise RuntimeError(f"{eid}: only {len(kept_indices)} valid trials after pupil filtering")
...
if any(len(x) < 2 for x in per_trial_t):
    raise AssertionError(f"{eid}: retained trial has fewer than two bins")
```

iii. Step 4 and Step 5 justify the extra pupil-based filtering by arguing that pupil diameter is a required output, so trials with irrecoverable pupil gaps should be removed rather than filled with fabricated values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the released event-detection stream, not from dF/F. The code reads `processing/ophys/event_detection/data` and its timestamps from each NWB.

ii.
```python
ophys_t = nwb["processing/ophys/event_detection/timestamps"][:].astype(np.float64)
event_dataset = nwb["processing/ophys/event_detection/data"]
events = read_float32_dataset(event_dataset)
```

iii. In Step 4 and Step 5, the AI explicitly argues that the paper used detected calcium events, not dF/F, and therefore event magnitudes are the better neural representation.

## 2-b. How is the `neural` data processed?

i. The code reads the full event matrix as `float32`, linearly interpolates it onto the concatenated 30 Hz target timestamps for all retained trials in the experiment, then slices trial segments and transposes each to `(neurons, time)`.

ii.
```python
events = read_float32_dataset(event_dataset)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
...
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. Step 5 says the AI wanted a common 30 Hz grid across sessions and viewed linear interpolation of the event stream as the closest match to the paper’s event-triggered analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no activity-based neuron filtering. Instead, the code assumes the local release already contains valid ROIs and asserts that the cell table length matches the event matrix and that all `valid_roi` flags are true.

ii.
```python
cell_ids = nwb[
    "processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id"
][:].astype(np.int64)
valid_rois = nwb[
    "processing/ophys/image_segmentation/cell_specimen_table/valid_roi"
][:].astype(bool)
if len(cell_ids) != nneurons or not np.all(valid_rois):
    raise AssertionError(f"{eid}: event/cell table mismatch or invalid ROI present")
```

iii. The notes say released ROIs already passed Allen QC, so the AI did not add a further signal-quality threshold. The code operationalizes that as an assertion rather than a filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned to 30 Hz bin centers within each trial, anchored to native trial start on the synchronized ophys clock. They are not aligned to native frame indices and not aligned to `change_time`.

ii.
```python
def trial_centers(start: float, stop: float) -> np.ndarray:
    n_bins = int(np.floor((stop - start) * TARGET_HZ + 1.0e-9))
    ...
    return start + (np.arange(n_bins, dtype=np.float64) + 0.5) * BIN_SIZE_S
...
"temporal_alignment_event": "native trial start on the synchronized ophys clock",
```

iii. Step 5 says the AI interpreted the instructions as requiring ophys-clock alignment plus a common trial-centered 30 Hz sampling grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz binning, i.e. 33.333... ms per bin. Yes, temporal rebinning is applied through interpolation from native timestamps to that grid.

ii.
```python
TARGET_HZ = 30.0
BIN_SIZE_S = 1.0 / TARGET_HZ
...
"time_bin_size": 1000.0 / TARGET_HZ,
"target_sampling_rate_hz": TARGET_HZ,
```

iii. The notes justify 30 Hz by citing paper analyses that interpolated events and running to 30 Hz, and by the desire for one common bin size across single-plane and multi-plane recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, specifically `image_name`, `start_time`, `stop_time`, `active`, and `omitted`, not from the trial table’s `initial_image_name` and `change_image_name`.

ii.
```python
presentations = presentation_group(nwb)
presentation_starts = presentations["start_time"][:].astype(np.float64)
presentation_stops = presentations["stop_time"][:].astype(np.float64)
active = presentations["active"][:].astype(bool)
omitted = np.nan_to_num(presentations["omitted"][:], nan=0.0).astype(bool)
image_names = decode_strings(presentations["image_name"][:])
```

iii. Step 4 and Step 5 justify this by saying the presentation table gives the exact on-screen image intervals, which the AI thought was the correct source for a non-grey image-identity label.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI uses a fixed global codebook with one `gray` class plus 16 named images. For each 30 Hz sample, it finds the most recent presentation interval; if the sample falls inside an active, non-omitted image presentation, it emits that image’s code, otherwise it emits `gray`.

ii.
```python
IMAGE_VALUES = [
    "gray", "im000", "im031", "im035", "im045", "im054", "im061", "im062",
    "im063", "im065", "im066", "im069", "im073", "im075", "im077", "im085", "im106",
]
...
image_codes = np.zeros(len(target_t), dtype=np.int8)
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
...
shown[valid_pidx] = (
    active[pidx[valid_pidx]]
    & ~omitted[pidx[valid_pidx]]
    & (target_t[valid_pidx] < presentation_stops[pidx[valid_pidx]])
)
image_codes[shown_indices] = np.asarray(
    [IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]], dtype=np.int8
)
```

iii. In Step 5, the AI says omissions and inter-stimulus gaps should be labeled `gray`, because the requested variable is “the image presented during the non-grey screen.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on the same `target_t` timestamps as the interpolated neural data and then sliced trial-by-trial using the same `[lo:hi]` offsets.

ii.
```python
target_t = np.concatenate(per_trial_t)
...
output_trial[0] = image_codes[lo:hi]
...
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. The notes repeatedly describe all outputs as being computed on the common 30 Hz ophys-based grid so that each output row is exactly time-aligned with `neural`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived primarily from the trial table’s `change_time` and `go` flags, with a later consistency check against the presentation table’s `is_change` annotation.

ii.
```python
change_times_all = trials["change_time"][:].astype(np.float64)
for trial_position, raw_index in enumerate(kept_indices):
    if not go[raw_index]:
        continue
...
presentation_change = np.nan_to_num(presentations["is_change"][:], nan=0.0).astype(bool)
true_change_times = presentation_starts[active & presentation_change]
```

iii. The AI’s notes say catch trials are sham changes and therefore should not produce a positive image-change label, while true go-trial changes should coincide exactly with `is_change` in the presentation table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each retained go trial, the code finds the first 30 Hz sample at or after `change_time` and sets only that single bin to 1. All other bins, including every catch-trial bin, are 0.

ii.
```python
change_codes = np.zeros(len(target_t), dtype=np.int8)
for trial_position, raw_index in enumerate(kept_indices):
    if not go[raw_index]:
        continue
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
    ...
    change_codes[lo + local_index] = 1
```

iii. Step 4 and Step 5 justify this as labeling the true change onset itself, rather than a wider 750 ms post-change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: `0` means no change and `1` means the single detected true change bin. Catch trials remain entirely in category `0`.

ii.
```python
"output_values": [
    IMAGE_VALUES,
    ["no change", "change"],
    QUINTILE_VALUES,
    QUINTILE_VALUES,
    OUTCOME_VALUES,
]
...
change_codes = np.zeros(len(target_t), dtype=np.int8)
...
change_codes[lo + local_index] = 1
```

iii. The notes describe this as the most literal categorical rendering of “right after a change in image identity.”

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is computed on the same per-trial `target_t` timestamps as the neural data and copied into the same trial slice as the neural matrix.

ii.
```python
target_t = np.concatenate(per_trial_t)
...
output_trial[1] = change_codes[lo:hi]
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. The code and notes both treat image change as another row on the shared 30 Hz ophys-based timeline.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from the processed running stream in the NWB: `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
running_t = nwb["processing/running/speed/timestamps"][:].astype(np.float64)
running = nwb["processing/running/speed/data"][:].astype(np.float64)
```

iii. In Step 4 and Step 5, the AI says it is intentionally using the release’s processed speed signal rather than recomputing or using the unfiltered raw stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the common 30 Hz `target_t` grid, then converted to quintile codes using within-session quantiles computed across all retained trial samples from that experiment.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)
running_codes, running_edges = quantile_codes(running_aligned)
```

iii. Step 5 says session-wise percentile binning was chosen to remove session-specific scale offsets and to keep categories balanced within each retained session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five categories, coded `0` through `4`, based on the 20th, 40th, 60th, and 80th percentiles of that session’s retained running samples.

ii.
```python
def quantile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int8)
...
"output_values": [
    IMAGE_VALUES,
    ["no change", "change"],
    QUINTILE_VALUES,
    QUINTILE_VALUES,
    OUTCOME_VALUES,
]
```

iii. The notes call this “within session across retained trial timepoints” percentile scope.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same `target_t` timestamps used for the neural event traces, and then trial-sliced with the same offsets.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)
...
output_trial[2] = running_codes[lo:hi]
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. The notes justify this with the Allen synchronization model and the decision to place every output on the same 30 Hz ophys-derived grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking timestamps plus both pupil ellipse axes, `pupil_tracking/width` and `pupil_tracking/height`. The code defines diameter as twice the major half-axis.

ii.
```python
eye_t = nwb["acquisition/EyeTracking/eye_tracking/timestamps"][:].astype(np.float64)
pupil_width = nwb["acquisition/EyeTracking/pupil_tracking/width"][:].astype(np.float64)
pupil_height = nwb["acquisition/EyeTracking/pupil_tracking/height"][:].astype(np.float64)
...
pupil_raw = 2.0 * np.maximum(pupil_width, pupil_height)
```

iii. Step 4 and Step 5 explicitly justify this as the major-axis definition of pupil diameter described in the whitepaper, rather than using a single stored width column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code first identifies NaN gaps, linearly fills only bounded gaps of at most 30 eye-tracking frames, drops any trial overlapping a longer or edge gap, interpolates the filled pupil signal to `target_t`, and then bins the retained values into within-session quintiles.

ii.
```python
pupil_filled, short_runs, residual_runs = fill_short_internal_gaps(
    pupil_raw, eye_t, MAX_PUPIL_GAP_FRAMES
)
invalid_trials = trials_overlapping_invalid_runs(
    candidate_starts, candidate_stops, eye_t, residual_runs
)
...
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
...
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
```

iii. The notes justify this as a compromise between preserving short blink-scale gaps and refusing to invent long stretches of missing pupil data when pupil is itself a target variable.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five categories, coded `0` through `4`, using that session’s 20th/40th/60th/80th percentile boundaries after filtering and interpolation.

ii.
```python
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
...
"pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
...
"percentile_scope": "within session across retained trial timepoints",
```

iii. The notes say session-local binning was chosen because pupil scale is rig- and animal-dependent.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil signal is interpolated to the same `target_t` timestamps as the neural data, and then the resulting quintile codes are sliced per trial with the same offsets.

ii.
```python
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
...
output_trial[3] = pupil_codes[lo:hi]
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. The notes justify this the same way as running alignment: all streams are synchronized, then resampled onto the common 30 Hz ophys-based trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the four boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcome_flags = np.vstack([trials[name][:].astype(bool) for name in OUTCOME_COLUMNS])
```

iii. The notes say these are the canonical mutually exclusive task outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code finds which of the four outcome flags is true, maps that to integer codes `0..3` by column order, and repeats the same category for every time bin in the trial.

ii.
```python
outcome = int(np.flatnonzero(outcome_flags[:, raw_index])[0])
output_trial = np.empty((5, hi - lo), dtype=np.int16)
...
output_trial[4].fill(outcome)
```

iii. The AI justifies the repetition in Step 5 as necessary because all outputs are stored together in one `(5, T)` categorical matrix per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is generally strict rather than permissive. It excludes experiments with missing metadata, passive behavior, missing eye tracking, or missing required NWB datasets. It interpolates only short internal pupil gaps, removes trials overlapping longer or edge pupil gaps, and raises hard errors for timestamp mismatches, out-of-range interpolation, nonfinite event data, nonfinite retained pupil data, or invalid ROI inconsistencies.

ii.
```python
if eid not in metadata.index:
    excluded.append({"ophys_experiment_id": eid, "reason": "missing metadata"})
...
if "EyeTracking" not in nwb["acquisition"]:
    excluded.append({"ophys_experiment_id": eid, "reason": "missing eye tracking"})
...
if bounded and end - start <= max_gap_frames:
    filled[start:end] = np.interp(...)
else:
    residual_runs.append((int(start), int(end)))
...
if target_t[0] < valid_t[0] or target_t[-1] > valid_t[-1]:
    raise ValueError("Target timestamps outside finite behavioral stream")
```

iii. Step 4, Step 5, and Step 10 justify this as preferring exclusion or failure over silent imputation when required output streams are missing or unreliable.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify the expensive parts as reading the large HDF5 event arrays and then doing the vectorized interpolation onto the concatenated target grid. Plot generation is an additional cost in `--show-processing` mode.

ii.
```python
events = read_float32_dataset(event_dataset)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
...
if result.diagnostic is not None:
    plot_processing(result.diagnostic, APP_ROOT / f"processing_{eid}.png")
```

iii. Step 6 and Step 7 explicitly discuss direct stream-only HDF5 reads, float32 loading, and vectorized interpolation as the dominant performance-sensitive parts of the implementation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most heavy operations are already vectorized. The remaining obvious Python loops are the per-gap loop in `fill_short_internal_gaps`, the per-go-trial loop that sets change pulses, and the per-trial loop that slices `neural_aligned` and assembles the output matrices.

ii.
```python
for start, end in zip(starts, ends):
    ...

for trial_position, raw_index in enumerate(kept_indices):
    if not go[raw_index]:
        continue
    ...
    change_codes[lo + local_index] = 1

for trial_position, raw_index in enumerate(kept_indices):
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
    ...
```

iii. In Step 6, the AI presents the implementation as already optimized around bulk reads and bulk interpolation, implying that the remaining loops were kept for clarity because they were not the main bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. The code revisits the retained trial list several times: once to compute per-trial timestamps, once to assign change pulses, and once more to slice and package trial outputs. It also repeatedly uses search and slicing logic over the same concatenated trial timeline.

ii.
```python
per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
...
for trial_position, raw_index in enumerate(kept_indices):
    ...
    local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
...
for trial_position, raw_index in enumerate(kept_indices):
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    ...
```

iii. The notes do not call this out as a problem; the AI’s performance discussion focuses on avoiding repeated file I/O and avoiding per-neuron interpolation loops instead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The core conversion path is mostly retained in the output, but the script does extra validation and diagnostic work that is not used by downstream decoding: change-time cross-checks against the presentation table, diagnostic plot preparation, and extensive per-session metadata/summary bookkeeping.

ii.
```python
nearest = np.min(np.abs(go_change_times[:, None] - true_change_times[None, :]), axis=1)
if not np.allclose(nearest, 0.0, rtol=0.0, atol=1.0e-9):
    raise AssertionError(f"{eid}: trial and presentation change times disagree")
...
if make_diagnostic:
    diagnostic = {
        "eid": eid,
        "ophys_t": ophys_t,
        "events_sample": events[:, : min(5, nneurons)],
        ...
    }
...
"session_info": [result.info for result in results],
```

iii. Step 6, Step 10, and Step 12 show that the AI intentionally added these checks and diagnostics for auditing and plotting, not because the decoder itself needs them.
