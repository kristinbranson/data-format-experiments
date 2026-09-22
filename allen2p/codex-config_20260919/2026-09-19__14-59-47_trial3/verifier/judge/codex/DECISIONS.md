# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local release directly with `h5py`, discovers NWBs by globbing, joins them to the experiment metadata CSV, and selects available active-behavior experiments having eye tracking and required datasets. It processes each selected NWB sequentially rather than using the AllenSDK cache.

ii.
```python
metadata = pd.read_csv(METADATA_PATH).set_index("ophys_experiment_id", drop=False)
paths = sorted(NWB_ROOT.glob("behavior_ophys_experiment_*.nwb"))
available = {experiment_id_from_path(path): path for path in paths}
...
if row["behavior_type"] != "active_behavior":
    ... continue
with h5py.File(path, "r") as nwb:
    if "EyeTracking" not in nwb["acquisition"]:
        ... continue
```

iii. The notes say direct HDF5 reads access the same underlying released streams while avoiding SDK materialization of masks, projections, and irrelevant tables. Active sessions were chosen to match the paper’s task analysis; sessions without eye tracking were excluded because pupil is required.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values, converted to strings and numerically sorted; each experiment session receives the corresponding subject index.

ii.
```python
subjects = sorted({result.subject for result in results}, key=int)
subject_lookup = {name: i for i, name in enumerate(subjects)}
"subject_idx": np.asarray([subject_lookup[result.subject] for result in results], dtype=np.int32)
```

iii. The agent treats `mouse_id` as the released unique animal identifier and reports retaining all 38 mice represented by its selected local cohort.

## 1-c. How are the data split into sessions?

i. Every selected NWB/`ophys_experiment_id`—one imaging plane—is a separate output session, even when several experiments share an `ophys_session_id`.

ii.
```python
for position, (eid, path) in enumerate(selected):
    result = process_session(eid=eid, path=path, meta=metadata.loc[eid], ...)
    results.append(result)
...
"neural": [result.neural for result in results]
```

iii. The notes justify the imaging plane as the natural neural sampling unit and say it is consistent with the paper; simultaneous planes therefore repeat behavioral trials in separate decoder sessions.

## 1-d. How are the data split into trials?

i. Native NWB trial rows are used. Exactly `go | catch` candidates are retained, and each trial spans its native `start_time` to `stop_time`. Complete 30 Hz bins are created inside that interval.

ii.
```python
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
eligible = go | catch
...
per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
```

iii. The agent trusts the synchronized released trial flags and boundaries instead of re-deriving trials. Variable-duration native trials are retained, with bin centers used to include only complete bins.

## 1-e. How are trials filtered based on quality controls?

i. Go/catch status implicitly excludes aborted and auto-rewarded rows, and assertions verify the categories. Trials overlapping unresolved pupil NaN runs (edge gaps or internal gaps longer than 30 eye frames) are removed; a session must retain at least two trials.

ii.
```python
eligible = go | catch
if np.any(eligible & (aborted | auto_rewarded)):
    raise AssertionError(...)
invalid_trials = trials_overlapping_invalid_runs(..., residual_runs)
kept_indices = candidate_indices[~invalid_trials]
if len(kept_indices) < 2:
    raise RuntimeError(...)
```

iii. The requested go/catch curation motivated the first filter. The notes argue that long missing-pupil spans should not be fabricated and report retaining about 95.8% of otherwise eligible trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from released FastLZero detected calcium-event magnitudes and their timestamps, not dF/F or SDK-smoothed events.

ii.
```python
ophys_t = nwb["processing/ophys/event_detection/timestamps"][:].astype(np.float64)
event_dataset = nwb["processing/ophys/event_detection/data"]
events = read_float32_dataset(event_dataset)
```

iii. The agent says the paper used inferred events to remove slow GCaMP decay, whereas dF/F is less appropriate and `filtered_events` adds visualization-oriented smoothing.

## 2-b. How is the `neural` data processed?

i. The complete time-by-cell event matrix is read as float32, linearly interpolated in one vectorized operation to concatenated 30 Hz trial-bin centers, sliced by trial, transposed to neuron-by-time, and made contiguous.

ii.
```python
events = read_float32_dataset(event_dataset)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
...
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. The notes cite the paper’s interpolation of event-triggered activity to 30 Hz and emphasize vectorization, float32 storage, and avoiding per-neuron loops and redundant reads.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No activity threshold is imposed. All released cells in each selected NWB are retained, while code asserts that every cell-table entry is a valid ROI and that event/cell dimensions agree. Nonfinite events are fatal.

ii.
```python
if not np.all(np.isfinite(events)):
    raise ValueError(...)
...
valid_rois = nwb["processing/ophys/image_segmentation/cell_specimen_table/valid_roi"][:].astype(bool)
if len(cell_ids) != nneurons or not np.all(valid_rois):
    raise AssertionError(...)
```

iii. The agent relies on release-level segmentation, ROI, event-inference, and experiment QC and rejects an additional unreferenced activity filter, even when sparse event traces yield all-zero trial windows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to native trial start on the synchronized ophys clock. Target samples are centers at `start + (k + 0.5)/30`, and neural events are interpolated at those times through trial stop.

ii.
```python
def trial_centers(start, stop):
    n_bins = int(np.floor((stop - start) * TARGET_HZ + 1.0e-9))
    return start + (np.arange(n_bins) + 0.5) * BIN_SIZE_S
...
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
```

iii. The agent describes the alignment event as native trial start, retains full variable-duration trials, and uses hardware-synchronized timestamps so all streams share the ophys clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 30 Hz (33.333 ms). Native event samples at roughly 11 or 31 Hz are linearly resampled; this is interpolation, not count aggregation.

ii.
```python
TARGET_HZ = 30.0
BIN_SIZE_S = 1.0 / TARGET_HZ
...
"time_bin_size": 1000.0 / TARGET_HZ
```

iii. The agent chose 30 Hz because bins must be common across sessions and the paper interpolated events and running to a 30 Hz relative timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the natural-image presentation table’s `image_name`, `start_time`, `stop_time`, `active`, and `omitted` fields.

ii.
```python
presentation_starts = presentations["start_time"][:]
presentation_stops = presentations["stop_time"][:]
active = presentations["active"][:].astype(bool)
omitted = np.nan_to_num(presentations["omitted"][:], nan=0.0).astype(bool)
image_names = decode_strings(presentations["image_name"][:])
```

iii. The agent uses exact presentation intervals because the requested identity is the image actually visible during non-gray epochs, including repeats and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A fixed global codebook contains gray plus 16 image names. Every target bin defaults to gray; a named code is assigned only if its center falls inside an active, non-omitted presentation interval. Omissions and inter-stimulus gaps remain gray.

ii.
```python
image_codes = np.zeros(len(target_t), dtype=np.int8)
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
shown[...] = active[...] & ~omitted[...] & (target_t[...] < presentation_stops[...])
image_codes[shown_indices] = [IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]]
```

iii. The notes state that an omission presents no image and therefore belongs to gray; the fixed 17-class codebook keeps image codes global across image sets A and B.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation membership is evaluated at exactly the concatenated 30 Hz bin centers used to interpolate neural data, then sliced with the same per-trial offsets.

ii.
```python
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
...
output_trial[0] = image_codes[lo:hi]
neural_trial = neural_aligned[lo:hi].T
```

iii. The common synchronized target grid is intended to guarantee sample-for-sample neural/output alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses the trial table’s `go` and `change_time` fields, and independently cross-checks them against active presentation `is_change` start times.

ii.
```python
change_times_all = trials["change_time"][:].astype(np.float64)
if not go[raw_index]:
    continue
...
presentation_change = np.nan_to_num(presentations["is_change"][:], nan=0.0).astype(bool)
```

iii. The agent distinguishes true go changes from catch sham changes and verifies that trial and presentation timestamps agree exactly.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and exactly one sample—the first target center at or after `change_time`—is set to one for each retained go trial. Catch trials remain zero.

ii.
```python
change_codes = np.zeros(len(target_t), dtype=np.int8)
local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
change_codes[lo + local_index] = 1
```

iii. The agent interprets “right after a change” as a one-bin event pulse and explicitly rejects labeling sham changes, offsets, repeats, omissions, or a longer post-change interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: 0 is “no change” and 1 is “change”; no numeric threshold is estimated.

ii.
```python
"output_values": [..., ["no change", "change"], ...]
```

iii. A direct binary event code matches the requested output type.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The true change is placed at the first 30 Hz target center at or after its synchronized onset, on the same target grid and per-trial slice as neural data.

ii.
```python
local_index = np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left")
output_trial[1] = change_codes[lo:hi]
```

iii. The notes state that trial and stimulus change times agree and that a shared target grid prevents temporal shifts.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB processed running-speed `data` and `timestamps`, in cm/s.

ii.
```python
running_t = nwb["processing/running/speed/timestamps"][:].astype(np.float64)
running = nwb["processing/running/speed/data"][:].astype(np.float64)
```

iii. The agent chooses the stored processed speed rather than raw/unfiltered encoder values and avoids re-filtering or clamping legitimate negatives.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Processed speed is linearly interpolated to all retained 30 Hz trial centers, then converted to session-specific quintile codes using all retained trial samples in that experiment.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)
running_codes, running_edges = quantile_codes(running_aligned)
```

iii. The notes cite the paper’s linear interpolation and choose session-local percentiles to represent comparable behavioral-state ranks and balanced targets.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles within each session are boundaries for codes 0–4; ties use `side="right"`.

ii.
```python
edges = np.quantile(values.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
codes = np.searchsorted(edges, values, side="right").astype(np.int8)
```

iii. Five equal-percentile bins were explicitly requested; within-session thresholds were chosen to remove session calibration/state offsets.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated directly at the same concatenated 30 Hz target centers used for neural interpolation and split with identical offsets.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)
...
output_trial[2] = running_codes[lo:hi]
```

iii. Hardware synchronization and a shared target grid are the stated basis for alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking timestamps and pupil ellipse `width` and `height`; raw diameter is twice the larger fitted half-axis.

ii.
```python
pupil_width = nwb["acquisition/EyeTracking/pupil_tracking/width"][:]
pupil_height = nwb["acquisition/EyeTracking/pupil_tracking/height"][:]
pupil_raw = 2.0 * np.maximum(pupil_width, pupil_height)
```

iii. The notes interpret the pupil’s major ellipse axis as diameter and say these stored fits already carry blink/outlier NaNs from Allen processing.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Bounded internal NaN runs of at most 30 eye frames are linearly filled; longer or edge runs remain invalid and remove overlapping trials. The filled signal is interpolated to 30 Hz target centers and binned into session-specific quintiles.

ii.
```python
pupil_filled, short_runs, residual_runs = fill_short_internal_gaps(
    pupil_raw, eye_t, MAX_PUPIL_GAP_FRAMES)
invalid_trials = trials_overlapping_invalid_runs(..., residual_runs)
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
```

iii. The one-second rule is intended to fill blink-scale gaps without inventing long stretches; session-local ranks address rig/animal pixel calibration differences.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The retained, aligned pupil values in each session are divided at their 20th, 40th, 60th, and 80th percentiles into codes 0–4.

ii.
```python
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
```

iii. This implements the required five equal-percentile categories while balancing each session independently.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. After short-gap filling, pupil is linearly interpolated at the exact same target bin centers and sliced with the same offsets as neural data.

ii.
```python
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
...
output_trial[3] = pupil_codes[lo:hi]
```

iii. The agent relies on hardware-synchronized clocks and rejects extrapolation outside finite pupil coverage.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the mutually exclusive trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
outcome_flags = np.vstack([trials[name][:].astype(bool) for name in OUTCOME_COLUMNS])
```

iii. These are the SDK/NWB’s canonical go/no-go outcomes; the code asserts exactly one for every eligible trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The position of the true flag maps outcomes to 0–3 in the declared order, and that code is repeated across every time bin of the trial.

ii.
```python
outcome = int(np.flatnonzero(outcome_flags[:, raw_index])[0])
output_trial[4].fill(outcome)
```

iii. Repetition lets one static variable coexist with four time-varying rows in a single `(5, T)` matrix consumed by the validator/decoder.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Known missing pupil values receive bounded short-gap interpolation; trials touching long/edge gaps and sessions without eye tracking are excluded. Unexpected missing/nonfinite neural, running, pupil, or structural inconsistencies raise hard errors. Target times outside source coverage also raise errors; output is written atomically.

ii.
```python
if bounded and end - start <= max_gap_frames:
    filled[start:end] = np.interp(...)
else:
    residual_runs.append(...)
...
if not np.all(np.isfinite(events)):
    raise ValueError(...)
temporary.replace(output)
```

iii. The agent explicitly avoids raw pupil substitution, global/zero filling, and silent extrapolation. It argues exclusion is safer than fabricating required pupil targets and records excluded sessions/trials in metadata.

## 9-a. What are the most time-consuming steps of the code?

i. Reading each large event matrix, interpolating it to all retained target times, retaining the multi-gigabyte converted neural payload, and serializing the final pickle dominate. The measured full run reports 81.4 s processing and 6.69 s serialization.

ii.
```python
events = read_float32_dataset(event_dataset)
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
...
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify large NWB/session neural streams as the main size cost and use direct sequential reads and float32 arrays to keep the full conversion to 88.09 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most expensive interpolation is already vectorized. Remaining loops include NaN-run filling, trial/change placement, trial slicing/output construction, image-name code assignment, experiment selection, and validation; some code assignment and trial assembly could be further batched, but variable lengths limit gains.

ii.
```python
for start, end in zip(starts, ends): ...
for trial_position, raw_index in enumerate(kept_indices): ...
[IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]]
```

iii. The notes specifically say they avoided per-neuron interpolation and repeated per-trial HDF5 reads by concatenating trial targets and interpolating whole matrices once.

## 9-c. What processing does the code repeat multiple times?

i. Trial loops are repeated for change-pulse placement, final slicing/assembly, and optional diagnostics/validation. Presentation membership and interpolation are computed once per session; raw streams are not reread during normal conversion. Validation subsequently traverses every output trial again.

ii.
```python
for trial_position, raw_index in enumerate(kept_indices):  # changes
    ...
for trial_position, raw_index in enumerate(kept_indices):  # assembly
    ...
for neural, inputs, outputs in zip(...):                    # validation
    ...
```

iii. The agent emphasizes one sequential read per stream/session and one concatenated alignment, accepting inexpensive repeated traversal for construction and strict checks.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In ordinary conversion little computed data is discarded beyond temporary aligned full-session arrays. With `--show-processing`, it copies diagnostic subsets and builds plots that the decoder never uses. It also performs extensive cross-checks and collects detailed metadata that do not enter model training, though they support validation and provenance.

ii.
```python
if make_diagnostic:
    diagnostic = {"events_sample": events[:, :min(5, nneurons)], ...}
...
if result.diagnostic is not None:
    plot_processing(...)
```

iii. The notes justify diagnostics and assertions as sanity checks. They deliberately avoid loading image masks/projections and other SDK objects that would truly be unused downstream.
