# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter reads the local `ophys_experiment_table.csv`, selects exact-project active experiments with eye data, and opens each selected NWB directly with `h5py`. It therefore loads 165 local experiments, not the reference converter's SDK-discovered cohort grouped across all planes of each ophys session.

ii.
```python
table = pd.read_csv(EXPERIMENT_TABLE)
selected = table[table["project_code"].eq("VisualBehavior")
                 & ~table["passive"].astype(bool)
                 & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)].copy()
...
with h5py.File(path, "r") as nwb:
```

iii. The agent said passive replay lacks contemporaneous outcomes, eye-less experiments cannot supply a required output without fabrication, and direct HDF5 access preserves SDK semantics while avoiding unrelated data loading.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB, checked against metadata `mouse_id`, then deduplicated and numerically sorted; `subject_idx` maps each experiment/session to that list.

ii.
```python
raw_subject = nwb["general/subject/subject_id"][()]
if raw_subject != str(int(row["mouse_id"])): raise ValueError(...)
subjects = sorted(set(session_subjects), key=int)
subject_idx = np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int32)
```

iii. The agent treated the NWB subject identifier as authoritative and added a metadata consistency check.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id` (one NWB/imaging plane) is treated as one output session and sorted by experiment ID. Experiments sharing `ophys_session_id` are not combined.

ii.
```python
selected.sort_values("ophys_experiment_id", inplace=True)
for session_num, (_, row) in enumerate(selected.iterrows(), start=1):
    converted, plot_context = convert_session(row, make_plot=make_plot)
```

iii. The agent reasoned that the exact `VisualBehavior` project is single-plane and repeatedly called experiments “sessions.”

## 1-d. How are the data split into trials?

i. Eligible SDK/NWB trial-table rows are sliced on the native ophys clock using half-open `[start_time, stop_time)` bounds, producing variable-length trials.

ii.
```python
starts = trials["start_time"][:][raw_rows]
stops = trials["stop_time"][:][raw_rows]
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
neural = event_block[a - block_left : b - block_left].T.copy()
```

iii. The notes justify this as the native SDK trial definition and explicitly audit endpoint and overlap behavior.

## 1-e. How are trials filtered based on quality controls?

i. The code keeps `(go OR catch) AND NOT aborted AND NOT auto_rewarded`, requires at least two trials and exactly one valid outcome, and rejects empty/overlapping slices.

ii.
```python
keep = (go | catch) & ~aborted & ~auto_rewarded
if len(raw_rows) < 2: raise ValueError(...)
if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1): raise ValueError(...)
```

iii. This is justified directly by the requested trial types and by avoiding biased free-reward and premature-lick trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB unfiltered L0 calcium-event matrix and its timestamps, not from dF/F.

ii.
```python
event_group = nwb["processing/ophys/event_detection"]
event_data = event_group["data"]
ophys_t = event_group["timestamps"][:]
```

iii. The agent chose L0 events because the supplied paper used detected event magnitudes for neural analyses/decoding, despite initially noting that dF/F was the precomputed standard stream.

## 2-b. How is the `neural` data processed?

i. A bounding time block is read directly as float32, trial row slices are transposed to neuron-by-time, and otherwise the unfiltered event magnitudes are unchanged.

ii.
```python
event_block = np.empty((block_right - block_left, event_data.shape[1]), dtype=np.float32)
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
neural = event_block[a - block_left : b - block_left].T.copy()
```

iii. The agent aimed to retain the paper's raw event representation and reduce memory/I/O by one bounding read per experiment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No signal-based filtering is added. Stored ROIs must all be `valid_roi`, cell-table length must match event columns, and all loaded events must be finite.

ii.
```python
if not np.all(np.isfinite(event_block)): raise ValueError(...)
if "valid_roi" in cell_table and not np.all(cell_table["valid_roi"][:]): raise ValueError(...)
if len(cell_table["id"]) != event_data.shape[1]: raise ValueError(...)
```

iii. The agent states that release QC and ROI curation are already embodied in the NWB and that ad hoc activity filtering would bias the data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to each trial's start and stop times on the ophys timestamp grid; the first retained frame is at or after trial start.

ii.
```python
left = np.searchsorted(ophys_t, starts, side="left")
right = np.searchsorted(ophys_t, stops, side="left")
```

iii. The agent cites the instruction to align by ophys timestamp and documents exhaustive off-by-one checks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native single-plane ophys frames are retained at a median 32.31 ms; no temporal rebinning is applied.

ii.
```python
"ophys_median_dt_s": float(np.median(np.diff(ophys_t))),
median_dt_ms = float(np.median([x["ophys_median_dt_s"] for x in session_info]) * 1000.0)
```

iii. The agent preferred native ophys samples because alignment to ophys timestamps was explicit, even though the paper sometimes interpolated analyses to 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the natural-image stimulus-presentation table: `start_time`, `stop_time`, `image_name`, `active`, and `omitted`.

ii.
```python
starts = group["start_time"][:]; stops = group["stop_time"][:]
names = decode_strings(group["image_name"][:])
active = group["active"][:].astype(bool)
omitted = np.nan_to_num(group["omitted"][:], nan=0.0).astype(bool)
```

iii. The agent argues that presentation intervals correctly preserve 250-ms flashes, gray intervals, and omissions, unlike a trial-level before/after label.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Presentations are sorted; each target frame is assigned the containing active, non-omitted known image. Everything else is global code 0 (`gray`), and 16 natural images have fixed codes 1–16.

ii.
```python
idx = np.searchsorted(starts, target_t, side="right") - 1
shown = nonnegative & (target_t < stops[safe_idx]) & active[safe_idx] & ~omitted[safe_idx] & known_image
image = np.zeros(len(target_t), dtype=np.int16)
image[shown] = np.fromiter((IMAGE_TO_CODE[name] for name in names[safe_idx[shown]]), ...)
```

iii. The agent justifies a gray category because the instruction asks for the image shown during non-gray periods and the stimulus actually alternates images and gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the concatenated ophys timestamps from the exact retained trial slices, then split using the same offsets as neural data.

ii.
```python
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
image_codes, change_codes = presentation_outputs(presentations, target_t)
output[0] = image_codes[lo:hi]
```

iii. The agent validated reconstructed stimulus rows against raw NWB timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table's `is_change` flag together with whether a known, active, non-omitted image is currently shown.

ii.
```python
is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. The agent treated the SDK presentation flag as the canonical changed-image event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The flag is projected from presentation intervals to ophys frames, so it is 1 throughout the changed image's on-screen interval and 0 otherwise.

ii.
```python
change = (shown & is_change[safe_idx]).astype(np.int16)
```

iii. The notes describe this as the “immediate” changed-image presentation, avoiding labeling gray time or a sham catch event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already Boolean and is cast to integer categories 0 (`no_change`) and 1 (`change`); no numeric threshold is used.

ii.
```python
change = (shown & is_change[safe_idx]).astype(np.int16)
"output_values": [..., ["no_change", "change"], ...]
```

iii. The agent regarded the released Boolean flag as eliminating any need for a fitted threshold.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is evaluated at the same retained ophys timestamps and sliced by the same per-trial offsets as neural data.

ii.
```python
image_codes, change_codes = presentation_outputs(presentations, target_t)
output[1] = change_codes[lo:hi]
```

iii. Raw reconstruction and processing plots were used to check coincidence with the changed flash.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed` data and timestamps from the NWB.

ii.
```python
running = nwb["processing/running/speed"]
running["timestamps"][:], running["data"][:]
```

iii. The agent chose the released filtered speed stream rather than re-deriving wheel velocity.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are sorted/deduplicated if needed, linearly interpolated to retained ophys frames with endpoint hold, then converted to quintile codes.

ii.
```python
running_aligned = interpolate_finite(running["timestamps"][:], running["data"][:], target_t, "running speed")
running_codes, running_edges = quintile_codes(running_aligned)
```

iii. Timestamp interpolation avoids equating indices from different clocks; released filtering is preserved.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five within-experiment/session percentile bins are computed over retained eligible-trial frames using 20/40/60/80% boundaries.

ii.
```python
edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
codes = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
```

iii. The agent says within-session quintiles balance classes and avoid treating between-session scale differences as biological.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Continuous running speed is interpolated directly to `target_t`, the concatenation of retained neural-frame timestamps, and split at identical offsets.

ii.
```python
running_aligned = interpolate_finite(..., target_t, ...)
output[2] = running_codes[lo:hi]
```

iii. The agent cites hardware synchronization and explicit timestamps, with `np.allclose` audits of reconstructed trials.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses blink-masked pupil `area` and timestamps from `acquisition/EyeTracking/pupil_tracking`.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil_area = pupil["area"][:]
```

iii. The agent concluded released area encodes a circle based on the ellipse major axis and retains blink masking via NaNs.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Nonnegative finite area is converted by `2*sqrt(area/pi)`; finite samples are linearly interpolated to ophys frames with endpoint hold, then quintile-coded.

ii.
```python
pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
pupil_aligned = interpolate_finite(pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter")
```

iii. This was justified as recovering major-axis diameter exactly while bridging short blink/outlier gaps without using raw invalid fits.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five within-experiment/session quintiles are computed over retained frames with the same quantile routine as running speed.

ii.
```python
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
```

iii. The agent argues within-session ranks balance labels and avoid camera-scale differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Derived finite diameter samples are timestamp-interpolated directly to retained neural-frame `target_t` and split with the neural trial offsets.

ii.
```python
pupil_aligned = interpolate_finite(..., target_t, ...)
output[3] = pupil_codes[lo:hi]
```

iii. The agent reports direct raw-data reconstruction checks and excludes three experiments with no eye samples rather than fabricating them.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from trial-table Boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
outcomes = np.argmax(outcome_flags[raw_rows], axis=1).astype(np.int16)
```

iii. The agent treats these mutually exclusive SDK flags as canonical and explicitly validates exclusivity.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The true flag's column index gives code 0–3, which is repeated across every frame of the trial.

ii.
```python
output[4].fill(outcomes[j])
```

iii. Repetition permits a static label to coexist with time-varying rows in one rectangular output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code fails loudly on missing selected NWBs, ID mismatches, inadequate trials, invalid outcomes/timestamps/events/ROIs, or insufficient finite behavior. It sorts/deduplicates malformed behavior timestamps, interpolates across blink NaNs, and excludes three entirely eye-less experiments. It does not skip arbitrary failed sessions.

ii.
```python
if not missing_files.empty: raise FileNotFoundError(...)
good = np.isfinite(source_t) & np.isfinite(source_x)
if np.any(np.diff(t) <= 0): ... t, x = t[unique], x[unique]
MISSING_EYE_EXPERIMENTS = {795953296, 806456687, 833631914}
```

iii. The agent preferred explicit validation over silent corruption and considered excluding eye-less sessions more defensible than fabricated pupil labels.

## 9-a. What are the most time-consuming steps of the code?

i. Reading dense event blocks and serializing the roughly 7.8-GiB pickle dominate; the logged full conversion spent about 58.6 s loading/aligning/slicing and 8.1 s serializing.

ii.
```python
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify neural HDF5 I/O as the main cost and report measured timings.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most alignment is vectorized. Remaining loops include per-session iteration, per-trial copying/assembly, string decoding/image lookup, and construction of `target_t`; some could be reduced, though variable trial shapes still require final slicing.

ii.
```python
target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])
for j, (a, b) in enumerate(zip(left, right)):
    neural_trials.append(neural)
```

iii. The agent says it deliberately vectorized timestamp alignment, interpolation, presentation lookup, percentile coding, and outcomes, while retaining a readable final trial loop.

## 9-c. What processing does the code repeat multiple times?

i. `interpolate_finite` is run separately for running and pupil; every experiment repeats NWB validation, presentation mapping, quantile calculation, and trial assembly. Optional plotting also consumes already computed arrays, but source processing is not reloaded in a second pass.

ii.
```python
running_aligned = interpolate_finite(...)
pupil_aligned = interpolate_finite(...)
running_codes, running_edges = quintile_codes(running_aligned)
pupil_codes, pupil_edges = quintile_codes(pupil_aligned)
```

iii. The agent emphasized one-pass, session-at-a-time processing and reuse of aligned arrays, so repeated work is mostly the same necessary operation on distinct streams/sessions.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads a contiguous event block from the earliest retained start to latest retained stop, including excluded inter-trial frames that are then discarded. It also collects extensive metadata/cell IDs and optionally builds plot context unused by decoder training.

ii.
```python
event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
neural = event_block[a - block_left : b - block_left].T.copy()
cell_specimen_ids = cell_table["cell_specimen_id"][:].astype(np.int64).tolist()
```

iii. The agent explicitly accepted the discarded bounding-block gaps because one sequential HDF5 read is faster than tens of thousands of small per-trial reads; metadata supports auditing and reproducibility.
