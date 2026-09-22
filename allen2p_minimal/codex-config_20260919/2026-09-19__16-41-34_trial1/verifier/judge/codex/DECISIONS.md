# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers local `behavior_ophys_experiment_*.nwb` files, joins their IDs to `ophys_experiment_table.csv`, removes passive rows, and reads the NWBs directly with `h5py`. It does not filter `project_code` to `VisualBehavior`, so the supplied `VisualBehaviorMultiscope` files are also included.

ii.
```python
files = {_experiment_id(path): path for path in sorted(
    experiment_dir.glob("behavior_ophys_experiment_*.nwb"))}
table = pd.read_csv(metadata_path)
table = table[table["ophys_experiment_id"].isin(files)].copy()
table = table[~table["passive"].astype(bool)].copy()
```

iii. The trajectory says the local bundle contains the experiment NWBs and metadata, and favors direct, reproducible local loading. It explicitly regards passive sessions as lacking the requested Go/Catch population. It does not justify including both project codes.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` values among retained session groups, converted to strings; each session receives the corresponding integer index.

ii.
```python
subjects = sorted({str(int(g.iloc[0]["mouse_id"])) for _, g in grouped})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_idx[mouse_id])
```

iii. No separate rationale was recorded; the implementation uses the metadata's animal identifier, as expected.

## 1-c. How are the data split into sessions?

i. Rows are grouped by `ophys_session_id`; all experiment planes in a group are combined into one session and sorted deterministically by session and experiment ID.

ii.
```python
table.sort_values(["ophys_session_id", "ophys_experiment_id"], inplace=True)
for session_id, experiments in table.groupby("ophys_session_id", sort=True):
    grouped.append((int(session_id), experiments.copy()))
```

iii. The trajectory explicitly identifies experiments sharing an `ophys_session_id` as simultaneous planes and says they should be merged.

## 1-d. How are the data split into trials?

i. Trial IDs come from `intervals/trials`. Within each retained trial, timepoints are the chronologically sorted active rows in the natural-image presentation table whose `trials_id` matches that trial. Thus a trial is a variable-length sequence of 750 ms presentation intervals, rather than all native ophys frames from trial start to stop.

ii.
```python
for trial_id in trial_ids:
    idx = np.flatnonzero((source_trial_ids == trial_id) & active)
    idx = idx[np.argsort(starts_source[idx])]
    row_groups.append(np.arange(cursor, cursor + len(idx), dtype=np.int64))
```

iii. The agent states that the paper's analysis unit is the 750 ms image-presentation interval and chose that cadence to match published processing and avoid assigning higher-resolution labels during gray periods.

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are kept, with aborted and auto-rewarded trials removed. Outcomes must be exactly one-hot, retained trials must have an active presentation, and sessions need at least two eligible trials and usable pupil data. Passive sessions are removed earlier.

ii.
```python
keep = ((trials["go"][:] | trials["catch"][:])
        & ~trials["aborted"][:] & ~trials["auto_rewarded"][:])
if not np.all(outcomes.sum(axis=1) == 1):
    raise RuntimeError("Every retained trial must have exactly one trial outcome")
if len(trial_ids) < 2:
    ...
```

iii. Go/Catch inclusion and aborted/auto-reward exclusion follow the task. The trajectory says sessions without eye tracking were excluded because pupil diameter would be undefined, and confirms the two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each NWB's inferred calcium-event timestamps and event magnitudes at `processing/ophys/event_detection/{timestamps,data}`.

ii.
```python
event_detection = nwb["processing/ophys/event_detection"]
timestamps = event_detection["timestamps"][:]
aggregated = _interval_sums(timestamps, event_detection["data"], starts)
```

iii. The trajectory calls use of SDK-provided inferred calcium events a key modeling choice based on the paper and methods, instead of raw dF/F.

## 2-b. How is the `neural` data processed?

i. Event magnitudes are summed over each half-open 750 ms presentation interval with prefix sums. Time-by-cell results from simultaneous planes are concatenated across cells, then each trial is transposed to neuron-by-time.

ii.
```python
np.cumsum(event_data, axis=0, dtype=np.float32, out=prefix[1:])
return prefix[right] - prefix[left]
activity = np.concatenate(plane_activity, axis=1)
session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
```

iii. The agent says aggregation at the native task cadence follows the paper and materially reduces the otherwise large frame-level output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit cell-level filtering beyond using the NWB event-detection arrays. Sessions with unusable pupil streams or fewer than two eligible trials are excluded, but sparse or all-zero trial matrices remain.

ii.
```python
aggregated = _interval_sums(timestamps, event_detection["data"], starts)
plane_activity.append(aggregated)
```

iii. The agent relied on released event-detection data. After validation, it characterized all-zero trial warnings as expected for sparse inferred events rather than grounds for filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural sample is aligned to the start of an active image presentation belonging to the trial; plane-specific ophys timestamps delimit `[start, start + 0.75 s)`.

ii.
```python
left = np.searchsorted(timestamps, starts, side="left") - first
right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
return prefix[right] - prefix[left]
```

iii. The trajectory justifies presentation-start alignment as the paper's analysis unit and as a common ophys-timestamp alignment for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 750 ms. Native event samples are explicitly rebinned by summing them within each presentation-plus-gray interval.

ii.
```python
BIN_SECONDS = 0.750
"time_bin_size": BIN_SECONDS * 1000.0,
```

iii. The agent says 750 ms represents a 250 ms image plus 500 ms gray interval and matches the paper's native task cadence, including omissions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived directly from `image_name` in the detected natural-image presentation table, restricted by that table's `active` and `trials_id` fields.

ii.
```python
names_source = np.asarray(_decode_strings(presentations["image_name"][:]), dtype=object)
idx = np.flatnonzero((source_trial_ids == trial_id) & active)
names.append(names_source[idx])
```

iii. The agent chose the presentation table because it supplies direct stimulus labels at the selected analysis cadence.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Unique real image names are sorted globally and omissions, if present, are appended as a separate category. Names are then mapped to integer codes.

ii.
```python
real_images = sorted(name for name in image_names if name != "omitted")
image_values = real_images + (["omitted"] if "omitted" in image_names else [])
image_to_code = {name: idx for idx, name in enumerate(image_values)}
```

iii. The trajectory says omissions are genuine 750 ms task intervals, and the code comment emphasizes stable global string identifiers and a dedicated omission class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The same concatenated presentation rows index image codes and aggregated neural activity within each trial.

ii.
```python
image_codes = np.asarray([image_to_code[name] for name in names], dtype=np.int16)
session_neural.append(activity[rows].T)
session_output.append(np.vstack([image_codes[rows], ...]))
```

iii. The shared presentation-row indexing was intended to guarantee exact alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is taken directly from the presentation table's boolean `is_change` column.

ii.
```python
changes_source = presentations["is_change"][:].astype(bool)
changes.append(changes_source[idx])
```

iii. The agent preferred the SDK presentation label over reconstructing changes from trial timing.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The boolean labels are concatenated and cast to 0/1 `int16`; no additional temporal expansion is performed.

ii.
```python
np.concatenate(changes).astype(np.int16)
```

iii. At 750 ms resolution, one presentation row already represents the flash and following gray interval, so the agent considered a separate window construction unnecessary.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated. The source boolean is encoded directly as `0 = no_change`, `1 = change`.

ii.
```python
changes_source = presentations["is_change"][:].astype(bool)
...
["no_change", "change"],
```

iii. The NWB already provides the categorical change flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change flags and neural sums use the same presentation starts and `rows` for each trial.

ii.
```python
session_output.append(np.vstack([image_codes[rows], changes[rows], ...]))
```

iii. This follows directly from adopting presentation intervals as shared time bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses processed running-wheel speed and timestamps from `processing/running/speed/{data,timestamps}`.

ii.
```python
running_ts = nwb["processing/running/speed/timestamps"][:]
running_raw = nwb["processing/running/speed/data"][:]
```

iii. No specific rationale beyond using the released processed behavioral stream was recorded.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite speed samples are linearly interpolated to reference ophys timestamps, averaged over each 750 ms presentation interval using prefix sums, and discretized using within-session quintile boundaries.

ii.
```python
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
running_codes, running_edges = _quintile_codes(running_values)
```

iii. The agent sought a common ophys clock and categorical labels at the same 750 ms resolution. The metadata explicitly records within-session 20/40/60/80 percentiles.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four within-session quantiles (20%, 40%, 60%, 80%) define five categories; `searchsorted(..., side="right")` assigns codes 0–4.

ii.
```python
edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
codes = np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. Quintiles satisfy the requested five equal-percentile bins; no rationale was given for choosing per-session rather than global thresholds.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated to the representative experiment's ophys timestamps and averaged over the exact presentation starts used for neural event aggregation.

ii.
```python
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
```

iii. The trajectory states that all streams are aligned through ophys timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil ellipse `width` and its timestamps at `acquisition/EyeTracking/pupil_tracking`.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil["timestamps"][:], pupil["width"][:]
```

iii. The code identifies width as the pupil-diameter proxy; NaNs are understood as blink-marked frames from the Allen pipeline.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Nonfinite samples are excluded, the remaining widths are linearly interpolated to ophys timestamps (thereby bridging blinks), averaged per 750 ms interval, and categorized by within-session quintiles.

ii.
```python
pupil_at_ophys = _interpolate_finite(
    pupil["timestamps"][:], pupil["width"][:], reference_ts)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
pupil_codes, pupil_edges = _quintile_codes(pupil_values)
```

iii. The helper docstring says interpolation over finite samples supplies labels without treating blink frames as real measurements.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four within-session quantiles define five percentile categories, assigned with right-sided `searchsorted`.

ii.
```python
edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
codes = np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. Quintiles implement the requested five equal-percentile bins; the session-local scope is recorded in metadata but not otherwise defended.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil width is interpolated onto reference ophys timestamps and averaged over the same presentation intervals used for neural activity.

ii.
```python
pupil_at_ophys = _interpolate_finite(..., reference_ts)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
```

iii. The agent intended the common ophys timebase and interval starts to guarantee alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
outcomes = np.column_stack([trials[name][:][keep] for name in OUTCOME_COLUMNS])
```

iii. These are the canonical mutually exclusive outcomes; the code verifies exactly one is set for every retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` converts the verified one-hot row to codes 0–3, and that static trial code is repeated across all presentation bins in the trial.

ii.
```python
trial_outcomes = outcomes.argmax(axis=1).astype(np.int16)
np.full(timepoints, outcome, dtype=np.int16)
```

iii. Repetition makes the static target compatible with the output matrix's common time dimension; the trajectory's custom validation checks it remains constant.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions lacking pupil width, with fewer than two finite pupil samples, or fewer than two eligible trials are excluded and logged. Behavioral nonfinite values are removed before interpolation, and streams with fewer than two finite samples raise an error. Missing active presentations, ambiguous presentation tables, empty intervals, and invalid outcome rows also raise errors. Output is written atomically through a temporary file.

ii.
```python
if pupil_path not in nwb: excluded.append(...); continue
if np.isfinite(nwb[pupil_path][:]).sum() < 2: excluded.append(...); continue
finite = np.isfinite(source_timestamps) & np.isfinite(source_values)
if finite.sum() < 2: raise RuntimeError(...)
os.replace(temporary, output_path)
```

iii. The trajectory explicitly says missing eye tracking makes a required target undefined, so those sessions are omitted. It otherwise favors failing loudly on structural inconsistencies rather than silently inventing labels.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large event-detection slices from every NWB and aggregating them is the principal I/O/computation cost; serially iterating 171 sessions and their planes dominates conversion.

ii.
```python
values.read_direct(event_data, source_sel=np.s_[first:last, :])
np.cumsum(event_data, axis=0, dtype=np.float32, out=prefix[1:])
```

iii. The trajectory inspected the 247 GB local bundle and estimated output size before choosing interval aggregation, indicating that full neural-array I/O and volume were the main performance concerns.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. `_session_presentations` scans the full presentation arrays once per trial with `np.flatnonzero`; grouping/sorting once by trial ID would avoid repeated scans. The image-name list comprehension and some metadata loops could also be vectorized, though they are minor. Plane/session loops are appropriate because they involve separate files.

ii.
```python
for trial_id in trial_ids:
    idx = np.flatnonzero((source_trial_ids == trial_id) & active)
image_codes = np.asarray([image_to_code[name] for name in names], dtype=np.int16)
```

iii. The agent did not explicitly discuss vectorization. It did implement prefix sums specifically to vectorize interval aggregation and avoid a costly interval-by-cell loop.

## 9-c. What processing does the code repeat multiple times?

i. Each retained representative NWB is opened and its trials/presentations are parsed during the screening pass and again during conversion. Files are also reopened per plane for neural data. The screening pass decodes presentation names once to build the global vocabulary and the conversion pass decodes them again.

ii.
```python
with h5py.File(representative, "r") as nwb:
    trial_ids, _ = _eligible_trial_ids(nwb)
    _, _, names, _ = _session_presentations(nwb, trial_ids)
...
with h5py.File(representative, "r") as nwb:
    trial_ids, trial_outcomes = _eligible_trial_ids(nwb)
    starts, row_groups, names, changes = _session_presentations(nwb, trial_ids)
```

iii. No explicit justification was recorded. The first pass establishes exclusions and global category mappings before final allocation, trading repeated lightweight metadata reads for simpler assembly and lower retained memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The first-pass presentation arrays other than unique image names are discarded. `omitted` is required only to identify the presentation group, not read as a value. Several detailed metadata fields and saved bin edges are not used by the decoder itself, though they aid provenance. The empty decoder-input arrays are required by the target schema rather than analytically useful.

ii.
```python
_, _, names, _ = _session_presentations(nwb, trial_ids)
image_names.update(names.tolist())
...
"neurons_per_experiment": neuron_counts,
"running_quintile_edges_cm_per_s": running_edges.tolist(),
```

iii. The trajectory does not identify discarded work. Most extra metadata appears intended for interpretability and validation rather than downstream model features.
