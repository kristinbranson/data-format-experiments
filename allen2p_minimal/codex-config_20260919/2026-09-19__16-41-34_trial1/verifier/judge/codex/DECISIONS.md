# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads local Allen release files directly from `/app/data/visual-behavior-ophys-1.1.0` using `h5py` and `pandas`, not the Allen SDK cache. It discovers experiment NWBs by globbing `behavior_ophys_experiment_*.nwb`, reads `project_metadata/ophys_experiment_table.csv`, keeps rows whose `ophys_experiment_id` has a local NWB file, drops rows marked `passive`, and later reads each session's NWB content directly from disk.

ii.
```python
files = {
    _experiment_id(path): path
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
}
...
table = pd.read_csv(metadata_path)
table = table[table["ophys_experiment_id"].isin(files)].copy()
table = table[~table["passive"].astype(bool)].copy()
```

iii. In the trajectory, the AI justified this as working from the supplied local NWB bundle and project metadata, noting the bundle contained 284 NWBs and that it was choosing a direct NWB-based pipeline rather than relying on the SDK cache. It also said it was resolving how simultaneous experiments should be combined into sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined as unique `mouse_id` values taken from the metadata rows of retained sessions, converted to sorted strings.

ii.
```python
subjects = sorted({str(int(g.iloc[0]["mouse_id"])) for _, g in grouped})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
mouse_id = str(int(experiments.iloc[0]["mouse_id"]))
subject_idx.append(subject_to_idx[mouse_id])
```

iii. The trajectory did not add a separate justification beyond treating `mouse_id` as the session's animal identifier.

## 1-c. How are the data split into sessions?

i. Sessions are defined by `ophys_session_id`. All experiments sharing one `ophys_session_id` are grouped together and treated as simultaneous planes within one recording session.

ii.
```python
table.sort_values(["ophys_session_id", "ophys_experiment_id"], inplace=True)
...
for session_id, experiments in table.groupby("ophys_session_id", sort=True):
    ...
grouped.append((int(session_id), experiments.copy()))
```

iii. The AI explicitly justified this in the trajectory: it said simultaneous multi-plane sessions should be merged by `ophys_session_id`, and later repeated that experiments sharing an `ophys_session_id` were concatenated as neurons in one recording session.

## 1-d. How are the data split into trials?

i. The AI first selects retained NWB trials from `intervals/trials`, but it does not use the full `start_time` to `stop_time` window as the trial signal. Instead, each retained trial is represented as the ordered sequence of active stimulus presentations belonging to that trial. Each timepoint in the saved trial is therefore one 750 ms image-presentation interval rather than one ophys frame.

ii.
```python
def _eligible_trial_ids(nwb: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    trials = nwb["intervals/trials"]
    keep = (
        (trials["go"][:] | trials["catch"][:])
        & ~trials["aborted"][:]
        & ~trials["auto_rewarded"][:]
    )
    trial_ids = trials["id"][:][keep].astype(np.int64)
```

```python
for trial_id in trial_ids:
    idx = np.flatnonzero((source_trial_ids == trial_id) & active)
    ...
    row_groups.append(np.arange(cursor, cursor + len(idx), dtype=np.int64))
```

iii. The trajectory gives the clearest justification here. The AI said the “actual analysis unit” was the 750 ms image-presentation interval, that each trial should become a variable-length sequence of flashes, and that this would avoid “inventing a higher-resolution label during the gray portion.”

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only Go or Catch trials that are not aborted and not auto-rewarded. It also requires each retained trial to map to exactly one of the four outcome columns, requires at least one active stimulus presentation for each retained trial, and drops sessions with fewer than two eligible trials. Separately, it excludes sessions missing usable pupil data before trial extraction.

ii.
```python
keep = (
    (trials["go"][:] | trials["catch"][:])
    & ~trials["aborted"][:]
    & ~trials["auto_rewarded"][:]
)
...
if not np.all(outcomes.sum(axis=1) == 1):
    raise RuntimeError("Every retained trial must have exactly one trial outcome")
```

```python
if pupil_path not in nwb:
    excluded.append({...})
    continue
if np.isfinite(nwb[pupil_path][:]).sum() < 2:
    excluded.append({...})
    continue
...
if len(trial_ids) < 2:
    excluded.append({...})
    continue
```

iii. The AI explicitly justified excluding aborted and auto-rewarded trials and requiring Go/Catch trials. It also said three active sessions were excluded because they had no eye-tracking stream, making the required pupil output undefined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved `neural` data is derived from NWB `processing/ophys/event_detection/data` with corresponding `processing/ophys/event_detection/timestamps`, not from `dff_traces`.

ii.
```python
with h5py.File(files[experiment_id], "r") as nwb:
    event_detection = nwb["processing/ophys/event_detection"]
    timestamps = event_detection["timestamps"][:]
    aggregated = _interval_sums(timestamps, event_detection["data"], starts)
```

iii. The trajectory explicitly justified this choice: the AI said a “key modeling choice” was to use inferred calcium events rather than raw `ΔF/F`.

## 2-b. How is the `neural` data processed?

i. Neural event magnitudes are summed within each 750 ms presentation interval using prefix sums, separately for each plane. The per-plane interval matrices are then concatenated across neurons so one session contains all neurons from all planes.

ii.
```python
prefix = np.empty((len(event_data) + 1, event_data.shape[1]), dtype=np.float32)
prefix[0] = 0.0
np.cumsum(event_data, axis=0, dtype=np.float32, out=prefix[1:])
left = np.searchsorted(timestamps, starts, side="left") - first
right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
return prefix[right] - prefix[left]
```

```python
plane_activity.append(aggregated)
...
activity = np.concatenate(plane_activity, axis=1)
...
session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
```

iii. The AI justified this by saying calcium-event magnitudes should be aggregated over the native 750 ms image-presentation interval, matching the cadence it inferred from the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level or trace-level quality filtering in the script. All cells present in `event_detection/data` for retained experiments are included. Quality control is applied only at the session/trial level.

ii.
```python
for experiment_id, region in zip(
    experiment_ids, experiments["targeted_structure"].astype(str)
):
    with h5py.File(files[experiment_id], "r") as nwb:
        event_detection = nwb["processing/ophys/event_detection"]
        ...
    plane_activity.append(aggregated)
```

iii. The trajectory did not give a separate neuron-QC justification beyond reporting that all-zero intervals were expected for sparse event traces and were not treated as a formatting failure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 750 ms active stimulus-presentation interval, not to trial start and not kept at per-frame resolution. The interval starts come from the natural-image presentation table.

ii.
```python
starts_source = presentations["start_time"][:]
...
left = np.searchsorted(timestamps, starts, side="left") - first
right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
```

iii. The AI explicitly justified this as aligning all streams to the “start of each 750 ms image-presentation interval.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 750 ms bins. The code rebins neural and behavioral streams from their native timestamps into one value per 750 ms presentation interval.

ii.
```python
BIN_SECONDS = 0.750
...
"time_bin_size": BIN_SECONDS * 1000.0,
```

```python
aggregated = _interval_sums(timestamps, event_detection["data"], starts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
```

iii. The trajectory repeatedly justified this as adopting the “native task cadence” of one 250 ms image plus 500 ms gray period.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `Image identity` is derived from the natural-image presentation table’s `image_name` field, not from `initial_image_name` and `change_image_name` in the trials table.

ii.
```python
presentations = _presentation_group(nwb)
...
names_source = np.asarray(_decode_strings(presentations["image_name"][:]), dtype=object)
```

iii. The AI justified this by saying stimulus labels should come directly from the presentation table at the same 750 ms cadence used for neural aggregation.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script decodes NWB string values, gathers all presentation `image_name` values across retained sessions, sorts real image labels, optionally appends `"omitted"` as a separate final category, and maps each presentation label to an integer code.

ii.
```python
real_images = sorted(name for name in image_names if name != "omitted")
image_values = real_images + (["omitted"] if "omitted" in image_names else [])
image_to_code = {name: idx for idx, name in enumerate(image_values)}
...
image_codes = np.asarray([image_to_code[name] for name in names], dtype=np.int16)
```

iii. The AI explicitly justified treating omissions as their own image-identity category and said this matched the 750 ms omission intervals in the paper’s analysis cadence.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned one-to-one with the same presentation intervals used for neural aggregation. Each saved neural time bin has one corresponding image code from the same `rows` slice.

ii.
```python
for rows, outcome in zip(row_groups, trial_outcomes):
    ...
    session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
    session_output.append(np.vstack([
        image_codes[rows],
        changes[rows],
        ...
    ]).astype(np.int16, copy=False))
```

iii. The AI justified this as avoiding any mismatch between neural bins and stimulus labels by using the same presentation intervals for both.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `Image change` is derived from the presentation table’s `is_change` flag, which marks whether each active presentation is a true image change.

ii.
```python
changes_source = presentations["is_change"][:].astype(bool)
...
np.concatenate(changes).astype(np.int16)
```

iii. The AI said “stimulus/change labels come directly from the SDK’s presentation table,” which is the recorded justification for using presentation-level change annotations.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code does almost no additional processing: it casts the presentation-table `is_change` values to integers, concatenates them in retained-trial order, and slices them by each trial’s presentation indices.

ii.
```python
changes.append(changes_source[idx])
...
return (
    np.concatenate(starts).astype(np.float64),
    row_groups,
    np.concatenate(names),
    np.concatenate(changes).astype(np.int16),
)
```

iii. The trajectory indicates the AI wanted change labels “directly” from the presentation table at the same cadence as the saved samples.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as an already-binary variable with categories `0 = no_change` and `1 = change`. No further thresholding is applied.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. The trajectory did not add any separate thresholding justification beyond treating `is_change` as the native binary label.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change uses the same presentation-interval indexing as the neural data. Each neural interval bin receives the corresponding `is_change` value for that presentation.

ii.
```python
session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
session_output.append(np.vstack([
    image_codes[rows],
    changes[rows],
    running_codes[rows],
    pupil_codes[rows],
    ...
]))
```

iii. The AI justified this through the same presentation-interval alignment argument it used for image identity and neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived directly from NWB `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
running_ts = nwb["processing/running/speed/timestamps"][:]
running_raw = nwb["processing/running/speed/data"][:]
```

iii. The trajectory did not add a separate running-specific justification beyond aligning behavioral streams to ophys timestamps.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the ophys event timestamp grid, averaged over each 750 ms presentation interval, and then converted to discrete quintile codes.

ii.
```python
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
...
running_codes, running_edges = _quintile_codes(running_values)
```

iii. The metadata and trajectory justify this as aligning behavior to the same ophys-timestamp presentation intervals used everywhere else.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five within-session quintile bins using the 20th, 40th, 60th, and 80th percentiles of the retained interval-mean running values from that session.

ii.
```python
def _quintile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int16)
    return codes, edges
```

iii. The only explicit justification appears in metadata, which says running speed uses within-session percentile boundaries over retained intervals.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to the ophys timestamp reference used for the session, then averaged over the same presentation intervals used to aggregate neural events. The resulting interval values are sliced with the same `rows` indices as the neural data.

ii.
```python
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
...
session_output.append(np.vstack([
    image_codes[rows],
    changes[rows],
    running_codes[rows],
    ...
]))
```

iii. The metadata states that behavioral streams were “linearly interpolated to ophys timestamps, then averaged per interval.”

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from NWB `acquisition/EyeTracking/pupil_tracking/width` and its `timestamps`.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil["timestamps"][:], pupil["width"][:]
```

iii. The trajectory justified excluding sessions without this stream because pupil diameter was a required decoder output.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script uses only finite pupil samples, linearly interpolates pupil width to the ophys event timestamp grid, averages it over each 750 ms presentation interval, and discretizes the resulting interval means into quintile codes.

ii.
```python
pupil_at_ophys = _interpolate_finite(
    pupil["timestamps"][:], pupil["width"][:], reference_ts
)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
...
pupil_codes, pupil_edges = _quintile_codes(pupil_values)
```

iii. The metadata says pupil diameter was “blink-filtered pupil ellipse width linearly interpolated to ophys timestamps, then averaged per interval,” although the actual code only filters to finite values rather than reading a separate blink flag.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five within-session quintile bins using the 20th, 40th, 60th, and 80th percentiles of retained interval-mean pupil width values.

ii.
```python
running_codes, running_edges = _quintile_codes(running_values)
pupil_codes, pupil_edges = _quintile_codes(pupil_values)
```

iii. The explicit justification is only in metadata, which says pupil diameter uses within-session percentile boundaries over retained intervals.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned exactly like running speed: interpolate to the ophys reference timestamps, average within the same 750 ms presentation intervals, then index those interval-level codes with the same `rows` used for neural bins.

ii.
```python
pupil_at_ophys = _interpolate_finite(
    pupil["timestamps"][:], pupil["width"][:], reference_ts
)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
...
session_output.append(np.vstack([
    ...,
    pupil_codes[rows],
    ...
]))
```

iii. The AI used the same interval-alignment rationale for all behavioral outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean NWB trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcomes = np.column_stack([trials[name][:][keep] for name in OUTCOME_COLUMNS])
```

iii. The trajectory did not add a separate justification, but the code assumes these four columns are the canonical mutually exclusive trial outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four booleans are stacked per retained trial, checked to ensure exactly one is true, converted to integer class labels by `argmax`, and then repeated across all saved presentation bins within that trial.

ii.
```python
outcomes = np.column_stack([trials[name][:][keep] for name in OUTCOME_COLUMNS])
if not np.all(outcomes.sum(axis=1) == 1):
    raise RuntimeError("Every retained trial must have exactly one trial outcome")
return trial_ids, outcomes.argmax(axis=1).astype(np.int16)
```

```python
for rows, outcome in zip(row_groups, trial_outcomes):
    ...
    np.full(timepoints, outcome, dtype=np.int16),
```

iii. The recorded justification is implicit: the AI treated trial outcome as a static per-trial label that should be copied across the saved bins of that trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil data is handled by excluding the session entirely if pupil tracking is absent or has fewer than two finite samples. Within usable sessions, missing samples are handled by interpolating only over finite values with `np.interp`, which also extrapolates boundary values. The code raises runtime errors if a retained trial has no active presentations or if an interval would contain zero ophys timestamps.

ii.
```python
if pupil_path not in nwb:
    excluded.append({...})
    continue
if np.isfinite(nwb[pupil_path][:]).sum() < 2:
    excluded.append({...})
    continue
```

```python
finite = np.isfinite(source_timestamps) & np.isfinite(source_values)
if finite.sum() < 2:
    raise RuntimeError("Behavioral stream has fewer than two finite samples")
return np.interp(
    target_timestamps,
    source_timestamps[finite],
    source_values[finite],
)
```

iii. The trajectory explicitly justified dropping three sessions with no eye-tracking stream because otherwise the required pupil-diameter target would be undefined.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the repeated large NWB reads and the per-session aggregation work: scanning session metadata, opening representative NWBs to test inclusion and collect presentation labels, reopening them for running/pupil streams, and opening every plane’s event-detection matrix to aggregate interval sums.

ii.
```python
for session_id, experiments in table.groupby("ophys_session_id", sort=True):
    representative = files[int(experiments.iloc[0]["ophys_experiment_id"])]
    with h5py.File(representative, "r") as nwb:
        ...
```

```python
for experiment_id, region in zip(
    experiment_ids, experiments["targeted_structure"].astype(str)
):
    with h5py.File(files[experiment_id], "r") as nwb:
        event_detection = nwb["processing/ophys/event_detection"]
        ...
```

iii. In the trajectory, the AI highlighted the very large local bundle size and tracked long-running conversion over many sessions, which supports the interpretation that file I/O and interval aggregation dominated runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes interval aggregation with prefix sums, but several loops remain: iterating over `trial_ids` in `_session_presentations`, iterating over planes in each session, mapping image names to codes with a Python list comprehension, and the final per-trial loop that builds `session_neural` and `session_output`.

ii.
```python
for trial_id in trial_ids:
    idx = np.flatnonzero((source_trial_ids == trial_id) & active)
    ...
```

```python
for rows, outcome in zip(row_groups, trial_outcomes):
    ...
    session_neural.append(...)
    session_output.append(...)
```

iii. The trajectory does not discuss these loops explicitly, but the presence of prefix-sum helpers suggests the AI was already trying to vectorize the heaviest interval computations.

## 9-c. What processing does the code repeat multiple times?

i. The script repeats some work across passes. It opens a representative NWB once during session pre-screening and again during the main conversion pass. It also computes presentation-level labels once in the pre-screening pass to collect global image names and again in the main pass to build the actual outputs.

ii.
```python
for session_id, experiments in table.groupby("ophys_session_id", sort=True):
    representative = files[int(experiments.iloc[0]["ophys_experiment_id"])]
    with h5py.File(representative, "r") as nwb:
        ...
        _, _, names, _ = _session_presentations(nwb, trial_ids)
        image_names.update(names.tolist())
```

```python
with h5py.File(representative, "r") as nwb:
    trial_ids, trial_outcomes = _eligible_trial_ids(nwb)
    starts, row_groups, names, changes = _session_presentations(nwb, trial_ids)
```

iii. The trajectory does not explicitly justify this repetition; it appears to be a consequence of doing one pass for dataset discovery/category collection and a second pass for full conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra pre-pass processing and metadata construction that the downstream decoder does not use directly. Examples include reading sessions just to build the global image-name set, collecting detailed `session_info`, storing per-session quintile edges, and assembling `excluded_sessions` explanations.

ii.
```python
session_info.append({
    "ophys_session_id": session_id,
    ...
    "running_quintile_edges_cm_per_s": running_edges.tolist(),
    "pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
})
```

```python
"metadata": {
    ...
    "excluded_sessions": excluded,
    "session_info": session_info,
},
```

iii. The trajectory does not present this as a deliberate optimization target; it mostly reflects a converter written to preserve explanatory metadata even though the decoder only consumes the main arrays and label vocabularies.
