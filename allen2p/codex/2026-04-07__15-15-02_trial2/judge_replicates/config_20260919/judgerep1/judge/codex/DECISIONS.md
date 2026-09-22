# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local release directly with `h5py`. It joins NWB filenames to `ophys_experiment_table.csv`, drops passive experiments and experiments without required eye-tracking groups, then performs two passes over the remaining files: a metadata/behavior pass and a conversion pass that also loads neural events.

ii.
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {int(path.stem.split("_")[-1]): path
            for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
...
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
raw = read_session_raw(session, load_events=False)  # pass 1
...
raw = read_session_raw(session)                     # pass 2
```

iii. The notes say the installed `pynwb/hdmf` stack could not instantiate these NWBs, so direct HDF5 reads were used while following SDK field semantics. The local files were treated as the available curated subset. Passive sessions were excluded because trial outcome is required, and files without pupil data were excluded because pupil diameter is required.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s from experiment metadata. A subject is registered when its first retained experiment is converted, and each output session receives its subject index.

ii.
```python
mouse_id=str(int(row.mouse_id)),
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
...
data["subject_idx"].append(subject_idx)
```

iii. The notes identify metadata `mouse_id` as the stable mouse identifier and plan a global subject list plus a per-session index.

## 1-c. How are the data split into sessions?

i. Each local `behavior_ophys_experiment_<id>.nwb` file/`ophys_experiment_id` is treated as one decoder session. Experiments sharing an `ophys_session_id` are not merged.

ii.
```python
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. The agent explicitly resolved the experiment/session discrepancy by treating each NWB as a decoder session because neural traces are experiment-specific. It noted that this leaves experiments from one behavioral/ophys session linked only indirectly through subject/session metadata.

## 1-d. How are the data split into trials?

i. Trial boundaries come from `intervals/trials/start_time` and `stop_time`. For every retained row, a half-open 30 Hz grid is made from start through just before stop; all streams are evaluated on that grid.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
...
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The notes say SDK/NWB trial boundaries are authoritative and the full start-to-stop window retains pre- and post-change behavior.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained only if it is explicitly go or catch and is neither aborted nor auto-rewarded. Sessions with fewer than two retained trials are skipped. Passive sessions and entire files lacking pupil groups are filtered earlier.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. This follows the task's explicit go/catch inclusion and aborted/auto-rewarded exclusion. The two-trial rule is required by decoder validation. The extra file-level exclusions were justified by the need for meaningful outcome and pupil targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB precomputed calcium-event matrix and its event-detection timestamps, not from dF/F.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. The strategy paper reportedly used detected calcium events, so the agent judged events closer to the paper's analysis than dF/F and avoided recomputing either signal.

## 2-b. How is the `neural` data processed?

i. Events are optionally filtered by the cell table's `valid_roi`, linearly interpolated across time to each trial's common 30 Hz grid, transposed to neuron-by-time, and cast to float32.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes cite released ROI QC and the paper's interpolation of event-triggered responses to common 30 Hz timestamps. No additional normalization or event detection is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent relies primarily on released-file QC and conditionally applies `valid_roi`. It does not impose activity, SNR, or other ad hoc neuron filters.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say the SDK normally excludes invalid ROIs and released cell tables already reflect segmentation, duplicate/union, and trace QC, so additional filtering would diverge from the reference pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start. Absolute 30 Hz timestamps begin at each trial's `start_time`; neural events are interpolated onto those timestamps through `stop_time`.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
```

iii. The full trial window was selected so all requested time-varying outputs could share one alignment and include both pre- and post-change epochs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses a fixed 1/30 s grid (33.333 ms). All signals, including neural events from native approximately 11 or 31 Hz recordings, are linearly resampled to this grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. The target requires a shared bin size, local recordings have mixed native rates, and the paper used common 30 Hz interpolation; the agent therefore selected 30 Hz rather than retaining native frames.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the chosen active task stimulus-presentation table's `start_time`, `stop_time`, `image_name`, and `omitted` columns.

ii.
```python
stim = read_interval_table(stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted",
     "trials_id", "active", "flashes_since_change"])
```

iii. The agent chose presentation intervals because they describe what is actually on screen at each instant, including flashes, gray inter-stimulus intervals, and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global category vocabulary is constructed as `gray` plus sorted nonempty, non-omitted image names. Each query time defaults to gray and is assigned an image code only when it lies inside a non-omitted presentation interval.

ii.
```python
image_values = ["gray"] + sorted(image_names)
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. The explicit gray class was justified as necessary for a complete time-varying screen identity because the task includes 500 ms gray periods and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are evaluated directly at the same absolute 30 Hz `grid` used for interpolated neural activity.

ii.
```python
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. A common query grid guarantees equal lengths and timestamp alignment across neural and stimulus outputs.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from stimulus-presentation `start_time`, `stop_time`, `is_change`, and `omitted`.

ii.
```python
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
```

iii. The agent chose the raw presentation's change flag to identify a real changed-image display and naturally leave catch trials at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Query times are associated with their most recent presentation start, tested against that presentation's stop, and labeled from `is_change & ~omitted` during the presentation interval; all other times are zero.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. Notes say this presentation-window label is less degenerate than a one-bin impulse while remaining faithful to the changed stimulus.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: false becomes category 0 (`no_change`) and true becomes 1 (`change`). No numeric threshold is learned.

ii.
```python
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
...
["no_change", "change"],
```

iii. The source field is boolean, so direct categorical casting is sufficient.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change intervals are sampled on exactly the same trial grid as neural events.

ii.
```python
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The common absolute time grid provides frame-for-frame alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed/data` and the corresponding timestamps.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. This is the SDK's processed/filtered speed stream, which the notes say matches reference running-wheel preprocessing.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to each 30 Hz trial grid. Four global percentile edges (20, 40, 60, 80) are computed from all retained trial samples in pass 1 and applied in pass 2.

ii.
```python
return np.interp(query_t, source_t, source_values).astype(np.float32)
...
percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global quintiles give shared categories and approximately balanced classes; linear interpolation synchronizes speed to the neural grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` maps values around the four global quintile edges into integer categories 0 through 4. Non-increasing duplicate edges are nudged upward by `1e-6`.

ii.
```python
for i in range(1, len(percentiles)):
    if percentiles[i] <= percentiles[i - 1]:
        percentiles[i] = percentiles[i - 1] + 1e-6
...
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Equal-percentile bins were required, and edge repair prevents collapsed categories in degenerate distributions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly to the same absolute 30 Hz timestamps used for neural interpolation.

ii.
```python
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. The notes cite hardware-synchronized clocks and the common grid as the alignment mechanism.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses pupil-tracking ellipse `width` and `height`, plus eye-tracking timestamps. Diameter is defined as the larger of width and height.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The mapping plan calls `max(width, height)` pupil diameter and assumes invalid/blink samples in released tracking arrays are represented as non-finite values.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite diameter samples are removed, remaining samples are linearly interpolated to 30 Hz, and global quintile edges are computed and applied like running speed. One valid value is broadcast; no valid values cause the session read to fail.

ii.
```python
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
if valid.sum() == 1:
    return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. The notes describe interpolation across small blink-related gaps after reference invalid-frame masking, followed by global quintiles for consistent categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global nan-percentile edges at 20/40/60/80% define integer bins 0–4; duplicate edges are nudged upward.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. This mirrors running-speed discretization and yields five global equal-percentile classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Valid pupil samples are interpolated from eye timestamps to the exact 30 Hz trial grid used for neural data.

ii.
```python
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Common synchronized timestamps and a shared query grid provide alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial table's boolean `hit`, `miss`, `false_alarm`, and `correct_reject` fields.

ii.
```python
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. These are the SDK's canonical mutually exclusive outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map in fixed order to 0–3 and the selected code is broadcast across every time bin in the trial. A retained trial with no label raises an error.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting makes the static output compatible with the uniform time-varying output matrix while preserving one label per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Files absent locally are excluded by the filename/metadata join. Files without required pupil/eye groups are excluded. Non-finite pupil samples are ignored and interpolated over; a single sample is broadcast and zero samples raise. Duplicate percentile edges are repaired. ROI filtering is conditional on compatible dimensions. Unlike the reference, conversion has no per-session `try/except`, so an unexpected malformed retained file aborts the run.

ii.
```python
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. Notes frame missing-eye exclusion as necessary for the required pupil target and interpolation as appropriate for small invalid/blink gaps. Explicit validation is favored over silently fabricating an entirely missing pupil stream.

## 9-a. What are the most time-consuming steps of the code?

i. The expected dominant cost is pass-2 neural-event loading and trial-by-trial interpolation across every neuron. Every file is also opened/read twice, once for global statistics and again for conversion.

ii.
```python
raw = read_session_raw(session, load_events=False)
...
raw = read_session_raw(session)
...
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The notes explicitly identify neural interpolation as dominant and estimate runtime by total neuron-bins. Direct HDF5 was selected to reduce SDK construction overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code loops over sessions and trials necessarily, but several Python loops could be reduced: pass-1 trial interpolation, per-trial pass-2 setup, image-name assignment inside `stimulus_identity_codes`, and outcome-label checking. Neural interpolation is already vectorized over neurons.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The agent notes that trial-level neural interpolation was vectorized over neurons, which removes the most expensive potential Python loop. It did not specifically justify the remaining small categorical loops.

## 9-c. What processing does the code repeat multiple times?

i. Each retained NWB is read twice. In both passes, trial grids are rebuilt and running/pupil streams are interpolated for every trial. Trial keep masks, stimulus tables, and timestamps are also parsed twice.

ii.
```python
# pass 1
raw = read_session_raw(session, load_events=False)
grid = session_grid(start, stop)
running_values.append(interpolate_vector(..., grid))
pupil_values.append(interpolate_pupil(..., grid))
...
# pass 2
raw = read_session_raw(session)
grid = session_grid(start, stop)
running_cont = interpolate_vector(..., grid)
pupil_cont = interpolate_pupil(..., grid)
```

iii. The two-pass design was chosen to keep memory bounded while calculating global edges, accepting repeat I/O and behavioral interpolation rather than retaining all first-pass arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass 1 computes and stores `valid_trial_counts` and per-session native frame intervals only for logging. `read_session_raw` also parses several stimulus fields (`trials_id`, `active`, `flashes_since_change`) that conversion never uses after selecting the table. When plotting is disabled, no plot-only work is performed.

ii.
```python
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
...
["start_time", "stop_time", "image_name", "is_change", "omitted",
 "trials_id", "active", "flashes_since_change"]
```

iii. These values support sanity/progress reporting and schema inspection, but they do not enter the saved decoder arrays; the notes emphasize validation, timing, and bottleneck detection as their purpose.
