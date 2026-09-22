# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates locally present `behavior_ophys_experiment_*.nwb` files, joins them to `ophys_experiment_table.csv`, removes passive experiments and experiments without required eye-tracking groups, and reads each NWB directly with `h5py`. It makes two passes: one without events for global statistics and one with events for conversion.

ii.
```python
file_map = {int(path.stem.split("_")[-1]): path
            for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
raw = read_session_raw(session, load_events=False)  # pass 1
...
raw = read_session_raw(session)                     # pass 2
```

iii. The notes say the local files are a curated subset and direct HDF5 was chosen because the installed `pynwb/hdmf` stack could not instantiate the NWBs through the SDK. Active sessions were selected because trial outcome is required; sessions without pupil data were dropped.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` strings encountered among retained experiment files; `subject_idx` is assigned on first encounter.

ii.
```python
mouse_id=str(int(row.mouse_id))
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The agent justified mouse ID as the released unique animal identifier and preserved it as a string.

## 1-c. How are the data split into sessions?

i. Every `ophys_experiment_id`/NWB is treated as a separate decoder session. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
SessionInfo(ophys_experiment_id=int(row.ophys_experiment_id),
            path=file_map[int(row.ophys_experiment_id)], ...)
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. The notes acknowledge that 284 experiment files represent only 247 ophys sessions, but choose experiment-level sessions because neural traces are experiment-specific and one NWB corresponds to one experiment.

## 1-d. How are the data split into trials?

i. Trial boundaries come from `intervals/trials`; retained trials are sliced from `start_time` (inclusive grid origin) to `stop_time` on a newly constructed 30 Hz grid, so lengths vary.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The agent says SDK/NWB trial bounds are the canonical experimental definitions and that the full interval retains pre-change and response periods.

## 1-e. How are trials filtered based on quality controls?

i. It keeps explicit go or catch trials and rejects aborted and auto-rewarded trials. A converted experiment must have at least two retained trials. Passive experiments and experiments lacking eye tracking are also removed upstream.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. The trial rule directly follows the task. Passive viewing was deemed incompatible with meaningful operant outcomes, and eye tracking was required to produce the pupil target.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the NWB `processing/ophys/event_detection/data` array and its timestamps, not from dF/F.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. The strategy paper reportedly used detected calcium events, so the agent considered precomputed events closer to that analysis than dF/F.

## 2-b. How is the `neural` data processed?

i. Event traces are optionally filtered by `valid_roi`, linearly interpolated across time for all neurons onto each trial's 30 Hz grid, transposed to neuron-by-time, and cast to float32.

ii.
```python
if valid_roi.sum() != events.shape[1]:
    events = events[:, valid_roi]
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The agent says released events already embody the reference detection pipeline and that common 30 Hz interpolation follows event-triggered analyses in the paper while satisfying a common bin size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. When the event columns match the cell table and `valid_roi` exists, invalid ROIs are removed; no further activity- or trace-based filtering is added.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes state SDK/release QC already excludes non-cell, duplicate, union, and problematic ROIs, so ad hoc filtering would diverge from reference curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start: each grid begins at the trial's `start_time` and runs until `stop_time`; metadata records `temporal_alignment_event = "trial start"` and `off_start = 0`.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
```

iii. This uses synchronized NWB times and retains the complete trial needed for time-varying outputs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output is fixed at 1/30 s (33.333 ms). All neural and behavioral streams are linearly resampled to that grid; this is interpolation, not count aggregation.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
...
return start + np.arange(n_bins, dtype=np.float64) * dt
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. The agent chose 30 Hz because local experiments mix roughly 11 and 31 Hz, eye/behavior are near 30 Hz, and the paper used common 30 Hz timestamps.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It uses the selected active stimulus-presentation table's `start_time`, `stop_time`, `image_name`, and `omitted` fields, rather than trial `initial_image_name`/`change_image_name`.

ii.
```python
stim = read_interval_table(stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted", ...])
```

iii. The agent wanted actual flashed identity, explicit gray interstimulus periods, and omissions represented over the full trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is `['gray'] + sorted(non-omitted image names)`. Each 30 Hz sample defaults to gray and is assigned an image code only when it lies inside a non-omitted presentation interval.

ii.
```python
image_values = ["gray"] + sorted(image_names)
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. The explicit gray class was justified by 500 ms gray periods and omissions, making identity truly describe what is on screen.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus intervals are queried at exactly the same per-trial 30 Hz `grid` used for neural interpolation.

ii.
```python
neural_trial = interpolate_matrix(..., grid).T
...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Shared synchronized timestamps and a shared query grid were intended to eliminate index offsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from stimulus presentation `start_time`, `stop_time`, `is_change`, and `omitted`.

ii.
```python
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
```

iii. The agent preferred the presentation table's direct change annotation to reconstructing the event from trial fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every grid time, the code locates the most recent presentation and checks whether the time is before its stop; the label is the presentation's non-omitted `is_change` value.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
in_interval = query_t[valid] < stops[idx_valid]
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. This was described as marking the changed-image presentation rather than a one-bin impulse, producing a less degenerate target.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already Boolean: 1 only inside a non-omitted presentation whose `is_change` flag is true, otherwise 0. No numerical threshold is learned.

ii.
```python
codes = np.zeros(query_t.shape, dtype=np.int64)
changed = is_change[sub_idx] & (~omitted[sub_idx])
```

iii. The two output values are explicitly `no_change` and `change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is evaluated on the same trial grid as neural interpolation.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The common synchronized grid is the agent's alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed/data` and its `timestamps`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. The notes identify this as the SDK's filtered running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated onto each 30 Hz trial grid. A first pass pools all retained-trial samples to compute global quintile edges, which are applied in pass two.

ii.
```python
running_values.append(interpolate_vector(..., grid))
running_edges = robust_quintile_edges(running_all)
...
running_cont = interpolate_vector(..., grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global percentiles provide shared definitions and approximately balanced classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th global percentiles form five bins numbered 0-4; tied edges are nudged upward by `1e-6`, and `np.digitize(..., right=False)` assigns labels.

ii.
```python
percentiles = np.nanpercentile(values, [20, 40, 60, 80])
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. This directly implements the requested five equal-percentile categories while handling duplicate quantiles.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is independently linearly interpolated at the exact same grid times as neural events.

ii.
```python
neural_trial = interpolate_matrix(..., grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. The notes rely on hardware-synchronized NWB timestamps and the common grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It reads pupil ellipse `width` and `height` plus eye-tracking timestamps, and defines diameter as their elementwise maximum. It does not read the NWB `likely_blink` field.

ii.
```python
pupil_width = np.asarray(...["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(...["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(...["eye_tracking"]["timestamps"], dtype=np.float64)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The notes call max(width, height) a diameter measure and claim blink-masked values from reference eye processing, although the conversion code itself only recognizes non-finite diameter values.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite diameter samples are dropped; the remaining values are linearly interpolated to trial grids. All retained samples supply global quintile edges, and pass two digitizes them.

ii.
```python
valid = np.isfinite(pupil_diameter)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The agent intended interpolation to fill small blink/missing gaps and global quintiles to yield consistent balanced classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses the same global 20/40/60/80 percentile and tie-adjustment procedure as running speed, yielding labels 0-4.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. This follows the requested five equal-percentile bins globally rather than defining incompatible per-session bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated from eye timestamps at the shared per-trial 30 Hz grid.

ii.
```python
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Shared synchronized clock times and grid queries are the stated alignment basis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the four trial-table Boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. These are the SDK's mutually exclusive canonical outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true flag is mapped in fixed order to codes 0-3; absence of any valid flag raises an error. The code is broadcast across all time bins of the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting was chosen to fit the time-varying output matrix while preserving a static per-trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments lacking eye groups are excluded. Pupil NaNs are removed and interpolated across (one valid point is repeated; zero valid points raises). Interpolation outside source bounds uses endpoint values. Repeated percentile edges are nudged. Missing required columns/groups, invalid outcomes, or other per-session errors generally raise and abort rather than being caught; there is no per-session exception handler.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if valid.sum() == 0: raise ValueError("No valid pupil samples available")
if valid.sum() == 1: return np.full(query_t.shape, float(pupil_diameter[valid][0]), ...)
...
raise KeyError(f"Missing interval column '{col}' in {group.name}")
```

iii. The notes frame exclusion as necessary to provide all targets and interpolation as filling small invalid/blink gaps; robust quantiles address tied distributions. They do not justify the absence of fault isolation.

## 9-a. What are the most time-consuming steps of the code?

i. Pass-two reading of full NWBs and trial-by-trial interpolation of the full neuron event matrix dominate; the agent estimated neural resampling scales with roughly two billion neuron-bins.

ii.
```python
raw = read_session_raw(session)
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Notes explicitly identify neural interpolation as the dominant expected cost and report per-session timing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer per-session and per-trial loops remain, as does a per-sample Python loop assigning stimulus identities. Neural interpolation itself is vectorized over neurons; trial grids have variable lengths, making complete trial-loop vectorization less direct.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]): ...
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes highlight vectorization across neurons as an implemented speedup but do not discuss vectorizing image lookup or batching overlapping trial interpolation.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened/read in both passes. Running and pupil are interpolated for every trial in pass one to calculate bin edges, then interpolated again in pass two for output. Trial grids and filtering are also recomputed.

ii.
```python
raw = read_session_raw(session, load_events=False)  # collect_global_statistics
...
raw = read_session_raw(session)                     # convert_sessions
```

iii. The two-pass design was justified as bounding memory by avoiding retention of neural arrays and all trial intermediates, trading extra I/O/behavior computation for memory safety.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads/casts stimulus fields `trials_id`, `active`, and `flashes_since_change` but never uses them. It computes native frame intervals and valid-trial-count dictionaries only for logging. It also constructs optional detailed plotting intermediates, though only when requested.

ii.
```python
[..., "trials_id", "active", "flashes_since_change"]
...
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(...)
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
```

iii. These fields support auditability, progress reports, and optional sanity plots, but do not enter `converted_data.pkl` or decoder training.
