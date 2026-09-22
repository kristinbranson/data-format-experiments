# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local release metadata CSV, maps metadata rows to locally present experiment NWBs, removes passive experiments and experiments without the required eye-tracking groups, and reads each retained NWB directly with `h5py`. It makes two passes: one for global statistics and one for conversion.

ii.
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {int(path.stem.split("_")[-1]): path for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The notes say direct HDF5 reading was chosen because the installed `pynwb/hdmf` stack could not instantiate the NWBs through the SDK. Passive sessions were excluded because trial outcome is required, and sessions lacking pupil data were excluded because pupil diameter is a required output.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` strings. A mapping is populated in first-seen experiment order and each converted session receives its subject index.

ii.
```python
mouse_id=str(int(row.mouse_id))
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The agent states that the metadata mouse identifier is the global subject identifier.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`/NWB file is treated as one decoder session. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
...
"session_ophys_experiment_ids": [int(s.ophys_experiment_id) for s in sessions]
```

iii. The notes acknowledge that the local 284 experiment NWBs represent 247 ophys sessions, but justify experiment-level sessions because neural traces are experiment-specific and direct merging would require multi-plane handling.

## 1-d. How are the data split into trials?

i. Trial boundaries come from `intervals/trials`. For every retained trial, the code creates a grid from `start_time` (inclusive) toward `stop_time` at 1/30 s intervals; trial lengths therefore vary.

ii.
```python
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
```

iii. The agent chose the SDK/NWB trial table and full start-to-stop window so pre-change and post-change task signals are represented.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials flagged go or catch and rejects aborted and auto-rewarded trials. Sessions with fewer than two retained trials are skipped. It also excludes passive experiments and experiments missing pupil tracking before trial filtering.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. This directly follows the requested go/catch inclusion and aborted/auto-reward exclusion. The notes justify active-only sessions by the operant trial-outcome task and the whole-session pupil requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the NWB precomputed calcium event-detection matrix and its timestamps.

ii.
```python
ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. The agent selected events because the strategy paper says its neural analyses used detected calcium events, despite the dataset also containing dF/F.

## 2-b. How is the `neural` data processed?

i. Events are linearly interpolated over time, simultaneously for all neurons, to each trial's 30 Hz grid and transposed to neuron-by-time. Experiments are not merged across planes.

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. A common 30 Hz grid was chosen because native imaging rates vary and the paper interpolated event responses to common 30 Hz timestamps for event-triggered analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. When a `valid_roi` column exists and the dimensions permit it, invalid ROIs are removed. No additional activity-based neuron filter is applied.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say released ROI validity already incorporates segmentation, duplicate/union, and trace quality controls, so ad hoc filtering was avoided.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are evaluated on an absolute-time grid beginning at trial `start_time` and ending before `stop_time`; metadata names trial start as the alignment event and gives offset 0.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
"temporal_alignment_event": "trial start",
"off_start": 0.0,
```

iii. The full trial window was intended to preserve all pre- and post-change signals while synchronizing streams in absolute experiment time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at 33.33 ms (30 Hz). Neural activity is linearly resampled from its native roughly 11 or 31 Hz rate; this is interpolation, not count-preserving aggregation.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
n_bins = max(1, int(math.ceil((stop - start) / dt)))
return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The agent cites mixed native rates, 30 Hz behavior/eye tracking, and the paper's 30 Hz event-triggered interpolation as reasons for one shared grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the selected active task stimulus-presentation table's `start_time`, `stop_time`, `image_name`, and `omitted` fields.

ii.
```python
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name", "is_change", "omitted", ...])
```

iii. The agent used presentation intervals so identity describes the actual non-gray screen and so gray/omitted intervals can be represented explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is built as `gray` plus sorted non-omitted image names. Each time is initialized to gray and replaced by the corresponding image code only while inside a non-omitted presentation interval.

ii.
```python
image_values = ["gray"] + sorted(image_names)
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. The notes say gray is necessary because the task has 500 ms gray intervals and omissions extend gray rather than showing an image; global codes ensure consistency.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus intervals are queried at exactly the same 30 Hz `grid` used for neural interpolation.

ii.
```python
neural_trial = interpolate_matrix(..., grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. The agent validated that all streams use synchronized absolute timestamps and a shared trial grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from stimulus presentation `is_change`, together with presentation start/stop times and `omitted`.

ii.
```python
is_change = stimulus["is_change"]
changed = is_change[sub_idx] & (~omitted[sub_idx])
```

iii. The agent preferred the stimulus table's explicit change marker over reconstructing it from the trial flags.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The signal defaults to zero and is set to one only at grid points inside a non-omitted stimulus interval whose `is_change` flag is true. Thus it covers the changed-image flash, not the following gray interval.

ii.
```python
codes = np.zeros(query_t.shape, dtype=np.int64)
in_interval = query_t[valid] < stops[idx_valid]
codes[assign] = changed.astype(np.int64)
```

iii. The notes say this yields a non-impulse but still transient change target aligned to the task's actual changed presentation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is applied. The Boolean `is_change & ~omitted` expression is cast directly to integer categories 0 (`no_change`) and 1 (`change`).

ii.
```python
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. The raw presentation field is already Boolean, so no learned or amplitude threshold is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change status is evaluated from stimulus intervals on the same trial grid as neural activity.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Shared synchronized timestamps and the common grid were used to prevent cross-stream offsets.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed/data` and that series' timestamps.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. This is the SDK's processed running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to the trial grid. Global 20th, 40th, 60th, and 80th percentiles are computed over all retained trial bins, with a small adjustment for duplicate edges, and used for digitization.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
percentiles = np.nanpercentile(values, [20, 40, 60, 80])
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global quintiles were chosen to give shared, approximately balanced decoder classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global quintile edges produce five integer classes 0 through 4 via `np.digitize(..., right=False)`.

ii.
```python
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Five equal-percentile bins are explicitly required, and global edges keep their meaning consistent.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed and neural events are independently interpolated at the same absolute 30 Hz grid points.

ii.
```python
neural_trial = interpolate_matrix(..., grid).T
running_cont = interpolate_vector(..., grid)
```

iii. The streams are hardware-synchronized, so querying both on the common grid was considered sufficient alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It reads pupil ellipse width and height plus eye-tracking timestamps, and defines diameter as their pointwise maximum. Although the notes repeatedly call the signal blink-masked, the conversion code does not read or apply a blink flag.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The agent intended to follow SDK eye-processing semantics, using both ellipse axes and interpolating over invalid/blink samples. The stated blink-masking intent is not present in the implementation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite diameter samples are discarded for interpolation; remaining values are linearly interpolated to the 30 Hz grid. Global quintile edges are computed and applied. No explicit blink removal occurs.

ii.
```python
valid = np.isfinite(pupil_diameter)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes justify interpolation across short invalid/blink gaps and global bins, but incorrectly describe the raw input as already blink-masked.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global 20/40/60/80 percentile edges yield five categories through `np.digitize`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. This implements the required five equal-percentile bins with consistent categories across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated at the same 30 Hz absolute timestamps used for neural interpolation.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The agent relies on synchronized eye and ophys clocks and the shared grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trial-table Boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. These are the canonical mutually exclusive outcomes for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true outcome in fixed order is mapped to 0-3 and broadcast across all time bins; absence of a valid label raises an error.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting lets the static trial target fit the common time-varying output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments without required eye-tracking groups are excluded wholesale. Non-finite pupil values are interpolated across when possible; zero or one valid sample causes an error or constant fill. Duplicate percentile edges are nudged apart. Sessions with fewer than two valid trials are skipped. Other malformed/missing fields generally raise and stop the run; `np.interp` also silently extends endpoint values outside sampled ranges.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
if valid.sum() == 0: raise ValueError("No valid pupil samples available")
if valid.sum() == 1: return np.full(query_t.shape, float(pupil_diameter[valid][0]))
percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. The notes prefer excluding sessions unable to supply a required output, filling small pupil gaps by interpolation, and retaining sessions only when decoder evaluation is possible.

## 9-a. What are the most time-consuming steps of the code?

i. The agent identifies pass-2 neural interpolation as the dominant cost because every retained trial resamples every neuron's full event matrix; NWB I/O is also repeated.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Sample timing was extrapolated by total neuron-bins, and the notes call neural interpolation the dominant expected cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over trials and the small Python loop that assigns image names could be vectorized or replaced by indexed lookup. Neural interpolation is already vectorized over neurons, and outcome lookup loops over four labels.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes emphasize that vectorizing across neurons removed the important inner loop; variable trial lengths make full trial batching less direct.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened and most non-neural streams are read in both passes. Running and pupil values are interpolated for every retained trial in pass 1 to obtain global edges, then interpolated again in pass 2 for saved outputs. Trial grids are also rebuilt.

ii.
```python
raw = read_session_raw(session, load_events=False)  # pass 1
...
raw = read_session_raw(session)                     # pass 2
```

iii. The two-pass design was intentional to keep memory bounded while obtaining global bins before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass 1 computes per-session native frame intervals and valid-trial counts that are only printed, and reads the complete stimulus/running/pupil/trial content whose interpolated arrays are then discarded and recomputed. Optional plots also compute/display continuous traces not stored in the final dataset. The final decoder uses only discretized running/pupil outputs, not the continuous intermediates.

ii.
```python
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
running_values.append(interpolate_vector(...))
pupil_values.append(interpolate_pupil(...))
```

iii. These diagnostics support sanity checks and global edge estimation, while the second-pass recomputation is the memory-saving tradeoff of the chosen architecture.
