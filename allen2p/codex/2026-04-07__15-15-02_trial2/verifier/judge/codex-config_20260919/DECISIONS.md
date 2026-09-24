# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local Allen release directly with `h5py`. It joins NWB filenames to `ophys_experiment_table.csv`, retains files present locally, excludes passive experiments and experiments without the required eye-tracking groups, and makes two passes: one for global category/bin statistics and one for neural conversion.

ii.
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {int(path.stem.split("_")[-1]): path
            for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(sessions)
data = convert_sessions(sessions, running_edges, pupil_edges, image_values, args.show_processing)
```

iii. The notes say direct HDF5 access was used because the installed `pynwb/hdmf` stack could not instantiate these NWB 2.6 files. The local files were interpreted as a curated release subset; active sessions were selected because trial outcome is required.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` strings, registered when an included experiment is converted. Each output session receives the corresponding integer subject index.

ii.
```python
mouse_id=str(int(row.mouse_id))
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
data["subject_idx"].append(subject_idx)
```

iii. The notes identify the NWB/metadata mouse identifier as the canonical subject identifier.

## 1-c. How are the data split into sessions?

i. Each local `ophys_experiment_id` NWB is treated as a separate decoder session, sorted by experiment ID. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
sessions = [SessionInfo(ophys_experiment_id=int(row.ophys_experiment_id),
                        path=file_map[int(row.ophys_experiment_id)], ...)
            for row in exp_table.itertuples(index=False)]
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. The notes explicitly resolve the experiment/session discrepancy by treating each experiment file as a session because its neural traces are experiment-specific, while acknowledging that 284 files represent only 247 unique ophys/behavior sessions.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`; each retained row is sliced from its `start_time` to `stop_time` on a newly constructed 30 Hz grid, producing variable-length trials.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The notes say SDK trial boundaries are authoritative and that the complete trial window preserves pre- and post-change behavior.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are retained; aborted and auto-rewarded trials are removed. Sessions with fewer than two retained trials are skipped. Passive and pupil-missing experiments are removed before trial processing.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. This is justified from the requested go/catch taxonomy, the instruction to exclude aborted and auto-rewarded trials, the decoder's minimum-trial requirement, and the need for all requested outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB precomputed calcium event matrix and its event-detection timestamps.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. The agent chose events because the strategy paper says its neural analyses used detected calcium events, despite the reference conversion choosing SDK dF/F traces.

## 2-b. How is the `neural` data processed?

i. Event traces are optionally filtered by `valid_roi`, linearly interpolated from native ophys timestamps to each trial's 30 Hz grid, transposed to neuron-by-time, and cast to `float32`.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
...
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. The notes cite SDK invalid-ROI behavior and the paper's interpolation to common 30 Hz timestamps. No normalization beyond the released event processing is added.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code attempts to retain only valid ROIs when the event matrix still corresponds to the full cell table; otherwise it assumes the NWB event matrix is already filtered. No further cell-level QC is applied.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes state that released NWBs already reflect Allen ROI curation and reject additional ad hoc filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start: a grid begins exactly at each trial's `start_time` and ends before `stop_time`; neural events are interpolated onto that grid.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
...
"temporal_alignment_event": "trial start",
```

iii. The agent considered trial start the natural alignment for full trial segmentation and used the same grid for every stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 1/30 s (33.33 ms). All neural and behavioral streams are linearly resampled/interpolated to that grid; this upsamples native ~11 Hz experiments and slightly resamples ~31 Hz experiments.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
n_bins = max(1, int(math.ceil((stop - start) / dt)))
return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The notes argue that mixed native frame rates require a common size and that the paper interpolated event-triggered analyses to 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the chosen active stimulus-presentation table's `start_time`, `stop_time`, `image_name`, and `omitted` columns.

ii.
```python
stim = read_interval_table(stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted",
     "trials_id", "active", "flashes_since_change"])
```

iii. The notes prefer presentation intervals over trial-level initial/change names so the time-varying output represents actual flashes, gray gaps, and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is built as `gray` plus sorted nonempty, non-omitted image names. At each query time the enclosing presentation receives its image code; times outside a presentation and omitted presentations receive `gray`.

ii.
```python
image_values = ["gray"] + sorted(image_names)
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. The agent says an explicit gray class is necessary for 500 ms gray intervals and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are evaluated on exactly the same per-trial 30 Hz `grid` used for interpolated neural activity.

ii.
```python
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The common query grid is the stated mechanism guaranteeing alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from stimulus-presentation `is_change`, `omitted`, `start_time`, and `stop_time`, rather than directly from the trial `go` flag and `change_time`.

ii.
```python
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
```

iii. The notes say stimulus `is_change` directly identifies the actual changed-image presentation and naturally leaves catch trials at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 30 Hz point, the code finds the active presentation and copies `is_change` only while that non-omitted presentation is on screen; all other times are zero.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. The agent describes this as less degenerate than a one-bin impulse and aligned with task structure.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is learned: the existing boolean `is_change` is encoded as 0 (`no_change`) or 1 (`change`), gated by presentation membership and non-omission.

ii.
```python
"output_values": [ ..., ["no_change", "change"], ...]
changed = is_change[sub_idx] & (~omitted[sub_idx])
```

iii. The source is already categorical, so the notes do not propose additional thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is evaluated on the same trial `grid` as neural interpolation, and is one throughout the changed-image presentation interval.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The common 30 Hz grid is the alignment justification.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB processed running `speed/data` and corresponding `speed/timestamps`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. The notes identify this as the SDK's filtered running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated onto each 30 Hz trial grid. Four global 20/40/60/80 percentile edges are computed in pass 1, made strictly increasing if tied, and used to digitize values into five bins.

ii.
```python
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
running_edges = robust_quintile_edges(running_all)
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global quintiles were chosen to give shared, approximately balanced classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Values are categorized with the four global quintile boundaries via `np.digitize(..., right=False)`, yielding integer classes 0–4.

ii.
```python
percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The notes state that percentile bins meet the five-equal-percentile-bin requirement and avoid session-specific meanings.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running values and neural events are independently interpolated onto the identical trial grid.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Hardware-synchronized timestamps plus a common query grid are the implicit alignment rationale.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses pupil ellipse `width` and `height` from `pupil_tracking` and timestamps from `eye_tracking`; diameter is defined as their pointwise maximum.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The notes call `max(width, height)` pupil diameter and expect reference blink masking already reflected as invalid values.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite diameter samples are discarded for interpolation; remaining values are linearly interpolated to trial grids. Global quintile edges are then computed and applied, with tied edges nudged upward.

ii.
```python
valid = np.isfinite(pupil_diameter)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes say interpolating after blink masking fills short gaps and global quintiles provide consistent categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses global 20/40/60/80 percentile thresholds and `np.digitize`, producing classes 0–4.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. This follows the same global equal-percentile rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Valid pupil samples are interpolated using their timestamps onto the same 30 Hz grid used for the neural trace.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The common grid and synchronized timestamps are used to ensure alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. The notes identify these as the SDK's mutually exclusive canonical outcome flags.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true flag is mapped in fixed order to 0–3. Absence of a valid flag raises an error. The static code is broadcast across every trial time bin.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting was chosen to keep all decoder outputs in one time-varying matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments missing pupil-tracking groups are excluded. Non-finite pupil samples are ignored and gaps/extrapolated ends are filled by `np.interp`; a single valid sample is repeated, while no valid samples raise. Tied percentile edges are nudged. Sessions with fewer than two trials are skipped, and missing outcome labels or required fields fail loudly. `np.interp` also clamps out-of-range running/neural/pupil queries to endpoint values.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
if valid.sum() == 1:
    return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
...
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. The notes describe excluding three pupil-missing active experiments and filling blink-related gaps by interpolation. The implementation favors deterministic failure for structurally missing required fields.

## 9-a. What are the most time-consuming steps of the code?

i. The two complete NWB passes and, especially, pass-2 interpolation of every trial's event matrix across all neurons dominate runtime and I/O.

ii.
```python
raw = read_session_raw(session, load_events=False)  # pass 1
...
raw = read_session_raw(session)                     # pass 2
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The notes explicitly identify neural interpolation as the dominant expected cost and estimate runtime by neuron-bin work.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over trials in both passes, the per-sample loop assigning image codes, and repeated per-trial interpolation/search operations could be batched or vectorized. Neural interpolation is already vectorized across neurons.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes highlight the across-neuron vectorization as a speedup but do not claim the trial and image-label loops are vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Every session is opened/read in both passes; each valid trial grid is reconstructed in both passes; running and pupil are interpolated once for global statistics and again for final output. Trial filtering and session-native timing are also recomputed/read.

ii.
```python
raw = read_session_raw(session, load_events=False)
...
raw = read_session_raw(session)
...
grid = session_grid(start, stop)
```

iii. The two-pass design is justified as a memory-saving tradeoff: it avoids retaining neural arrays while global thresholds are computed.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass 1 reads several trial/stimulus columns not needed for global statistics, computes and returns valid-trial counts and native frame intervals used only for logging, and interpolates all behavior solely to derive bin edges. Pass 2 builds continuous running/pupil arrays that are immediately digitized and discarded. If `--show-processing` is used, plotting-only arrays and figures are also produced.

ii.
```python
return running_edges, pupil_edges, image_values, valid_trial_counts, native_dt_by_session
...
running_cont = interpolate_vector(...)
pupil_cont = interpolate_pupil(...)
running_bins = digitize_with_edges(running_cont, running_edges)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Most of this is intentional validation, logging, or the necessary intermediate work for global quintiles; the notes present the optional plots as sanity checks rather than decoder inputs.
