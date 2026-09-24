# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds the experiment list from the AllenSDK `VisualBehaviorOphysProjectCache`, but opened with `from_local_cache()` instead of the S3 cache, and then intersects the SDK's `get_ophys_experiment_table()` with the set of NWB files that actually exist on disk under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`. It attaches the local file path to each row and sorts by `date_of_acquisition, mouse_id, ophys_session_id, targeted_structure, imaging_depth`. Each experiment is then loaded individually with `BehaviorOphysExperiment.from_nwb_path(row["local_path"])` inside the main loop; from that object it reads `ophys_timestamps`, `events`, `running_speed`, `eye_tracking`, `stimulus_presentations` and `trials`. Notably the AI does **not** filter on `project_code`, so both `VisualBehavior` (165 sessions) and `VisualBehaviorMultiscope` (34 sessions) experiments are included; it does filter out passive experiments (`~experiments["passive"]`), leaving 202 candidate experiments of the 284 on disk.

ii.
```python
def available_experiment_table(data_dir: Path) -> pd.DataFrame:
    nwb_root = data_dir / NWB_DIRNAME
    local_paths = {}
    for path in sorted(nwb_root.glob("behavior_ophys_experiment_*.nwb")):
        match = re.search(r"(\d+)\.nwb$", path.name)
        if match:
            local_paths[int(match.group(1))] = path
    cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
    experiments = cache.get_ophys_experiment_table()
    experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
    experiments["local_path"] = [str(local_paths[int(idx)]) for idx in experiments.index]
    experiments = experiments.sort_values(
        ["date_of_acquisition", "mouse_id", "ophys_session_id",
         "targeted_structure", "imaging_depth"])
    return experiments
```
```python
experiments = experiments.loc[~experiments["passive"]].copy()
...
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. From the trajectory (step 56, 60): "The local cache is incomplete: the project table lists release metadata for more experiments than there are NWB files on disk. That's a real constraint … so I'm switching to an explicit 'available NWB files only' filter before implementing the converter." and "The actual on-disk dataset is a partial release: 284 NWB experiments across 247 sessions and 38 mice … the converter has to target the downloaded subset on disk rather than the full paper cohort." In `CONVERSION_NOTES.md` the AI records that paper-level totals (376 sessions / 82 mice) cannot be reproduced from the local copy, and that active-only filtering is applied because "The whitepaper and paper distinguish active change-detection sessions from passive viewing."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` strings of the included experiments. A mouse is registered into `subjects` lazily, only once one of its experiments has survived all filters and produced ≥2 usable trials; `subject_idx` stores the index of that mouse for every emitted session. The result is 38 subjects (one more than the reference's 37, because the extra mouse comes from the `VisualBehaviorMultiscope` experiments the AI also included).

ii.
```python
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
...
subject_idx.append(subject_to_idx[subject])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int16),
```

iii. No explicit justification is given beyond `mouse_id` being the SDK's animal identifier; `CONVERSION_NOTES.md` simply reports "38 mice" in the local cache and "38 subjects" in the final dataset. The registration-on-success ordering avoids listing mice that contributed no usable session.

## 1-c. How are the data split into sessions?

i. A decoder "session" is **one `BehaviorOphysExperiment` NWB file (one imaging plane)**, not one `ophys_session_id`. The AI explicitly does not merge planes recorded simultaneously in the same behavioral session. Because the multiscope experiments are included, a single behavioral session can contribute up to 8 decoder "sessions" that share identical trials, running, pupil, stimulus and outcome data but different neurons: the final dataset has 199 sessions covering only 171 unique `ophys_session_id` values, and mouse 457841 alone produces 34 "sessions". Passive sessions are dropped entirely, so only OPHYS_1/3/4/6 (active) session types appear.

ii.
```python
experiments = experiments.loc[~experiments["passive"]].copy()
...
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_temp_sessions.append(output_temp_trials)
    session_info.append({
        "experiment_id": int(experiment_id),
        "ophys_session_id": int(row["ophys_session_id"]),
        ...
    })
```

iii. `CONVERSION_NOTES.md`, "Session unit": "Each decoder 'session' is one Allen `BehaviorOphysExperiment` NWB file, not one unique behavior session. This matches the AllenSDK data model and avoids incorrectly merging different imaging planes from multiscope sessions into one neuron matrix." Trajectory step 200 repeats this: "each decoder 'session' corresponds to one Allen `BehaviorOphysExperiment` NWB file, which is the right unit for multiscope data because a single behavior session can contain multiple imaging planes." Passive exclusion is justified in the notes by the whitepaper's distinction between active change-detection and passive viewing sessions.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `dataset.trials` table. The AI keeps rows with `(go | catch) & ~aborted & ~auto_rewarded`, and uses the full trial window from `start_time` to `stop_time` (variable-length, ~7–12 s). Within the window it builds a regular 100 ms grid of bin centres starting at `start_time` (`make_target_times`), and every stream (neural, running, pupil, stimulus labels) is evaluated on that grid. Trials producing fewer than 2 bins are dropped. Median trial length in the output is ~85 bins (≈8.5 s).

ii.
```python
trials = dataset.trials.copy()
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
...
for trial_id, trial_row in trials.iterrows():
    start_time = float(trial_row["start_time"])
    stop_time = float(trial_row["stop_time"])
    target_times = make_target_times(start_time, stop_time, bin_size_s)
    if target_times.size < 2:
        continue
```
```python
def make_target_times(start_time, stop_time, bin_size_s):
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]
```

iii. Trajectory step 31: "The trial table confirms the key filter logic: `go` and `catch` are the behavior-defined trial types, while aborted and auto-rewarded trials are explicitly labeled and can be removed." `CONVERSION_NOTES.md`: "Per instructions and Allen trial definitions, trials are kept only when `go == True` or `catch == True`, `aborted == False`, `auto_rewarded == False` … This retains the experimentally meaningful go/catch trials while excluding premature-lick resets and free-reward trials."

## 1-e. How are trials filtered based on quality controls?

i. Filtering happens at several levels:
- **Experiment level:** passive experiments dropped; experiments that fail to load, have <2 ophys timestamps, have an empty `events` table, have no usable running signal, or have no finite `pupil_area` at all are skipped and logged in `metadata['source_subset']['excluded_sessions']` (3 experiments dropped in the full run, all for "no valid pupil": 795953296, 806456687, 833631914); experiments with <2 valid go/catch trials, or <2 trials surviving alignment, are skipped.
- **Trial level:** aborted/auto-rewarded/non-go-non-catch dropped; trials with <2 time bins dropped; trials where running or pupil interpolation returns `None` dropped; trials whose `hit/miss/false_alarm/correct_reject` flags are all False dropped.
- No neuron-level filtering beyond what the SDK already applies.

ii.
```python
if ophys_timestamps.size < 2: ... continue
if len(events_df) == 0: ... continue
if running_interp_source is None: ... continue
if not np.isfinite(pupil_diameter).any(): ... continue
if len(trials) < 2: ... continue
...
if target_times.size < 2: continue
if running_trial is None or pupil_trial is None: continue
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None: continue
...
if len(neural_trials) < 2:
    excluded_sessions.append({"experiment_id": int(experiment_id),
                              "reason": "fewer_than_two_kept_trials"})
    continue
```

iii. `CONVERSION_NOTES.md` "Exclusions": "No sessions were dropped for missing events or too few valid trials. Three experiment files were excluded because pupil data were entirely invalid." The ≥2-trial rule follows the instruction "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI also states that "AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically when loading each `BehaviorOphysExperiment`" (trajectory step 25: "I've confirmed the SDK excludes invalid ROIs when loading experiments, which is the right baseline for the quality-control requirement").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrix is built from `dataset.events["events"]` — the AllenSDK L0-detected discrete calcium event magnitude traces (one 1-D trace per valid ROI, on the `ophys_timestamps` timebase). dF/F is deliberately not used, and `filtered_events` is deliberately not used.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    print(f"  skip {experiment_id}: no valid event traces")
    continue
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. Trajectory step 16: "I need to pin down two decisions before implementation: … whether to use dF/F or event traces as the neural signal." `CONVERSION_NOTES.md` "Neural signal": "Neural activity uses AllenSDK discrete calcium events from `dataset.events['events']`. This matches the paper's statement that neural analyses were performed on detected calcium events rather than raw fluorescence. The converter intentionally does not use dF/F traces and does not use `filtered_events`." (The trajectory shows the AI inspected `events`/`dff_traces` columns via the SDK at step 33 and grepped the tutorials, but never actually extracted the text of `paper.pdf`, so the cited "paper's statement" is asserted rather than shown.)

## 2-b. How is the `neural` data processed?

i. Almost no processing is applied to the traces themselves: the per-ROI event traces are stacked into an `(n_neurons, T)` float32 matrix; there is no normalisation, smoothing, z-scoring or neuron selection. The only transformation is temporal: for each trial the code picks, for every 100 ms bin centre, the **single nearest ophys frame** and copies that column (`neural_full[:, nearest]`). Because the native frame interval is ~32 ms for single-plane experiments, this keeps roughly one of every three frames and discards the rest rather than summing/averaging events within the bin. Measured on the converted sample data, the per-bin nonzero rate (0.20%) equals the native per-frame nonzero rate in the task period (~0.17%), i.e. roughly two-thirds of all detected events never enter the dataset; the validator consequently reports thousands of "all neural data is zero" trial warnings.

ii.
```python
def nearest_indices(source_times, target_times):
    idx = np.searchsorted(source_times, target_times, side="left")
    idx = np.clip(idx, 1, source_times.size - 1)
    left, right = idx - 1, idx
    choose_right = np.abs(source_times[right] - target_times) < np.abs(
        target_times - source_times[left])
    return np.where(choose_right, right, left).astype(np.int64)
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md`: "neural samples are aligned by nearest ophys timestamp"; the goal was a "common 100 ms bin size … for every session" so that "single-plane and multiscope sessions can coexist in one decoder dataset" (trajectory step 69). The AI acknowledges but dismisses the consequence: "`train_decoder.py` reports warnings for some trials whose neural event matrices are entirely zero. These are not format errors. They arise when valid event traces contain no detected events during a given trial window, which is plausible for sparse event representations."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level quality control is implemented. The AI relies on the AllenSDK, which returns only valid ROIs in `events`/`cell_specimen_table`. Whole experiments are dropped if the `events` table is empty. Sessions with as few as 4 neurons are retained (`n_neurons: mean 146.6, min 4, max 666`).

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    continue
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. `CONVERSION_NOTES.md`: "AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically when loading each `BehaviorOphysExperiment`." Trajectory step 25: "I've confirmed the SDK excludes invalid ROIs when loading experiments, which is the right baseline for the quality-control requirement."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the trial start (`trials.start_time` from the SDK trials table), on the ophys clock. The 100 ms bin-centre grid begins at `start_time + 50 ms` and runs until `stop_time`; each bin centre is mapped to the nearest ophys frame. All other streams (running, pupil, image identity, image change, outcome) are evaluated on exactly the same `target_times` grid, so they are aligned to the neural data by construction. Metadata records `temporal_alignment_event = "trial start time from AllenSDK trials table"`, `off_start = 0.0`, `off_end = None` (variable trial length).

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
"temporal_alignment_event": "trial start time from AllenSDK trials table",
"off_start": 0.0,
"off_end": None,
```

iii. `README.md`: "Time series are aligned on Allen trial start times and resampled to 100 ms bins." Trajectory step 69: "align all outputs to the same trial-centered timestamps, and resample to a common 100 ms grid so single-plane and multiscope sessions can coexist in one decoder dataset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the data are rebinned to a uniform **100 ms** grid (`--time-bin-ms`, default 100.0), for all sessions, and `metadata['time_bin_size'] = 100.0`. Native ophys intervals are 32.31 ms (single-plane) to 93.23 ms (multiscope); these are recorded in metadata (`native_ophys_frame_interval_ms_summary`) but the data are resampled away from them. The rebinning is implemented as nearest-neighbour **sampling** of one frame per bin, not as aggregation of all frames in the bin, so for 32 ms data ~2/3 of frames (and of the detected events they carry) are thrown away. Running speed and pupil are linearly interpolated onto the same grid (appropriate for continuous signals).

ii.
```python
TIME_BIN_MS_DEFAULT = 100.0
...
bin_size_s = time_bin_ms / 1000.0
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
"time_bin_size": float(time_bin_ms),
"native_ophys_frame_interval_ms_summary": {
    "mean": float(np.mean(native_dt_ms)), "median": float(np.median(native_dt_ms)),
    "min": float(np.min(native_dt_ms)), "max": float(np.max(native_dt_ms))},
```

iii. `CONVERSION_NOTES.md`: "The local dataset mixes single-plane and multiscope experiments. Their native frame intervals differ: median 32.32 ms, min 32.31 ms, max 93.23 ms. Using 100 ms bins preserves compatibility across both acquisition modes while staying close to the slower multiscope sampling interval." Trajectory step 87/93: the multiscope spot-check was run specifically "because that's the only place the native frame rate changes and could expose a resampling bug."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`, using each flash's `image_name`, `start_time`, `end_time`, `omitted` and its `trials_id` link to the trial. It is **not** taken from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains("change_detection", na=False)
].copy()
change_detection = change_detection.sort_values("start_time")
...
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
trial_stim = trial_stim.sort_values("start_time")
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
```

iii. Trajectory step 36: "I've confirmed a useful alignment rule from the stimulus table: each trial is already linked to the sequence of flashed image intervals via `trials_id`, and the changed image row is marked with `is_change`." `CONVERSION_NOTES.md`: "Stimulus labels come only from `stimulus_presentations` rows whose `stimulus_block_name` contains `change_detection`."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial the label array is initialised to a dedicated `gray` category (code 0) and then, for every non-omitted flash belonging to that trial, all bins whose centre falls in `[start_time, end_time)` are set to that image's integer code. Image names are assigned codes lazily, in first-encounter order, into a global `image_values` list, so codes are shared across sessions. The result is 17 categories (`gray` + 16 images); because the flash cadence is 250 ms on / 500 ms grey, `gray` accounts for 66.8% of all time bins. Omitted flashes remain `gray`.

ii.
```python
image_values = [GRAY_LABEL]
image_to_idx = {GRAY_LABEL: 0}
...
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
    if stim_row.image_name not in image_to_idx:
        image_to_idx[stim_row.image_name] = len(image_values)
        image_values.append(stim_row.image_name)
    image_idx = image_to_idx[stim_row.image_name]
    mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
    image_labels[mask] = image_idx
```

iii. `CONVERSION_NOTES.md`: "`image_identity` is the image shown during flashed image epochs; all other times are labeled `gray`; omissions remain `gray` … This matches the task structure described in the references: 250 ms image flash followed by 500 ms gray, with 5% omissions." The AI cross-checks the result: "The `gray` fraction is close to the expected 2/3 from the 250 ms image + 500 ms gray task cadence."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The label is computed directly on `target_times`, the same 100 ms bin-centre grid used to sample the neural matrix, by boolean masks on flash start/end times. A bin is labelled with an image iff its centre lies inside that flash, so labels and neural columns are aligned index-for-index with no additional shifting.

ii.
```python
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest]
```

iii. Trajectory step 69: "align all outputs to the same trial-centered timestamps." The single shared `target_times` array is the alignment mechanism for all five outputs.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` boolean of the same `change_detection` stimulus-presentation rows (plus their `start_time`/`end_time`), not from the trials table's `change_time`/`go` columns.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
for stim_row in trial_stim.itertuples():
    ...
    if bool(stim_row.is_change):
        change_labels[mask] = 1
```

iii. Trajectory step 36: "the changed image row is marked with `is_change`." `CONVERSION_NOTES.md`: "`image_change` is `1` only during flashed epochs with `is_change == True`, else `0`."

## 4-b. What processing is involved in computing `output` *Image change*?

i. No processing beyond the boolean-mask assignment: bins whose centre falls inside the changed flash (a 250 ms window) get 1, all others 0. Catch trials carry no `is_change == True` flash, so they are all-zero, as are omitted flashes. Overall 2.7% of bins are labelled `change` (vs 7.65% in the reference, which used a 750 ms flash+grey window).

ii.
```python
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. `CONVERSION_NOTES.md`: "`image_change` is rare, as expected for the roving-baseline change-detection design." The instruction ("value of 1 right after a change in image identity, otherwise 0") is satisfied by marking the change flash itself.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — it is already binary. `output_values` for this dimension is `['no_change', 'change']`, i.e. 0/1.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. Implicit: the source variable `is_change` is a boolean flag in the SDK stimulus table.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the same `target_times` grid and the same `[start_time, end_time)` masks, so the change flag occupies the same bin indices as the neural columns.

ii.
```python
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. Same as 3-c — one shared bin grid for every stream.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `dataset.running_speed`, using its `timestamps` and `speed` columns (the SDK's pre-processed, low-pass-filtered wheel encoder speed in cm/s).

ii.
```python
running_df = dataset.running_speed
running_interp_source = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    np.array([ophys_timestamps[0]], dtype=np.float64))
if running_interp_source is None:
    ... continue
```

iii. No explicit discussion; `CONVERSION_NOTES.md` states "`running_processing`: AllenSDK processed running speed aligned by timestamp and binned into global quintiles", i.e. the SDK's standard accessor is taken as the reference implementation (matching the whitepaper's encoder description the AI read in `methods.txt`).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Per trial, the speed trace is linearly interpolated (`np.interp`) onto the trial's 100 ms bin centres; non-finite samples are dropped first, timestamps are sorted and de-duplicated, and values outside the recorded range are held constant at the first/last value rather than set to NaN. The interpolated trial vectors are accumulated across the whole dataset and then discretised into 5 global quintile bins.

ii.
```python
def interp_signal(source_times, source_values, target_times):
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    if valid.sum() == 0:
        return None
    times = np.asarray(source_times[valid], dtype=np.float64)
    values = np.asarray(source_values[valid], dtype=np.float64)
    order = np.argsort(times); times, values = times[order], values[order]
    unique_times, unique_idx = np.unique(times, return_index=True)
    values, times = values[unique_idx], unique_times
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)
...
running_trial = interp_signal(running_df["timestamps"].to_numpy(np.float64),
                              running_df["speed"].to_numpy(np.float64), target_times)
all_running.append(running_trial)
```

iii. `CONVERSION_NOTES.md`: "running speed is linearly interpolated to those timestamps"; "Running and pupil are discretized into global quintiles over all kept trial time bins in the converted dataset."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into 5 equal-count bins defined by the 20/40/60/80th percentiles of all interpolated running values in the converted dataset (all trials, all sessions pooled), applied with `np.digitize`. The resulting edges (−0.004, 0.304, 15.78, 33.45 cm/s) are stored in metadata, and the output distribution is exactly 20% per bin.

ii.
```python
running_values = np.concatenate(all_running, axis=0)
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
def remap_to_bins(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int16)
...
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. Instruction-driven ("discretized into five equal percentile bins"). `CONVERSION_NOTES.md`: "Running and pupil bins are exactly balanced because they are defined by global percentiles."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. By interpolating directly onto `target_times`, the same 100 ms bin-centre grid used for the neural matrix; no lag correction is applied (the SDK streams are hardware-synchronised to a common clock).

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest]
running_trial = interp_signal(..., target_times)
```

iii. Trajectory step 69 / `CONVERSION_NOTES.md`: "align all outputs to the same trial-centered timestamps"; "running speed is linearly interpolated to those timestamps."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `dataset.eye_tracking["pupil_area"]`, converted to an equivalent circular diameter. The AI does not explicitly reference `likely_blink`, but the AllenSDK applies `filter_on_blinks()` when building the eye-tracking table, which already sets `pupil_area` (as opposed to `pupil_area_raw`) to NaN on likely-blink frames; the AI's interpolator drops non-finite samples, so blink frames are in fact excluded and interpolated over.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi)
if not np.isfinite(pupil_diameter).any():
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_valid_pupil"})
    continue
```

iii. `CONVERSION_NOTES.md`: "Pupil diameter is computed as equivalent diameter from AllenSDK `pupil_area`: `diameter = 2 * sqrt(area / pi)`." Trajectory step 38 shows the AI grepping the tutorials/methods for `pupil_area`/`pupil_width`/"pupil diameter" and finding the tutorials plot `pupil_area`, before choosing the area-derived diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area → equivalent diameter, then the same `interp_signal` path as running speed: non-finite (blink) samples dropped, timestamps sorted/de-duplicated, linear interpolation onto the trial's 100 ms bin centres, constant edge extrapolation. Values are pooled across the whole dataset and discretised into 5 global quintile bins. Experiments whose pupil trace is entirely non-finite are dropped (3 experiments).

ii.
```python
pupil_diameter = 2.0 * np.sqrt(eye_df["pupil_area"].to_numpy(np.float64) / np.pi)
...
pupil_trial = interp_signal(eye_df["timestamps"].to_numpy(np.float64),
                            pupil_diameter, target_times)
...
all_pupil.append(pupil_trial)
pupil_values = np.concatenate(all_pupil, axis=0)
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
```

iii. `CONVERSION_NOTES.md`: "`pupil_processing`: Equivalent pupil diameter derived from AllenSDK processed pupil_area, aligned by timestamp and binned into global quintiles." Dropping fully-invalid-pupil experiments is documented: "Three experiment files were excluded because pupil data were entirely invalid."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed: global 20/40/60/80th-percentile edges over all interpolated pupil values, `np.digitize` into bins 0–4. Edges (73.8, 83.8, 92.9, 105.4 px diameter) are stored in metadata; the realised distribution is exactly 20% per bin.

ii.
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. Instruction-driven ("discretized into five equal percentile bins"); the notes report the balanced bin check.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same `target_times` bin-centre grid as the neural data; no extra alignment step, and no lag correction between the eye-tracking camera clock and the ophys clock (the SDK returns them on a common synchronised timebase).

ii.
```python
pupil_trial = interp_signal(eye_df["timestamps"].to_numpy(np.float64),
                            pupil_diameter, target_times)
```

iii. Same shared-grid justification as running speed (trajectory step 69).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the SDK trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — checked in that priority order.

ii.
```python
def outcome_to_index(trial_row: pd.Series) -> int | None:
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
    return None
```

iii. `CONVERSION_NOTES.md`: "`trial_outcome` is static within each trial and encoded as 0: hit, 1: miss, 2: false_alarm, 3: correct_reject." Trajectory step 31 notes the trials table was inspected to confirm these columns.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The scalar outcome code is broadcast to a constant row across all time bins of the trial, so the output array stays `(5, n_timepoints)`. Trials whose four flags are all False return `None` and are dropped. Realised distribution: hit 30.3%, miss 57.1%, false alarm 1.7%, correct reject 10.8% (the reference, which also retained passive sessions, reports 18.7/68.7/1.1/11.5%).

ii.
```python
def build_output_array(image_labels, change_labels, running_bins, pupil_bins, outcome_idx):
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
    return np.vstack([
        image_labels.astype(np.int16),
        change_labels.astype(np.int16),
        running_bins.astype(np.int16),
        pupil_bins.astype(np.int16),
        outcome,
    ])
```

iii. The instruction lists trial outcome as "Static per-trial", while the format spec says "If at all possible, make it time-varying"; broadcasting a constant row satisfies both the per-trial semantics and the uniform `(n_output, n_timepoints)` shape the validator expects.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i.
- NWB load failures are caught per experiment, logged with the exception type in `metadata['source_subset']['excluded_sessions']`, and the experiment is skipped.
- Experiments with <2 ophys timestamps, empty `events`, an unusable running trace, or an all-NaN pupil trace are skipped and logged.
- Non-finite samples in running/pupil (including SDK blink-NaNs) are dropped before interpolation; duplicate/unsorted timestamps are de-duplicated and sorted; single-sample signals are broadcast to a constant.
- Values requested outside the recorded range are filled with the first/last observed value (`left=values[0], right=values[-1]`) rather than NaN, so no NaNs reach the output and the validator's NaN check passes.
- Trials shorter than 2 bins, or with no resolvable outcome, are dropped; sessions left with <2 trials are dropped.
- Omitted stimulus flashes are simply left labelled `gray`.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
except Exception as exc:
    excluded_sessions.append({"experiment_id": int(experiment_id),
                              "reason": f"load_failed:{type(exc).__name__}"})
    print(f"  skip {experiment_id}: failed to load ({exc})")
    continue
...
valid = np.isfinite(source_times) & np.isfinite(source_values)
if valid.sum() == 0:
    return None
...
interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
...
if bool(stim_row.omitted) or stim_row.image_name == "omitted":
    continue
```

iii. `CONVERSION_NOTES.md` records the exclusion list and the exact excluded experiment IDs, and the AI justifies not treating the all-zero-neural warnings as errors ("These are not format errors … plausible for sparse event representations"). The metadata field `source_subset.excluded_sessions` is explicitly written so exclusions are auditable.

## 9-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is loading each NWB file through `BehaviorOphysExperiment.from_nwb_path` (202 files), which parses the whole experiment — traces, stimulus table, eye tracking, running — and is both I/O- and CPU-bound. The trajectory shows the full conversion running for a long stretch with the agent repeatedly polling it (steps 99–133: "still CPU-bound", "many-file batch job range"). The second cost is the per-trial inner loop, which for every one of the ~51,000 trials re-extracts, re-sorts and re-uniques the *entire session's* running and eye-tracking arrays inside `interp_signal`, and linearly scans the whole stimulus table for `trials_id == trial_id`. A third cost is writing the 2.7 GB pickle of dense float32 matrices that are 99.8% zeros.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])   # per experiment
...
for trial_id, trial_row in trials.iterrows():                        # ~250 trials/session
    running_trial = interp_signal(running_df["timestamps"].to_numpy(np.float64),
                                  running_df["speed"].to_numpy(np.float64), target_times)
    pupil_trial = interp_signal(eye_df["timestamps"].to_numpy(np.float64),
                                pupil_diameter, target_times)
    trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
```

iii. Not analysed by the AI; its notes and trajectory only observe that the full run is slow ("The full conversion is still running… CPU usage is high and memory is modest, which is what I'd expect while the SDK parses and aligns hundreds of sessions"), attributing the cost entirely to SDK parsing rather than to its own per-trial work.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i.
- The per-trial calls to `interp_signal` could be replaced by a single per-session interpolation of running speed and pupil onto all ophys timestamps (or onto the concatenation of all trials' target times), as the reference does; the sort/unique/`isfinite` work is currently repeated once per trial per signal.
- `change_detection.loc[change_detection["trials_id"] == trial_id]` inside the trial loop is an O(n_trials × n_flashes) scan; a single `groupby("trials_id")` (or a `searchsorted` on flash start times) would make it O(n_flashes).
- The inner `for stim_row in trial_stim.itertuples()` loop that builds a boolean mask over `target_times` for each flash could be replaced by one `np.searchsorted` of `target_times` into the flash boundary array.
- `nearest_indices` and the trial slicing could be computed for all trials at once from a concatenated target-time vector.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
    running_trial = interp_signal(running_df["timestamps"].to_numpy(dtype=np.float64), ...)
    pupil_trial = interp_signal(eye_df["timestamps"].to_numpy(dtype=np.float64), ...)
    trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
    for stim_row in trial_stim.itertuples():
        mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
```

iii. The AI never discusses vectorisation; no justification is offered for the per-trial recomputation.

## 9-c. What processing does the code repeat multiple times?

i.
- `running_df["timestamps"].to_numpy()` / `["speed"].to_numpy()` and `eye_df["timestamps"].to_numpy()` are re-materialised, re-masked, re-sorted and re-uniqued on every trial (hundreds of times per session, ~51,000 times overall) even though the underlying session-level arrays never change.
- The running interpolation is additionally performed once per experiment purely as a validity probe (`running_interp_source`), then thrown away.
- `change_detection` is re-scanned in full for every trial.
- Discretisation is a genuine second pass (needed for global percentiles) but requires keeping every trial's running/pupil vectors in memory twice (`all_running`/`all_pupil` plus the per-trial dicts).
- The AI also ran the whole conversion more than once during development (full run, then two sample runs) — a workflow cost rather than a code cost.

ii.
```python
running_interp_source = interp_signal(running_df["timestamps"].to_numpy(dtype=np.float64),
                                      running_df["speed"].to_numpy(dtype=np.float64),
                                      np.array([ophys_timestamps[0]], dtype=np.float64))
...
for trial_id, trial_row in trials.iterrows():
    running_trial = interp_signal(running_df["timestamps"].to_numpy(dtype=np.float64),
                                  running_df["speed"].to_numpy(dtype=np.float64), target_times)
```

iii. Not discussed by the AI.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- The full-session `(n_neurons, T)` event matrix is stacked and cast to float32 for every experiment, but only ~1/3 of its columns are ever read (nearest-neighbour sampling); the rest of the work — and the data — is discarded.
- The whole-session `pupil_diameter` array (`2*sqrt(area/pi)` over ~150k frames) is computed before any trial filtering, including for the 3 experiments that are then dropped for invalid pupil.
- `running_interp_source` is computed and immediately discarded (used only as a truthiness test).
- `native_dt_ms` per experiment, the large `session_info` list (one dict of 13 fields per session) and `source_subset` are written into metadata but unused by the decoder.
- Trial-level `input_trials` arrays of shape `(0, T)` are allocated per trial; they carry no information (the task specifies no decoder inputs) though the validator does require the field.
- `change_detection = ....copy()` and `.sort_values()` per experiment, and `trial_stim.copy()` per trial, copy stimulus rows that are only read.
- The output pickle stores dense float32 matrices that are 99.8% zeros (2.7 GB), where a sparse or lower-precision representation would carry identical information.
- The extra deliverables produced along the way (`sample_data.pkl`, `sample_trials.png`, multiple log files) are not used by the evaluation.

ii.
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)   # full session
...
nearest = nearest_indices(ophys_timestamps, target_times)                    # ~1/3 of columns used
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
running_interp_source = interp_signal(..., np.array([ophys_timestamps[0]]))  # discarded
...
input_trials.append(np.zeros((0, target_times.shape[0]), dtype=np.float32))
...
"session_info": session_info,
"native_ophys_frame_interval_ms_summary": {...},
```

iii. The AI does not flag any of this as wasted work; the metadata extras are presented as provenance ("source_subset", "session_info") and the empty input arrays are deliberate ("empty decoder inputs represented as `(0, T)` arrays", `CONVERSION_NOTES.md`), which the validator accepts.
