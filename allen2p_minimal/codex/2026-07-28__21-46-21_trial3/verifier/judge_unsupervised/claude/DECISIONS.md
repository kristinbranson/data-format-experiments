# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the AllenSDK `VisualBehaviorOphysProjectCache.from_local_cache()` to discover all locally available NWB experiment files. It cross-references the experiment table with actual NWB files on disk under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`. Each experiment is loaded individually via `BehaviorOphysExperiment.from_nwb_path()`. Only experiments with NWB files physically present are considered.

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
    ...
```
And in the main loop:
```python
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. The agent stated in CONVERSION_NOTES.md: "Only NWB files physically present under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` were used." The trajectory (step 69) confirms the design to "load only NWB files that actually exist on disk."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mouse_id` field from the experiment table. A running list of unique subjects is maintained; each session is mapped to its subject via `subject_to_idx`. The final `subjects` list contains 38 unique mouse IDs as strings.

ii.
```python
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The CONVERSION_NOTES.md confirms: "38 mice" in the local cache. The agent uses the experiment table's `mouse_id` column, which is standard for the Allen SDK.

## 1-c. How are the data split into sessions?

i. Each decoder "session" corresponds to one Allen `BehaviorOphysExperiment` (one NWB file), not one unique behavior session. Multiple imaging planes from the same behavior session are treated as separate decoder sessions. Only active (non-passive) sessions are included (`passive == False` filter). This yields 199 sessions from 171 unique ophys sessions.

ii.
```python
experiments = experiments.loc[~experiments["passive"]].copy()
...
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. CONVERSION_NOTES.md: "Each decoder 'session' is one Allen BehaviorOphysExperiment NWB file, not one unique behavior session. This matches the AllenSDK data model and avoids incorrectly merging different imaging planes from multiscope sessions into one neuron matrix."

## 1-d. How are the data split into trials?

i. Trials are obtained from the AllenSDK `dataset.trials` table. Each trial has a `start_time` and `stop_time`. The code creates regular time bins within each trial window using `make_target_times(start_time, stop_time, bin_size_s)` with 100 ms bins. Trials with fewer than 2 time bins are skipped.

ii.
```python
trials = dataset.trials.copy()
...
for trial_id, trial_row in trials.iterrows():
    start_time = float(trial_row["start_time"])
    stop_time = float(trial_row["stop_time"])
    target_times = make_target_times(start_time, stop_time, bin_size_s)
    if target_times.size < 2:
        continue
```

iii. The agent chose to segment trials using the SDK's trial table boundaries, aligned to trial start time. CONVERSION_NOTES.md: "Trials are segmented from `trial.start_time` to `trial.stop_time`."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, trials where the outcome cannot be determined (not hit, miss, false_alarm, or correct_reject), trials with fewer than 2 time bins, and trials where running or pupil interpolation fails are excluded. Sessions with fewer than 2 valid trials are excluded entirely.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
if len(trials) < 2:
    ...continue
```
Per-trial filtering:
```python
if target_times.size < 2:
    continue
...
if running_trial is None or pupil_trial is None:
    continue
...
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
```

iii. CONVERSION_NOTES.md: "Per instructions and Allen trial definitions, trials are kept only when: go == True or catch == True, aborted == False, auto_rewarded == False." This matches the instructions exactly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the AllenSDK `dataset.events` table, specifically the `"events"` column, which contains discrete detected calcium events (not dF/F or raw fluorescence).

ii.
```python
events_df = dataset.events
...
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. CONVERSION_NOTES.md: "Neural activity uses AllenSDK discrete calcium events from `dataset.events['events']`. This matches the paper's statement that neural analyses were performed on detected calcium events rather than raw fluorescence." Trajectory step 16 shows the agent deliberated between dF/F and event traces.

## 2-b. How is the `neural` data processed?

i. The raw events are stacked into a full neuron-by-time matrix for the entire session. For each trial, the nearest ophys timestamp indices are found for each target time bin, and neural data is sampled at those indices. No additional normalization, z-scoring, or smoothing is applied. The data is cast to float32.

ii.
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md: "neural samples are aligned by nearest ophys timestamp." The agent does not apply any additional processing beyond what the AllenSDK already provides.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI relies on AllenSDK default ROI validity filtering, which automatically excludes invalid ROIs when loading each `BehaviorOphysExperiment`. No additional neuron-level filtering (e.g., based on SNR or activity level) is applied. Sessions with no events are excluded.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    continue
```

iii. CONVERSION_NOTES.md: "AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically when loading each BehaviorOphysExperiment."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to **ophys timestamps** as specified in the instructions. For each trial, target time bins start from `trial.start_time` and go to `trial.stop_time` with 100 ms spacing. The nearest ophys timestamp index is found for each target time bin center, and the neural value at that index is used.

ii.
```python
ophys_timestamps = np.asarray(dataset.ophys_timestamps, dtype=np.float64)
...
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest]
```

iii. The instructions say "Temporally align based on ophys timestamp." The agent uses `dataset.ophys_timestamps` and nearest-neighbor indexing to align neural data to the target time grid. CONVERSION_NOTES.md confirms this approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 100 ms (default). This represents rebinning since the native ophys frame intervals range from ~32 ms (single-plane) to ~93 ms (multiscope). The rebinning is done via nearest-neighbor sampling of ophys frames, not averaging within bins.

ii.
```python
TIME_BIN_MS_DEFAULT = 100.0
...
bin_size_s = time_bin_ms / 1000.0
...
def make_target_times(start_time, stop_time, bin_size_s):
    edges = np.arange(start_time, stop_time, bin_size_s)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]
```

iii. Trajectory step 69: "resample to a common 100 ms grid so single-plane and multiscope sessions can coexist in one decoder dataset." CONVERSION_NOTES.md: "a common 100 ms bin size is used for every session."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically from rows in the `change_detection` stimulus block. The `image_name` column provides the identity of each image.

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains("change_detection", na=False)
].copy()
...
image_idx = image_to_idx[stim_row.image_name]
```

iii. CONVERSION_NOTES.md: "Stimulus labels come only from stimulus_presentations rows whose stimulus_block_name contains change_detection."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each time bin is labeled with the identity of the image being displayed at that time. If a stimulus presentation overlaps the time bin (`start_time <= target_time < end_time`) and is not omitted, the bin receives the corresponding image index. All other bins are labeled as "gray" (index 0). Omitted stimuli remain labeled as gray.

ii.
```python
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
...
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
    ...
    mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
    image_labels[mask] = image_idx
```

iii. CONVERSION_NOTES.md: "image_identity is the image shown during flashed image epochs; all other times are labeled gray; omissions remain gray."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity labels are computed on the same target time grid as the neural data (the 100 ms bin centers within each trial), so they are inherently aligned. Each time bin has both a neural data value (from nearest ophys frame) and an image label (from stimulus presentation overlap).

ii.
The same `target_times` array is used for both neural indexing and stimulus labeling:
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest]
...
mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
image_labels[mask] = image_idx
```

iii. All outputs are computed on the same time grid, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table (filtered to the change_detection block).

ii.
```python
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. CONVERSION_NOTES.md: "image_change is 1 only during flashed epochs with is_change == True, else 0."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each trial, a binary time series is initialized to 0. For each stimulus presentation in the change_detection block that overlaps the trial and has `is_change == True`, the corresponding time bins are set to 1. Only the duration of the change image presentation is marked (not a single time point).

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The instructions state: "Have value of 1 right after a change in image identity, otherwise 0." The agent marks the entire change stimulus epoch as 1, which covers the time bins during the change image presentation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is applied; it is determined directly from the `is_change` flag.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```
Output values: `["no_change", "change"]`

iii. The instructions specify this as a "binary variable," so no additional discretization is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as Image identity - computed on the same target time grid as neural data.

ii. Same `target_times` and `mask` logic as for image identity.

iii. Alignment is inherent because all signals share the same time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides a DataFrame with `timestamps` and `speed` columns.

ii.
```python
running_df = dataset.running_speed
...
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. CONVERSION_NOTES.md: "AllenSDK processed running speed aligned by timestamp and binned into global quintiles."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from the SDK's running speed timestamps to the target 100 ms time bins. Invalid/NaN values are filtered before interpolation. Values outside the source time range are extrapolated using the first/last valid values.

ii.
```python
def interp_signal(source_times, source_values, target_times):
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    ...
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)
```

iii. CONVERSION_NOTES.md: "running speed is linearly interpolated to those timestamps."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global quintiles (20th, 40th, 60th, 80th percentile edges) computed across all time bins in all kept trials. `np.digitize` with `right=False` is used to assign bin indices (0-4).

ii.
```python
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
running_bins = remap_to_bins(trial["running"], running_edges)
...
def remap_to_bins(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int16)
```

iii. CONVERSION_NOTES.md: "Running and pupil are discretized into global quintiles over all kept trial time bins." The instructions state: "discretized into five equal percentile bins."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same target time bins as neural data, ensuring alignment.

ii.
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. Same `target_times` used for all modalities.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically the `pupil_area` column. The area is converted to equivalent diameter.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. CONVERSION_NOTES.md: "Equivalent pupil diameter derived from AllenSDK processed pupil_area."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area from eye tracking is converted to equivalent diameter via `diameter = 2 * sqrt(area / pi)`. The resulting diameter is linearly interpolated from eye tracking timestamps to the target 100 ms time bins. Invalid (NaN/Inf) values are filtered before interpolation.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
...
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. CONVERSION_NOTES.md: "Equivalent pupil diameter derived from AllenSDK processed pupil_area, aligned by timestamp and binned into global quintiles."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: discretized into 5 bins using global quintiles (20th, 40th, 60th, 80th percentiles) computed over all time bins across all kept trials.

ii.
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. CONVERSION_NOTES.md: "Running and pupil are discretized into global quintiles over all kept trial time bins." The instructions specify "five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same target time bins as neural data, ensuring alignment.

ii.
```python
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. Same `target_times` used for all modalities.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from four boolean columns in the AllenSDK `trials` table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. CONVERSION_NOTES.md: "trial_outcome is static within each trial and encoded as: 0: hit, 1: miss, 2: false_alarm, 3: correct_reject."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is determined by checking the boolean columns in priority order (hit > miss > false_alarm > correct_reject). The first True value determines the outcome. If none is True, the trial is excluded. The outcome is a static per-trial value that is broadcast to all time bins within the trial.

ii.
```python
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
outcome = np.full(t, outcome_idx, dtype=np.int16)
```

iii. The instructions specify "Trial outcome. Static per-trial." The agent makes it static by filling all time bins with the same value.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data scenarios are handled:
- Sessions with too few ophys timestamps (< 2) are skipped.
- Sessions with no event traces are skipped.
- Sessions with entirely invalid pupil data are skipped (3 sessions excluded).
- Sessions with no running signal are skipped.
- Individual trials where running or pupil interpolation fails are skipped.
- Trials with no determinable outcome are skipped.
- NaN/Inf values in running speed and pupil data are filtered in the `interp_signal` function before interpolation.
- Omitted stimulus presentations are treated as gray.

ii.
```python
if ophys_timestamps.size < 2:
    ...continue
if len(events_df) == 0:
    ...continue
if not np.isfinite(pupil_diameter).any():
    ...continue
# In interp_signal:
valid = np.isfinite(source_times) & np.isfinite(source_values)
if valid.sum() == 0:
    return None
```

iii. CONVERSION_NOTES.md documents that 3 sessions were excluded for invalid pupil data. The agent's approach is conservative: skip trials/sessions with missing data rather than imputing.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()` - this involves parsing large HDF5/NWB files with neural, stimulus, and behavioral data.
2. The main loop over 202 candidate experiments, each requiring NWB loading.
3. Interpolation of running and pupil signals for every trial.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```
This is called 202 times in the main loop.

iii. The trajectory shows the full conversion took substantial time, and the agent planned for a "long CPU job."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The inner loop over `trial_stim.itertuples()` for assigning stimulus labels could use vectorized numpy operations with searchsorted or similar.
2. The loop over trials for interpolating running and pupil could potentially be batched if all target times were concatenated first and split afterward.
3. The `interp_signal` function is called separately for running and pupil for each trial, repeating the NaN filtering and sorting of the same source data.

ii.
```python
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
    ...
    mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))
    image_labels[mask] = image_idx
```

iii. No explicit justification was given for the loop-based approach, but the per-trial loop structure is straightforward and correct.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats:
1. Converting running_df timestamps and speed to numpy arrays for every trial (`running_df["timestamps"].to_numpy()` and `running_df["speed"].to_numpy()`), even though these are the same within a session.
2. Similarly for eye tracking data - `eye_df["timestamps"].to_numpy()` is called for every trial.
3. The `interp_signal` function re-filters NaN values, sorts, and deduplicates the source data on every call, even though this is the same across trials within a session.

ii.
```python
# Called once per trial, but source data is session-level:
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. No justification given. The repeated conversions are an inefficiency but don't affect correctness.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. The initial running speed interpolation test at session level (`running_interp_source`) is computed solely to check that running data exists, then discarded.
2. The `input` arrays are empty (`(0, T)` shaped) since the instructions specify "No inputs for this task." The code still creates and stores these empty arrays for every trial.
3. Session info metadata is computed in detail but may not be used by the decoder.

ii.
```python
# Test interpolation that's discarded:
running_interp_source = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    np.array([ophys_timestamps[0]], dtype=np.float64),
)
if running_interp_source is None:
    ...continue

# Empty input arrays:
input_trials.append(np.zeros((0, target_times.shape[0]), dtype=np.float32))
```

iii. The empty inputs are required by the target data format even though no decoder inputs were specified. The running check is a minor inefficiency for validation purposes.
