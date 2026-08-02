# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers locally available NWB experiment files by scanning the directory `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` for `.nwb` files, then cross-references them against the `VisualBehaviorOphysProjectCache` experiment table loaded via `from_local_cache`. Only experiments whose NWB files are physically present on disk are used. Each experiment is loaded individually using `BehaviorOphysExperiment.from_nwb_path()`.

ii.
```python
def available_experiment_table(data_dir: Path) -> pd.DataFrame:
    nwb_root = data_dir / NWB_DIRNAME
    local_paths = {}
    for path in sorted(nwb_root.glob("behavior_ophys_experiment_*.nwb")):
        match = re.search(r"(\d+)\.nwb$", path.name)
        if match:
            local_paths[int(match.group(1))] = path
    ...
    cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
    experiments = cache.get_ophys_experiment_table()
    experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
    ...

# Loading each experiment:
dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. The AI explicitly loads from local NWB files rather than the S3 cache, and scans for physically present files. The trajectory shows the agent decided to "load only NWB files that actually exist on disk."

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values encountered during processing. Each unique mouse_id is registered as a new subject when first seen.

ii.
```python
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

iii. The mouse_id field from the experiment table uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a single Allen `BehaviorOphysExperiment` (one NWB file = one imaging plane), NOT grouped by `ophys_session_id`. This means multi-plane sessions are treated as separate decoder sessions rather than being merged. Passive sessions are filtered out (`passive == False`). Both `VisualBehavior` and `VisualBehaviorMultiscope` project codes are included.

ii.
```python
experiments = experiments.loc[~experiments["passive"]].copy()
...
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. From CONVERSION_NOTES.md: "Each decoder 'session' is one Allen BehaviorOphysExperiment NWB file, not one unique behavior session. This matches the AllenSDK data model and avoids incorrectly merging different imaging planes from multiscope sessions into one neuron matrix."

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `dataset.trials` table. Go and catch trials are kept; aborted and auto-rewarded trials are excluded. Each trial uses the `start_time` to `stop_time` window. Time bins are created at 100 ms intervals (bin centers) within this window. Trials with fewer than 2 time bins are skipped.

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

iii. The AI followed the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) requiring `go` or `catch` to be True, (2) excluding `aborted` trials, (3) excluding `auto_rewarded` trials, (4) requiring at least 2 time bins in the trial window, (5) requiring valid running and pupil signals, (6) requiring a valid outcome (hit/miss/false_alarm/correct_reject). Sessions with fewer than 2 valid trials are excluded.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
...
if target_times.size < 2:
    continue
if running_trial is None or pupil_trial is None:
    continue
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
if len(neural_trials) < 2:
    ...continue
```

iii. The AI applied multiple layers of filtering: trial type, minimum duration, data availability, and outcome validity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events["events"]` — the AllenSDK's discrete calcium events (deconvolved events), NOT dF/F traces.

ii.
```python
events_df = dataset.events
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Neural activity uses AllenSDK discrete calcium events from dataset.events['events']. This matches the paper's statement that neural analyses were performed on detected calcium events rather than raw fluorescence." The trajectory shows the agent investigated both dF/F and events and chose events.

## 2-b. How is the `neural` data processed?

i. Neural data is sampled at the nearest ophys timestamp to each 100 ms time bin center. No additional normalization or filtering is applied beyond what the SDK provides.

ii.
```python
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The AI chose nearest-neighbor sampling rather than interpolation for neural data, which preserves the discrete event nature of the signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI relies on AllenSDK's default ROI validity filtering. Experiments with no valid event traces (`len(events_df) == 0`) are excluded. No additional neuron-level quality filtering is applied.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    continue
```

iii. From CONVERSION_NOTES.md: "AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically when loading each BehaviorOphysExperiment."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time (`start_time` from the trials table). For each trial, a grid of 100 ms time bins is created from `start_time` to `stop_time`. Neural data is sampled at the nearest ophys timestamp to each bin center.

ii.
```python
def make_target_times(start_time, stop_time, bin_size_s):
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]

nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The AI uses trial start as the alignment event. The `make_target_times` function creates evenly spaced bin centers at 100 ms intervals.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to 100 ms time bins. This is a deliberate rebinning from the native ophys frame rate (~30-93 ms depending on acquisition mode) to a common 100 ms grid.

ii.
```python
TIME_BIN_MS_DEFAULT = 100.0
...
bin_size_s = time_bin_ms / 1000.0
target_times = make_target_times(start_time, stop_time, bin_size_s)
```

iii. From CONVERSION_NOTES.md: "Using 100 ms bins preserves compatibility across both acquisition modes while staying close to the slower multiscope sampling interval." The AI includes both single-plane (~11 Hz / ~93 ms) and multiscope (~30 Hz / ~32 ms) experiments, so a common bin size was needed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically rows where `stimulus_block_name` contains "change_detection". The `image_name` column from each stimulus presentation is used. Periods without a stimulus flash are labeled as "gray".

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains(
        "change_detection", na=False
    )
].copy()
...
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
    image_labels[mask] = image_to_idx[stim_row.image_name]
```

iii. The AI used the stimulus_presentations table rather than the trials table's `initial_image_name`/`change_image_name` columns, providing finer-grained timing of when each image flash is actually on screen vs. during gray inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global dictionary. A "gray" label (code 0) is used as the default for all time bins, then overwritten with the actual image code during each stimulus flash epoch (`start_time` to `end_time` of each stimulus presentation). Omitted stimuli remain gray.

ii.
```python
GRAY_LABEL = "gray"
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
    mask = (target_times >= float(stim_row.start_time)) & (
        target_times < float(stim_row.end_time)
    )
    image_labels[mask] = image_idx
```

iii. The AI includes "gray" as a distinct image category (index 0). Image codes are assigned dynamically as new images are encountered.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100 ms time bin centers as the neural data. Each bin is labeled based on whether its center time falls within a stimulus presentation epoch.

ii.
```python
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. By using the same `target_times` array for both neural and output data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table (filtered to the change_detection block).

ii.
```python
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The AI uses the SDK's `is_change` flag from stimulus presentations rather than computing it from the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array initialized to 0 is set to 1 during stimulus flash epochs where `is_change == True`. The change label is 1 only during the flash duration (250 ms), not during the subsequent gray period.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The AI marks image change only during the actual stimulus flash, as determined by the stimulus_presentations start_time and end_time.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable: 0 = no change, 1 = change. No thresholding is needed.

ii.
```python
["no_change", "change"]
```

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the same 100 ms time bin centers as the neural data using `target_times`.

ii. See 3-c.

iii. Same alignment approach as all other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
running_df = dataset.running_speed
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the 100 ms time bin centers. NaN/non-finite values in the source are excluded before interpolation. Values are then discretized into 5 equal percentile bins (quintiles) computed globally across all sessions. Bin edges are at the 20th, 40th, 60th, and 80th percentiles.

ii.
```python
def interp_signal(source_times, source_values, target_times):
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    ...
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)

running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)

def remap_to_bins(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int16)
```

iii. Global percentile-based binning ensures roughly equal class counts. The AI uses `np.interp` with edge clamping (left/right fill) rather than `scipy.interpolate.interp1d` with NaN fill.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using 4 edges at the 20th, 40th, 60th, and 80th percentiles computed globally. `np.digitize` maps values to bins 0-4.

ii.
```python
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. This produces 5 quintile bins with approximately equal numbers of samples.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly to the same 100 ms `target_times` array used for neural data, ensuring alignment.

ii.
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. Same time bin centers for all data streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_area` column. The diameter is computed as `2 * sqrt(pupil_area / pi)`.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. The AI computes equivalent circular diameter from pupil area, rather than using `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to equivalent diameter, then linearly interpolated to the 100 ms time bin centers. NaN/non-finite values are excluded during interpolation. The result is discretized into 5 quintile bins using global percentile edges. No explicit blink removal is performed before interpolation.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
...
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. The AI's `interp_signal` function filters out non-finite values before interpolation, which partially handles blink artifacts (since `pupil_area` may be NaN during blinks). However, there is no explicit use of the `likely_blink` flag.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 quintile bins using global percentile edges at the 20th, 40th, 60th, and 80th percentiles.

ii.
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. Same global percentile approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the same 100 ms `target_times` array.

ii.
```python
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. Same alignment approach as all other time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

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

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task. Trials where none of these are True are excluded (returns None).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is constant across all time bins within a trial. Trials with no matching outcome are skipped entirely.

ii.
```python
outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
...
outcome = np.full(t, outcome_idx, dtype=np.int16)
```

iii. The mapping is consistent with the reference. Trials without a valid outcome are excluded rather than assigned a fallback.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiment loads**: Skipped with logging via try/except.
- **Missing ophys timestamps**: Experiments with < 2 timestamps are skipped.
- **Missing events**: Experiments with no event traces are skipped.
- **Missing running/pupil**: Trials with None running or pupil signals are skipped. Sessions with entirely invalid pupil data are excluded (3 experiments).
- **NaN values in behavioral signals**: `interp_signal` filters out non-finite values before interpolation and clamps extrapolation to edge values.
- **Too few trials**: Sessions with < 2 valid trials after filtering are excluded.
- **No valid outcome**: Trials where no outcome flag matches are skipped.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
except Exception as exc:
    ...continue

if ophys_timestamps.size < 2:
    ...continue

if running_trial is None or pupil_trial is None:
    continue

outcome_idx = outcome_to_index(trial_row)
if outcome_idx is None:
    continue
```

iii. The AI applied defensive checks at multiple levels, logging exclusions in the `excluded_sessions` metadata for traceability.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `BehaviorOphysExperiment.from_nwb_path()`, which reads large NWB files containing neural traces, behavioral data, and stimulus presentations. This is I/O bound.

ii. N/A

iii. Each NWB file contains full-session data for all neurons plus behavioral and stimulus streams.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop iterates over each valid trial sequentially, performing interpolation, stimulus label assignment, and data extraction. The stimulus label assignment inner loop (iterating over `trial_stim` rows) could potentially be vectorized with interval operations. The `interp_signal` function is called separately for running and pupil in each trial rather than being vectorized across all trials.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
    for stim_row in trial_stim.itertuples():
        ...
```

iii. The per-trial and per-stimulus-presentation loops are not performance bottlenecks compared to data loading.

## 9-c. What processing does the code repeat multiple times?

i. The `interp_signal` function for running speed is called once to check data availability (with a single target time) and then again per trial. The running and pupil raw data arrays are re-extracted from the DataFrame on every trial iteration rather than being precomputed once per session.

ii.
```python
# First call: availability check
running_interp_source = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    np.array([ophys_timestamps[0]], dtype=np.float64),
)

# Then per trial:
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. The `.to_numpy()` calls inside the trial loop recreate numpy arrays from the DataFrame on each iteration.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes pupil diameter as `2*sqrt(area/pi)` from `pupil_area`, but this monotonic transformation is unnecessary since the result is immediately discretized into percentile bins (the bin assignments would be identical if using `pupil_area` directly, since percentile ranks are preserved under monotonic transformations).

ii.
```python
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. Percentile-based binning is invariant to monotonic transformations, so the area-to-diameter conversion is mathematically redundant for the downstream discretization step.
