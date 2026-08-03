# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent first enumerates locally present NWB files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, intersects those files with the Allen cache experiment table, sorts the resulting rows, filters out passive experiments, and then loads each surviving NWB file one at a time with `BehaviorOphysExperiment.from_nwb_path(...)`. Trials are not preloaded globally; each file is opened and then its trial table is processed inside the main conversion loop.

ii. 
```python
def available_experiment_table(data_dir: Path) -> pd.DataFrame:
    nwb_root = data_dir / NWB_DIRNAME
    local_paths = {}
    for path in sorted(nwb_root.glob("behavior_ophys_experiment_*.nwb")):
        ...
    cache = VisualBehaviorOphysProjectCache.from_local_cache(str(data_dir))
    experiments = cache.get_ophys_experiment_table()
    experiments = experiments.loc[experiments.index.intersection(local_paths.keys())].copy()
```

```python
experiments = experiments.loc[~experiments["passive"]].copy()
...
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. `CONVERSION_NOTES.md` says the converter used only NWB files physically present on disk and kept only active change-detection sessions. The trajectory also records the same pivot: step 69 says it would “load only NWB files that actually exist on disk” and “keep active task sessions.”

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by `mouse_id`. The converter casts each `mouse_id` to string, builds a unique `subjects` list, and records one `subject_idx` entry per kept session.

ii. 
```python
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)

subject_idx.append(subject_to_idx[subject])
```

iii. The notes repeatedly describe “subjects” as mice and report summary counts in mice/subjects. There is no alternative subject definition documented in the trajectory or notes.

## 1-c. How are the data split into sessions?

i. The agent treats each `BehaviorOphysExperiment` NWB file, meaning each experiment row in the experiment table, as one decoder session. It does not merge multiple experiments that share the same `ophys_session_id`.

ii. 
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    neural_sessions.append(neural_trials)
```

```python
session_info.append(
    {
        "experiment_id": int(experiment_id),
        "ophys_session_id": int(row["ophys_session_id"]),
        ...
    }
)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this: “Each decoder ‘session’ is one Allen `BehaviorOphysExperiment` NWB file, not one unique behavior session,” mainly to avoid merging multiscope imaging planes. The trajectory repeats that rationale in steps 200 and 258.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. For each kept row, the converter takes `start_time` and `stop_time`, builds a per-trial 100 ms time grid between them, and slices/aligned all signals onto that grid.

ii. 
```python
trials = dataset.trials.copy()
...
for trial_id, trial_row in trials.iterrows():
    start_time = float(trial_row["start_time"])
    stop_time = float(trial_row["stop_time"])
    target_times = make_target_times(start_time, stop_time, bin_size_s)
```

```python
def make_target_times(start_time: float, stop_time: float, bin_size_s: float) -> np.ndarray:
    edges = np.arange(start_time, stop_time, bin_size_s, dtype=np.float64)
    centers = edges + (0.5 * bin_size_s)
    return centers[centers < stop_time]
```

iii. The notes say “trials are segmented using the AllenSDK `trials` table,” and “Trials are segmented from `trial.start_time` to `trial.stop_time`.”

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are marked `go` or `catch`, and not `aborted` and not `auto_rewarded`. The code then applies extra practical exclusions: it skips trials that would have fewer than 2 target bins, skips trials whose aligned running or pupil signal is unavailable, skips trials with no recognized outcome, and drops sessions left with fewer than 2 kept trials.

ii. 
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
```

```python
if target_times.size < 2:
    continue
...
if running_trial is None or pupil_trial is None:
    continue
...
if outcome_idx is None:
    continue
...
if len(neural_trials) < 2:
    ...
    continue
```

iii. `CONVERSION_NOTES.md` explicitly names the `go`/`catch`, `aborted`, and `auto_rewarded` filter as matching the instructions. The extra bin-count and signal-availability checks are implementation-level safeguards; no explicit separate justification for them is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the AllenSDK event table, specifically `dataset.events["events"]`.

ii. 
```python
events_df = dataset.events
if len(events_df) == 0:
    ...
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says “Neural activity uses AllenSDK discrete calcium events from `dataset.events["events"]`,” and says this was chosen instead of dF/F or `filtered_events`.

## 2-b. How is the `neural` data processed?

i. The event traces from all ROIs in a session are vertically stacked into one neuron-by-time matrix. For each trial, the code builds a common 100 ms time grid and samples the full event matrix at the nearest ophys frame to each bin center.

ii. 
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The notes justify the event-trace choice and the 100 ms common bin size. Step 69 of the trajectory says the agent chose a common 100 ms grid so single-plane and multiscope sessions could coexist.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit per-neuron QC in `convert_data.py`. The script relies on whatever ROI validity filtering the AllenSDK has already applied to `dataset.events`. At the session level it skips experiments with no event table rows.

ii. 
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    ...
    continue
```

iii. `CONVERSION_NOTES.md` states that “AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically.” That is the only explicit QC justification documented for neural data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned trial-by-trial to the Allen trial start. Each trial uses timestamps running from `trial.start_time` to `trial.stop_time`, and neural samples are chosen by nearest ophys timestamp to those per-trial bin centers.

ii. 
```python
start_time = float(trial_row["start_time"])
stop_time = float(trial_row["stop_time"])
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

```python
"temporal_alignment_event": "trial start time from AllenSDK trials table",
"off_start": 0.0,
```

iii. The notes say temporal alignment is “based on ophys timestamps” and the metadata says the alignment event is “trial start time from AllenSDK trials table.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a uniform 100 ms bin size across all sessions and trials. Yes, temporal rebinning is applied: native ophys frames are resampled onto 100 ms trial-centered bins by nearest-neighbor selection.

ii. 
```python
TIME_BIN_MS_DEFAULT = 100.0
...
bin_size_s = time_bin_ms / 1000.0
...
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
```

iii. `CONVERSION_NOTES.md` explicitly says the converter uses “a common 100 ms bin size” because the local dataset mixes single-plane and multiscope experiments with native frame intervals ranging from about 32 ms to 93 ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`. Within those rows it uses `trials_id`, `image_name`, `omitted`, `start_time`, and `end_time`.

ii. 
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains(
        "change_detection", na=False
    )
].copy()
```

```python
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
for stim_row in trial_stim.itertuples():
    if bool(stim_row.omitted) or stim_row.image_name == "omitted":
        continue
```

iii. The notes explicitly say stimulus labels come only from `stimulus_presentations` rows in the `change_detection` block.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each trial starts with every time bin labeled `gray`. The converter then walks through each non-omitted flashed image interval in that trial and assigns the corresponding image label only during `start_time <= t < end_time`. All other bins remain `gray`.

ii. 
```python
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
...
if stim_row.image_name not in image_to_idx:
    image_to_idx[stim_row.image_name] = len(image_values)
    image_values.append(stim_row.image_name)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. `CONVERSION_NOTES.md` says “`image_identity` is the image shown during flashed image epochs” and “all other times are labeled `gray`,” with omissions left gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is defined on the same per-trial `target_times` grid as the neural data. Stimulus rows are matched to a trial via `trials_id`, then the flashed interval is projected onto the same bin centers used for `neural_trial`.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. The notes say all outputs were aligned to the same 100 ms trial timestamps and that stimulus labels were assigned during flashed image epochs on that grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the same filtered `change_detection` `stimulus_presentations` rows, specifically the boolean `is_change` flag together with the trial-linked stimulus intervals.

ii. 
```python
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. `CONVERSION_NOTES.md` says “`image_change` is `1` only during flashed epochs with `is_change == True`, else `0`.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. The converter initializes a zero vector per trial, then sets bins to 1 wherever a `change_detection` stimulus row for that trial is marked `is_change` and overlaps the current trial bin centers.

ii. 
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The notes justify this as matching the flashed-image structure of the task.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no continuous threshold. It is assigned directly as a binary categorical variable: `0` by default and `1` when `stim_row.is_change` is true for the relevant flashed interval.

ii. 
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The notes describe it as a direct binary label with values `["no_change", "change"]`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned on the same 100 ms `target_times` grid as the neural data. The trial-specific `is_change` interval is mapped to the bins used to sample `neural_trial`.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The notes say that running, pupil, neural activity, and stimulus labels are all aligned to the same trial-centered timestamps.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically the `timestamps` and `speed` columns.

ii. 
```python
running_df = dataset.running_speed
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. The notes say the converter uses “AllenSDK processed running speed.”

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates running speed onto each trial’s 100 ms bin centers, collects all kept running values across the dataset, computes global percentile edges, and later digitizes each trial’s interpolated running trace into quintile bins.

ii. 
```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
...
all_running.append(running_trial)
...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
```

iii. `CONVERSION_NOTES.md` says running speed is linearly interpolated to the common timestamps and discretized into global quintiles.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global percentile bins using the 20th, 40th, 60th, and 80th percentiles of all kept running samples, then `np.digitize(...)` assigns each time bin to a category.

ii. 
```python
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
running_bins = remap_to_bins(trial["running"], running_edges)
```

```python
def remap_to_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int16)
```

iii. The notes explicitly say “Running and pupil are discretized into global quintiles over all kept trial time bins.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to the same per-trial 100 ms `target_times` grid as `neural_trial` by linear interpolation from the running timestamps.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. The notes say running speed is “linearly interpolated to those timestamps,” referring to the common 100 ms trial timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically `timestamps` and `pupil_area`.

ii. 
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. `CONVERSION_NOTES.md` explicitly says pupil diameter is computed from AllenSDK `pupil_area`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code converts pupil area to an equivalent diameter with `2 * sqrt(area / pi)`, then linearly interpolates that diameter to each trial’s 100 ms time grid, aggregates all kept pupil values, and digitizes them later into global quintile bins.

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
...
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
```

iii. The notes justify the equivalent-diameter formula explicitly and say pupil is interpolated and then binned globally.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins using the 20th, 40th, 60th, and 80th percentiles of all kept pupil samples, then assigned with `np.digitize(...)`.

ii. 
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. The notes say pupil diameter is discretized into global quintiles.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned to the same 100 ms `target_times` grid as the neural data by interpolation from eye-tracking timestamps.

ii. 
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. The notes state that pupil diameter is “linearly interpolated to those timestamps.”

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns in the Allen trial table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

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
```

iii. The notes list the same four Allen trial outcome categories and the integer code mapping used in the converted outputs.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The converter maps the first true outcome flag in each trial row to an integer code and then broadcasts that single per-trial category across every time bin in the trial output matrix.

ii. 
```python
outcome_idx = outcome_to_index(trial_row)
...
def build_output_array(..., outcome_idx: int,) -> np.ndarray:
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` says `trial_outcome` is “static within each trial,” which is why the code repeats the same value for all bins in that trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or problematic data mostly by skipping them. `interp_signal(...)` drops non-finite timestamp/value pairs before interpolation; sessions with no valid running data are excluded; sessions with no finite pupil diameter are excluded; trials with missing aligned running/pupil or missing outcome labels are skipped; omitted stimuli remain labeled `gray`. At the session level, experiments with no event traces, too few ophys timestamps, or fewer than two kept trials are also excluded.

ii. 
```python
valid = np.isfinite(source_times) & np.isfinite(source_values)
if valid.sum() == 0:
    return None
```

```python
if running_interp_source is None:
    ...
    continue
...
if not np.isfinite(pupil_diameter).any():
    ...
    continue
...
if running_trial is None or pupil_trial is None:
    continue
...
if bool(stim_row.omitted) or stim_row.image_name == "omitted":
    continue
```

iii. The notes explicitly document the exclusion of three sessions with entirely invalid pupil data and note that omitted flashes remain gray. They do not claim any imputation beyond interpolation over valid samples.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is the per-experiment NWB load, trial-by-trial alignment, per-stimulus labeling inside each trial, and the second pass over all stored trial outputs to bin running and pupil. These are the dominant nested loops in the implementation.

ii. 
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    for trial_id, trial_row in trials.iterrows():
        ...
        for stim_row in trial_stim.itertuples():
            ...
```

```python
for output_temp_trials in output_temp_sessions:
    for trial in output_temp_trials:
        running_bins = remap_to_bins(trial["running"], running_edges)
        pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. There is no explicit efficiency discussion in the notes. The trajectory focuses on correctness and dataset compatibility, especially in step 69 when the 100 ms common grid was chosen for mixed acquisition modes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop, the per-trial stimulus loop, and the final second-pass binning loop could all be vectorized or at least partially batched. Repeated conversion of pandas columns to NumPy arrays inside the trial loop could also be hoisted out of the loop.

ii. 
```python
for trial_id, trial_row in trials.iterrows():
    ...
    running_trial = interp_signal(
        running_df["timestamps"].to_numpy(dtype=np.float64),
        running_df["speed"].to_numpy(dtype=np.float64),
        target_times,
    )
```

```python
for stim_row in trial_stim.itertuples():
    ...
```

```python
for output_temp_trials in output_temp_sessions:
    for trial in output_temp_trials:
        ...
```

iii. No explicit justification for leaving these loops unvectorized was documented. The code favors straightforward per-trial logic over bulk operations.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly converts the same running and eye-tracking columns to NumPy inside every trial, repeatedly subsets and sorts per-trial stimulus rows, and makes a two-pass output construction in which continuous running/pupil traces are first stored and later revisited for digitization.

ii. 
```python
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

```python
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
trial_stim = trial_stim.sort_values("start_time")
```

```python
output_temp_trials.append({... "running": running_trial, "pupil": pupil_trial, ...})
...
for output_temp_trials in output_temp_sessions:
    ...
```

iii. Again, there is no explicit documented justification beyond the general trajectory emphasis on producing a correct reproducible converter quickly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the temporary storage of continuous per-trial `running` and `pupil` arrays in `output_temp_trials`, which are only used to compute binned categories and are then thrown away. The one-point `running_interp_source` interpolation is also computed solely as an existence check and never reused. The decoder inputs are also populated with all-zero arrays because the schema requires an `input` field even though this task uses no decoder inputs.

ii. 
```python
running_interp_source = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    np.array([ophys_timestamps[0]], dtype=np.float64),
)
```

```python
input_trials.append(np.zeros((0, target_times.shape[0]), dtype=np.float32))
output_temp_trials.append(
    {
        "image_labels": image_labels,
        "change_labels": change_labels,
        "running": running_trial,
        "pupil": pupil_trial,
        "outcome_idx": outcome_idx,
    }
)
```

iii. No explicit justification was documented for these extra temporary structures. They appear to be convenience choices rather than reference-driven processing decisions.
