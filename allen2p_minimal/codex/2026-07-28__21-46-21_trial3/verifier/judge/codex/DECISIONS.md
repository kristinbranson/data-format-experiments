# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI first enumerates NWB files physically present under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, intersects those filenames with the AllenSDK experiment table from a local cache, sorts the resulting rows, filters out passive experiments later, and then loads each kept experiment directly from its NWB path with `BehaviorOphysExperiment.from_nwb_path(...)`. It does not use the AllenSDK S3 loader and does not reconstruct multi-plane sessions before loading data.

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
```

```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
```

iii. In the trajectory and the `CONVERSION_NOTES.md` patch, the agent justified this as using only the NWB files physically present in the local dataset and treating each on-disk `BehaviorOphysExperiment` file as the unit of conversion. It explicitly noted that the local copy is smaller than the full paper dataset and chose not to rely on remote S3 loading.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the experiment table rows that survive filtering. The AI stores them as strings and keeps a `subject_to_idx` mapping.

ii.
```python
subjects: list[str] = []
subject_to_idx: dict[str, int] = {}
...
subject = str(row["mouse_id"])
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
...
subject_idx.append(subject_to_idx[subject])
```

iii. In the trajectory, the agent repeatedly summarized the converted dataset in terms of the number of mice and said the local active cache contained 38 mice. That shows it was intentionally using `mouse_id` as the subject split.

## 1-c. How are the data split into sessions?

i. Each decoder session is one Allen `BehaviorOphysExperiment` NWB file, i.e. one experiment-table row. The AI does not group rows sharing the same `ophys_session_id`; it keeps experiment-plane files separate.

ii.
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    session_info.append(
        {
            "experiment_id": int(experiment_id),
            "ophys_session_id": int(row["ophys_session_id"]),
            ...
        }
    )
```

iii. The trajectory is explicit here: the agent wrote that each decoder “session” is one `BehaviorOphysExperiment` NWB file, not one unique behavior session, because it wanted to avoid “incorrectly merging different imaging planes from multiscope sessions into one neuron matrix.”

## 1-d. How are the data split into trials?

i. Trials are taken from `dataset.trials`. For each kept trial, the AI uses `trial_row["start_time"]` and `trial_row["stop_time"]` to define the trial window, then creates a regular 100 ms grid of bin centers inside that interval with `make_target_times(...)`.

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

iii. In the trajectory, the agent said the Allen trial table provides the behavior-defined `go` and `catch` trial types and that trials should be segmented from the AllenSDK `trials` table with ophys-timestamp alignment.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials where `go` or `catch` is true, and excludes `aborted` and `auto_rewarded` trials. It skips sessions with fewer than 2 such trials, skips trials whose 100 ms grid has fewer than 2 bins, drops trials lacking a valid outcome label, and can drop entire sessions for missing running or pupil data.

ii.
```python
trial_mask = (
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
)
trials = trials.loc[trial_mask].copy()
if len(trials) < 2:
    ...
```

```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
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

iii. In the trajectory and notes, the agent justified this as matching the instruction to keep only meaningful `go` and `catch` trials while excluding premature-lick resets and free-reward trials. It also logged extra session exclusions, especially fully invalid pupil sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the AllenSDK event traces, specifically the `events` column of `dataset.events`. It does not use `dff_traces.dff`.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    ...
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
```

iii. The trajectory and notes state this was intentional: the agent claimed the paper’s neural analyses used detected calcium events, so it “intentionally does not use dF/F traces and does not use `filtered_events`.”

## 2-b. How is the `neural` data processed?

i. After loading all event traces in an experiment, the AI stacks them into a neuron-by-time matrix for the whole experiment and then samples those traces at per-trial 100 ms target times by choosing the nearest ophys frame. There is no additional normalization or smoothing.

ii.
```python
neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)
...
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

```python
def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(source_times, target_times, side="left")
    ...
    return np.where(choose_right, right, left).astype(np.int64)
```

iii. In the trajectory, the agent justified the 100 ms target grid as a compatibility choice across single-plane and multiscope recordings with different native frame intervals, and justified nearest-frame sampling as alignment “based on ophys timestamps.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit per-neuron filtering beyond what the AllenSDK loader already returns. The only extra neural QC is that experiments with zero event rows are skipped entirely.

ii.
```python
events_df = dataset.events
if len(events_df) == 0:
    excluded_sessions.append({"experiment_id": int(experiment_id), "reason": "no_events"})
    print(f"  skip {experiment_id}: no valid event traces")
    continue
```

iii. In the trajectory, the agent said it had “confirmed the SDK excludes invalid ROIs when loading experiments” and treated that default AllenSDK filtering as the relevant baseline quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to the trial window from `start_time` to `stop_time`, with the trial effectively anchored at its start because the target time grid begins at `start_time + 50 ms`. Each bin is then mapped to the nearest ophys timestamp.

ii.
```python
start_time = float(trial_row["start_time"])
stop_time = float(trial_row["stop_time"])
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
```

iii. The trajectory says the converted dataset used “Allen trial start alignment” and “temporal alignment is based on ophys timestamps,” which is also reflected in `metadata["temporal_alignment_event"]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a common 100 ms bin size for all sessions. Yes, temporal rebinning/resampling is applied: trials are represented on a new 100 ms grid rather than on native ophys frames.

ii.
```python
TIME_BIN_MS_DEFAULT = 100.0
...
bin_size_s = time_bin_ms / 1000.0
...
"time_bin_size": float(time_bin_ms),
```

```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
nearest = nearest_indices(ophys_timestamps, target_times)
```

iii. In the trajectory and notes, the agent justified this by saying the local dataset mixes single-plane and multiscope files with different native frame intervals, so a shared 100 ms timebase was chosen “while staying close to the slower multiscope sampling interval.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`. Within each trial, the AI uses `trials_id`, `image_name`, `start_time`, `end_time`, and `omitted`, plus a synthetic `"gray"` state outside image flashes.

ii.
```python
stimulus_presentations = dataset.stimulus_presentations
change_detection = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"].str.contains(
        "change_detection", na=False
    )
].copy()
...
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
if bool(stim_row.omitted) or stim_row.image_name == "omitted":
    continue
```

iii. The trajectory notes justify this as matching the flashed-image task structure: use the image shown during non-gray flash epochs, label all other times as gray, and leave omissions gray.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI initializes every bin to the gray label, then overwrites bins that fall inside non-omitted flashed stimulus intervals with an integer image code. The `image_to_idx` mapping is built incrementally and globally across sessions.

ii.
```python
image_values = [GRAY_LABEL]
image_to_idx = {GRAY_LABEL: 0}
...
image_labels = np.full(target_times.shape[0], image_to_idx[GRAY_LABEL], dtype=np.int16)
...
if stim_row.image_name not in image_to_idx:
    image_to_idx[stim_row.image_name] = len(image_values)
    image_values.append(stim_row.image_name)
image_idx = image_to_idx[stim_row.image_name]
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. In the trajectory and notes, the agent justified this as a direct encoding of the flash/gray task cadence rather than a simpler pre-change/post-change label derived only from the trials table.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same per-trial `target_times` grid used for neural data. A bin gets an image label if its target time falls inside a flashed image interval; otherwise it stays gray.

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
image_labels[mask] = image_idx
```

iii. The trajectory explicitly says running, pupil, stimulus labels, and neural data were all aligned to the same 100 ms trial-centered timestamp grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the same `change_detection` subset of `stimulus_presentations`, specifically the per-stimulus `is_change` flag and each flashed interval’s `start_time` and `end_time`.

ii.
```python
trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. In the notes written during the trajectory, the agent justified this as using the actual flashed epochs marked as changed images, rather than using only trial-level metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI starts with an all-zero vector for the trial, then sets bins to 1 only for flashed image intervals whose stimulus row has `is_change == True`.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The trajectory notes state that `image_change` should be `1` “only during flashed epochs with `is_change == True`, else `0`.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already treated as a binary categorical variable with integer labels 0 and 1, corresponding to `["no_change", "change"]`. No additional thresholding beyond the `is_change` flag is applied.

ii.
```python
change_labels = np.zeros(target_times.shape[0], dtype=np.int16)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    ["hit", "miss", "false_alarm", "correct_reject"],
],
```

iii. There is no separate thresholding discussion in the trajectory; the justification is implicit in the agent’s choice to use the AllenSDK binary `is_change` annotation directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `image_change` is aligned on the same per-trial `target_times` grid as neural data. Bins are labeled according to whether their target time falls inside a flashed changed-image interval.

ii.
```python
target_times = make_target_times(start_time, stop_time, bin_size_s)
...
mask = (target_times >= float(stim_row.start_time)) & (
    target_times < float(stim_row.end_time)
)
if bool(stim_row.is_change):
    change_labels[mask] = 1
```

iii. The trajectory repeatedly describes one shared ophys-based timebase for neural and output variables, and the code implements that by using `target_times` everywhere.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
running_df = dataset.running_speed
...
running_df["timestamps"].to_numpy(dtype=np.float64)
running_df["speed"].to_numpy(dtype=np.float64)
```

iii. In the trajectory and notes, the agent described this as using “AllenSDK processed running speed.”

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto each trial’s 100 ms target grid with `np.interp`, then pools all kept trial values across the dataset and computes global quintile edges at the 20th, 40th, 60th, and 80th percentiles.

ii.
```python
def interp_signal(
    source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
) -> np.ndarray | None:
    ...
    interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
    return interp.astype(np.float32)
```

```python
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
```

iii. The trajectory notes justify this as “aligned by timestamp and binned into global quintiles.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI converts running speed to 5 discrete bins using `np.digitize` against global percentile edges.

ii.
```python
def remap_to_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int16)
...
running_edges = np.percentile(running_values, [20, 40, 60, 80]).astype(np.float64)
...
running_bins = remap_to_bins(trial["running"], running_edges)
```

iii. The trajectory justification is that global quintiles create the requested five percentile bins across the converted dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same per-trial `target_times` grid used to sample neural activity, so both outputs and neural data share trial-local 100 ms bins.

ii.
```python
nearest = nearest_indices(ophys_timestamps, target_times)
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
running_trial = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    target_times,
)
```

iii. The trajectory and notes explicitly state that running speed is “linearly interpolated to those timestamps.”

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically the `pupil_area` column and the corresponding `timestamps`.

ii.
```python
eye_df = dataset.eye_tracking
pupil_diameter = 2.0 * np.sqrt(
    np.asarray(eye_df["pupil_area"].to_numpy(dtype=np.float64)) / np.pi
)
```

iii. The trajectory notes say the agent intentionally used “equivalent diameter derived from AllenSDK processed `pupil_area`.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to equivalent diameter with `2 * sqrt(area / pi)`, interpolates that continuous signal onto each trial’s 100 ms target grid, and then computes global quintile bin edges from all kept trial bins. It does not remove blink frames.

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

iii. The trajectory notes justify this as using an “equivalent pupil diameter” measure aligned by timestamp and binned into global quintiles.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI maps pupil diameter to 5 bins with `np.digitize` using dataset-wide 20/40/60/80 percentile edges.

ii.
```python
pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80]).astype(np.float64)
...
pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)
```

iii. The trajectory says pupil was handled the same way as running: global quintiles across all kept trial time bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated directly onto each trial’s `target_times`, the same grid used for neural data and other outputs.

ii.
```python
neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)
pupil_trial = interp_signal(
    eye_df["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    target_times,
)
```

iii. In the trajectory and notes, the agent described pupil as being “aligned by timestamp” to the same common per-trial timebase.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The trajectory notes explicitly list those four AllenSDK outcome categories and give the 0/1/2/3 encoding used in the converted dataset.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps each trial to a single integer outcome code and repeats that code across all time bins in the trial when constructing the output array.

ii.
```python
outcome_idx = outcome_to_index(trial_row)
...
def build_output_array(
    image_labels: np.ndarray,
    change_labels: np.ndarray,
    running_bins: np.ndarray,
    pupil_bins: np.ndarray,
    outcome_idx: int,
) -> np.ndarray:
    t = image_labels.shape[0]
    outcome = np.full(t, outcome_idx, dtype=np.int16)
    return np.vstack([... , outcome])
```

iii. The trajectory notes justify this as making `trial_outcome` static within each trial while still matching the decoder’s time-varying output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by skipping whole experiments that fail to load, have too few ophys timestamps, no event traces, no running signal, or no valid pupil values. It skips individual trials with too few 100 ms bins, missing running/pupil interpolation results, or no recognized outcome. For interpolation, it filters non-finite source samples and uses edge-value extrapolation outside the observed time range.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
except Exception as exc:
    excluded_sessions.append(
        {"experiment_id": int(experiment_id), "reason": f"load_failed:{type(exc).__name__}"}
    )
    continue
```

```python
if ophys_timestamps.size < 2:
    ...
if len(events_df) == 0:
    ...
if running_interp_source is None:
    ...
if not np.isfinite(pupil_diameter).any():
    ...
```

```python
interp = np.interp(target_times, times, values, left=values[0], right=values[-1])
```

iii. In the trajectory, the agent highlighted that three sessions were excluded for fully invalid pupil data and described sparse all-zero event trials as acceptable rather than erroneous.

## 9-a. What are the most time-consuming steps of the code?

i. The heaviest work is loading each NWB experiment file and iterating through all experiments/trials to build per-trial outputs. Per-trial interpolation and per-trial stimulus masking add additional cost, but the dataset load/scan is the dominant step.

ii.
```python
for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):
    ...
    dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])
    ...
    for trial_id, trial_row in trials.iterrows():
        ...
```

iii. The trajectory explicitly described the full conversion as “CPU-bound inside the Python loader” while it was iterating through a few hundred NWB files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not vectorize the per-trial loop over `trials.iterrows()`, the per-stimulus loop over `trial_stim.itertuples()`, or the repeated interpolation calls for running and pupil within each trial. Those are the clearest vectorization opportunities.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
    running_trial = interp_signal(...)
    pupil_trial = interp_signal(...)
    ...
    for stim_row in trial_stim.itertuples():
        ...
```

iii. No explicit vectorization rationale appears in the trajectory. The implementation suggests the agent prioritized a simple, inspectable trial-by-trial conversion pipeline over more aggressive vectorization.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly converts the same `running_df` and `eye_df` columns to NumPy arrays inside every trial, repeatedly interpolates running and pupil separately for each trial, and repeatedly filters/sorts the trial-specific stimulus table inside the trial loop.

ii.
```python
for trial_id, trial_row in trials.iterrows():
    ...
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
    ...
    trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id].copy()
    trial_stim = trial_stim.sort_values("start_time")
```

iii. The trajectory does not contain an explicit justification for these repetitions. They appear to be a byproduct of the chosen straightforward per-trial implementation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded computation is `running_interp_source`, which is computed only to test whether running interpolation is possible and is never used afterward. The script also computes and stores substantial metadata (`session_info`, native frame interval summaries, exclusion logs) that are useful for provenance but not consumed by the downstream decoder itself.

ii.
```python
running_interp_source = interp_signal(
    running_df["timestamps"].to_numpy(dtype=np.float64),
    running_df["speed"].to_numpy(dtype=np.float64),
    np.array([ophys_timestamps[0]], dtype=np.float64),
)
if running_interp_source is None:
    ...
```

```python
metadata = {
    ...
    "source_subset": {...},
    "session_info": session_info,
    "native_ophys_frame_interval_ms_summary": {...},
}
```

iii. The trajectory justifies the extra metadata as documentation and sanity-check support. It does not provide a separate justification for the unused `running_interp_source` probe.
