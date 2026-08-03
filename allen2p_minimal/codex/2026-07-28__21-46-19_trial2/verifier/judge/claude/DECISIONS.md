# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache()` to access the Allen SDK cache, then calls `get_ophys_experiment_table()` to list all experiments. It filters to experiments that are (a) locally available on disk (NWB files present), (b) have `project_code` in `['VisualBehavior', 'VisualBehaviorMultiscope']`, (c) have `behavior_type == 'active_behavior'`, and (d) have `targeted_structure` in `['VISp', 'VISl']`. Each experiment is loaded individually via `cache.get_behavior_ophys_experiment(experiment_id)`.

ii.
```python
def select_experiments(cache, cache_dir):
    experiment_table = cache.get_ophys_experiment_table().reset_index()
    local_ids = get_local_experiment_ids(cache_dir)
    selected = experiment_table[
        (experiment_table["ophys_experiment_id"].isin(local_ids))
        & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
        & (experiment_table["behavior_type"] == "active_behavior")
        & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
    ].copy()
```

iii. The AI explains in CONVERSION_NOTES.md that the local cache only has a subset of experiments, so it restricts to locally available NWB files. It includes both VisualBehavior and VisualBehaviorMultiscope project codes and filters to VISp/VISl and active_behavior to match the paper's focus on visual cortex during the active task.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values extracted from each successfully processed `SessionData` object's metadata.

ii.
```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Mouse IDs come from the SDK's per-experiment metadata. The sorted set ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each ophys **experiment** (single imaging plane) is treated as its own session. The AI does NOT group multiple experiments from the same `ophys_session_id` together. Each experiment is processed independently.

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
    if session is not None:
        session_data.append(session)
```

iii. The AI treats each experiment as an independent session. The CONVERSION_NOTES.md does not explicitly discuss the choice to not merge experiments from the same ophys session.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's `ds.trials` table, filtered to keep only go/catch trials that are not aborted and not auto-rewarded. Within each trial, the AI uses the `stimulus_presentations` table (filtered to active `change_detection` block) to define the time bins. Stimulus presentations are matched to trials via the `trials_id` column. Each trial thus consists of a sequence of image-presentation intervals.

ii.
```python
trials = ds.trials.copy()
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
kept_trials = trials.loc[keep_trials].copy()
...
stim = make_change_detection_table(ds)
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
```

iii. The AI uses `go | catch` as the inclusion criterion (equivalent to excluding aborted and auto-rewarded, but more explicit). The stimulus_presentations table is used to define time bins within each trial, matching the paper's image-interval analysis approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to exclude aborted and auto-rewarded trials, keeping only go and catch trials. Additionally, entire experiments are skipped if they have no valid ROI events, no eye tracking data, all-NaN pupil width, or fewer than 2 kept trials. Trials with no matching stimulus presentations are also skipped.

ii.
```python
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
...
if len(ds.events) == 0: return None
if ds.eye_tracking.empty: return None
if not np.isfinite(pupil_width).any(): return None
...
if len(trial_data) < 2: return None
```

iii. The AI documents in CONVERSION_NOTES.md that 3 experiments were skipped due to missing eye tracking data. The go/catch filter matches the instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `ds.events` (deconvolved calcium events from the Allen SDK), NOT from `dff_traces` (dF/F). The `events` attribute provides inferred discrete neural events rather than the continuous fluorescence trace.

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. The CONVERSION_NOTES.md states: "Neural activity is taken from BehaviorOphysExperiment.events. This follows the paper's use of discrete calcium events rather than dF/F traces."

## 2-b. How is the `neural` data processed?

i. The event traces are summed within each image-presentation interval using a cumulative sum approach (`interval_reduce_sum_matrix`). This produces one value per neuron per ~750ms interval, rather than preserving the native ophys frame rate.

ii.
```python
def interval_reduce_sum_matrix(timestamps, matrix, starts, ends):
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate([
        np.zeros((matrix.shape[0], 1), dtype=np.float64),
        np.cumsum(matrix, axis=1, dtype=np.float64),
    ], axis=1)
    reduced = csum[:, end_idx] - csum[:, start_idx]
    return reduced.astype(np.float32)

neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps, matrix=event_matrix,
    starts=interval_starts, ends=interval_ends,
)
```

iii. The AI justifies this by saying the paper "explicitly assigns behavioral events to each 750 ms image presentation interval" and that this "keeps a fixed task-relevant time step across sessions."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to individual neurons. Experiments with no valid ROI events (`len(ds.events) == 0`) are skipped entirely. The AllenSDK's built-in quality control (cell_specimen_table filtering) is relied upon.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. The CONVERSION_NOTES.md states: "The AllenSDK already excludes invalid ROIs from events/cell_specimen_table, so no additional ROI-quality filter was added."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation intervals rather than to the ophys timestamps directly. The intervals are defined by successive stimulus onset times from the active `change_detection` block. Each trial's neural data consists of summed events within the stimulus intervals belonging to that trial (matched via `trials_id`).

ii.
```python
stim = make_change_detection_table(ds)
interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps, matrix=event_matrix,
    starts=interval_starts, ends=interval_ends,
)
...
trial_data.append(TrialData(
    neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
    ...
))
```

iii. The AI's CONVERSION_NOTES.md explains that successive stimulus onsets define bin boundaries and that bins are assigned to trials via `stimulus_presentations.trials_id`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes, temporal rebinning is applied. The data is rebinned from the native ophys frame rate (~11 Hz, ~90ms) to image-presentation intervals (~750ms). The `time_bin_size` in metadata is computed as the median interval duration in milliseconds.

ii.
```python
interval_durations = np.concatenate([s.interval_durations for s in session_data])
"time_bin_size": float(np.median(interval_durations) * 1000.0),
```

iii. CONVERSION_NOTES.md reports: "median interval duration: 0.75061 s" and "mean interval duration: 0.75067 s", matching the expected 250ms image + 500ms grey cadence.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations["image_name"]` — the image name assigned to each stimulus presentation interval.

ii.
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. The AI uses the stimulus presentations table directly, which naturally provides the image name per interval. This includes "omitted" as a category for omitted stimulus presentations.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping. The "omitted" category is sorted last (using a custom sort key). NaN image names are filled with "unknown".

ii.
```python
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. The global mapping ensures consistent codes across all sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is naturally aligned because it comes from the same stimulus_presentations table that defines the time bins. Each time bin corresponds to one image presentation, and the image_name for that presentation is used directly.

ii.
```python
trial_stim = stim[stim["trials_id"] == trial_id].copy()
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
```

iii. Since both neural and image identity are indexed by stimulus presentation intervals, alignment is inherent.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations["is_change"]` — a boolean column indicating whether each stimulus presentation is a change stimulus.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. The SDK's `is_change` column directly provides this information per stimulus interval.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is converted to integer (0/1). NaN values are filled with False. No additional processing or windowing is applied — the change indicator is 1 only for the single interval containing the change.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
...
change_idx = trial.image_change.astype(np.int64)
```

iii. Since each time bin is an image-presentation interval, the change flag naturally applies to just the one interval where the change occurred.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no_change, 1 = change). No thresholding is needed.

ii.
```python
["no_change", "change"],
```

iii. N/A — binary by nature.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — aligned via stimulus presentation intervals. The `is_change` flag applies to the same intervals used for neural binning.

ii. See 4-a code snippet.

iii. Inherent alignment through shared interval indexing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed["speed"]`.

ii.
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. Standard SDK interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged (mean) within each image-presentation interval using `interval_reduce_mean`. Then, all running speed values across all sessions are discretized into 5 bins using rank-based quintiles (`pd.qcut` on ranks).

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
...
def rank_quintiles(values):
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)

all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)
```

iii. The mean-within-interval approach aggregates the running speed to match the ~750ms time bins. Rank-based quintiles guarantee exactly equal bin counts even with ties.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 quintile bins using rank-based binning across all sessions globally.

ii.
```python
running_bins = rank_quintiles(all_running)
```

iii. Rank-based quintiles ensure balanced categories.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same image-presentation intervals used for neural data, ensuring alignment.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
...
running_raw=running_by_interval[stim_idx].astype(np.float32, copy=False),
```

iii. Both running and neural use the same interval boundaries.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_width"]`.

ii.
```python
pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
...
pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
```

iii. `pupil_width` is used as the diameter-like quantity. The CONVERSION_NOTES.md notes this is "the direct diameter-like quantity exposed by the SDK."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in pupil_width are linearly interpolated over time using `np.interp` (the `fill_nan_by_time` function). The interpolated values are then averaged within each image-presentation interval. Finally, all pupil values are discretized into 5 quintile bins using rank-based binning.

ii.
```python
def fill_nan_by_time(timestamps, values):
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    if finite.all(): return values
    values[~finite] = np.interp(
        timestamps[~finite], timestamps[finite], values[finite]
    ).astype(np.float32)
    return values

pupil_filled = fill_nan_by_time(timestamps=eye_t, values=...)
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
...
pupil_bins = rank_quintiles(all_pupil)
```

iii. The AI does NOT explicitly filter out blink frames using `likely_blink` before interpolation. Instead, it interpolates over all NaN values (which may include blinks, since the SDK sometimes sets blink frames to NaN). The AI's CONVERSION_NOTES.md claims "this is already blink-filtered by the AllenSDK (likely_blink rows are NaN)" but the code does not explicitly use the `likely_blink` column.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: rank-based quintiles across all sessions globally.

ii.
```python
pupil_bins = rank_quintiles(all_pupil)
```

iii. Rank-based quintiles ensure balanced categories.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed — averaged within the same image-presentation intervals.

ii.
```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
...
pupil_raw=pupil_by_interval[stim_idx].astype(np.float32, copy=False),
```

iii. Alignment via shared interval boundaries.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def infer_trial_outcome(trial_row):
    if bool(trial_row["hit"]): return "hit"
    if bool(trial_row["miss"]): return "miss"
    if bool(trial_row["false_alarm"]): return "false_alarm"
    if bool(trial_row["correct_reject"]): return "correct_reject"
    raise ValueError(f"Could not infer trial outcome for trial {trial_row.name}")
```

iii. The four outcomes are the canonical change-detection outcomes. Unlike the reference which falls back to 'other', the AI raises an error if none match.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The string outcome is mapped to an integer code via a fixed mapping. The code is repeated across all time bins within the trial.

ii.
```python
TRIAL_OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
...
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
```

iii. Same 4-category mapping as the reference.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- Experiments with no events, no eye tracking, or all-NaN pupil are skipped entirely.
- NaN values in pupil data are linearly interpolated over time before interval averaging.
- For running speed, the `interval_reduce_mean` function handles empty intervals by interpolating to interval centers.
- NaN image names are filled with "unknown".
- Trials with no matching stimulus presentations are skipped.
- Experiments with fewer than 2 valid trials are skipped.

ii.
```python
if len(ds.events) == 0: return None
if ds.eye_tracking.empty: return None
if not np.isfinite(pupil_width).any(): return None
...
pupil_filled = fill_nan_by_time(timestamps=eye_t, values=...)
...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
...
if len(trial_data) < 2: return None
```

iii. The AI's approach is conservative, skipping problematic experiments rather than trying to fix them. NaN interpolation for pupil is a reasonable approach to handle blinks/missing data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment via `cache.get_behavior_ophys_experiment()` is the most time-consuming step, as it reads large NWB files from disk. The AI processes 202 experiments sequentially.

ii. N/A

iii. I/O bound data loading dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that assigns stimulus presentations to trials iterates over each trial's stimulus presentations. The image-to-index mapping (`[image_to_idx[name] for name in trial.image_names]`) uses a list comprehension that could potentially be vectorized with a pandas categorical or numpy approach.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...
```

iii. The trial-level loop involves DataFrame filtering per trial which is somewhat inefficient but not a bottleneck compared to data loading.

## 9-c. What processing does the code repeat multiple times?

i. The code does not repeat any major processing steps. Each experiment is loaded and processed once. The `interval_reduce_mean` and `interval_reduce_sum_matrix` functions compute their cumulative sums once per experiment for all intervals simultaneously.

ii. N/A

iii. The design is single-pass per experiment.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes detailed summary statistics (`summarize_selected`, `session_info` metadata) and interval duration statistics that are stored in metadata but not used by the decoder. It also computes `omission_count` and `interval_count` for sanity check reporting. The `make_sample_dataset` function creates a separate sample subset which is additional processing.

ii.
```python
session_info.append({
    "ophys_experiment_id": session.experiment_id,
    "behavior_session_id": session.behavior_session_id,
    ...
})
```

iii. This extra metadata is useful for documentation but not consumed by the decoder.
