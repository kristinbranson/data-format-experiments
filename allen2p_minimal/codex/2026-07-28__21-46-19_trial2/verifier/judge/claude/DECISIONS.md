# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the Allen SDK's `VisualBehaviorOphysProjectCache.from_s3_cache()`. It gets the experiment table, then filters to experiments that are (1) locally available on disk as NWB files, (2) have `project_code` in `["VisualBehavior", "VisualBehaviorMultiscope"]`, (3) have `behavior_type == "active_behavior"`, and (4) have `targeted_structure` in `["VISp", "VISl"]`. Each matching experiment is loaded individually via `cache.get_behavior_ophys_experiment()`.

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

iii. The agent reasoned: "I've locked the conversion design... The main choices are: AllenSDK `events` as the neural signal, paper-matched familiar active `VisualBehaviorMultiscope` `VISp`/`VISl` experiments, and 750 ms stimulus-interval bins assigned to trials through `stimulus_presentations.trials_id`." The agent also noted the provided data directory was a fixed subset, so it constrained selection to locally available NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values extracted from the processed `SessionData` objects. They are sorted alphabetically.

ii.
```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent uses the SDK's `mouse_id` metadata field from each loaded experiment's metadata dictionary.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as a separate session. There is no grouping of experiments by `ophys_session_id`. This means multi-plane sessions from the Multiscope project each contribute multiple "sessions" to the output (one per imaging plane).

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
    if session is not None:
        session_data.append(session)
```

iii. The agent's design treats each experiment as a standalone session. From the trajectory: "The main choices are... paper-matched familiar active `VisualBehaviorMultiscope` `VISp`/`VISl` experiments."

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `ds.trials` table. Non-aborted, non-auto-rewarded trials that are either `go` or `catch` are kept. Within each trial, the time-varying data is segmented not by `start_time`/`stop_time`, but by the stimulus presentation intervals from `stimulus_presentations` that belong to that trial (matched via `trials_id`). Each stimulus presentation interval (~750ms) becomes one time bin.

ii.
```python
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
...
stim = make_change_detection_table(ds)
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
    ...
    trial_data.append(TrialData(
        neural=neural_by_interval[:, stim_idx],
        image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
        ...
    ))
```

iii. The agent stated: "The trial table already encodes the inclusion mask cleanly, and the stimulus table already maps every image flash or omission back to a `trials_id`. I'm now checking whether the paper's 750 ms 'image presentation interval' lines up cleanly enough to use as the shared binning axis."

## 1-e. How are trials filtered based on quality controls?

i. Aborted trials and auto-rewarded trials are excluded. Only `go` or `catch` trials are kept. Experiments with no valid ROI events, no eye tracking data, or entirely NaN pupil data are skipped entirely. Sessions with fewer than 2 valid trials are skipped. Trials with no matching stimulus intervals are also skipped.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
if ds.eye_tracking.empty:
    print(f"Skipping {experiment_id}: no eye tracking data")
    return None
...
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
...
if len(trial_data) < 2:
    print(f"Skipping {experiment_id}: fewer than 2 kept trials")
    return None
```

iii. The agent designed the filtering based on the instructions to exclude aborted and auto-rewarded trials and include only go/catch trials. The additional quality checks (empty events, eye tracking) prevent crashes on incomplete data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `events` trace (deconvolved spike events) from the Allen SDK, accessed via `ds.events["events"]`. This is NOT the `dff_traces` (dF/F calcium fluorescence).

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. The agent explicitly chose events over dF/F: "I've locked the conversion design... AllenSDK `events` as the neural signal." The metadata confirms: `"neural_signal": "AllenSDK events trace summed within each interval"`.

## 2-b. How is the `neural` data processed?

i. The events trace is summed within each stimulus presentation interval (~750ms) using a cumulative-sum-based approach (`interval_reduce_sum_matrix`). This produces one value per neuron per interval, rather than per ophys frame.

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

iii. The agent chose to sum events within intervals as this aligns with the paper's analysis approach of using stimulus presentation intervals as the temporal unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Experiments with no valid ROI events (`len(ds.events) == 0`) are skipped entirely. No per-neuron filtering is applied beyond what the SDK provides.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. The agent relies on the SDK's built-in quality control for cell segmentation and event detection.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation intervals rather than ophys timestamps directly. The `ophys_timestamps` are used to find which ophys frames fall within each interval (defined by `start_time` of each stimulus presentation), and events are summed within those intervals. Each trial's neural data consists of the intervals belonging to that trial.

ii.
```python
interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps, matrix=event_matrix,
    starts=interval_starts, ends=interval_ends,
)
...
trial_data.append(TrialData(
    neural=neural_by_interval[:, stim_idx],
    ...
))
```

iii. The agent stated: "the behavioral and stimulus targets are more consistent with the paper if I express them on each 750 ms image interval."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the stimulus presentation interval duration (~750ms), NOT the native ophys frame rate (~11 Hz / ~90ms). Temporal rebinning IS applied: ophys-rate events are summed within each ~750ms interval. The reported `time_bin_size` in metadata is the median interval duration in milliseconds.

ii.
```python
"time_bin_size": float(np.median(interval_durations) * 1000.0),
```

The median interval duration was ~750.6ms based on the conversion output.

iii. The agent deliberately chose 750ms interval binning based on the paper's stimulus presentation structure: "750 ms stimulus-interval bins assigned to trials through `stimulus_presentations.trials_id`."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column of the `stimulus_presentations` table (filtered to `change_detection` block, `active` presentations).

ii.
```python
stim = make_change_detection_table(ds)
...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. The stimulus_presentations table directly provides the image shown during each presentation interval.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global sorted mapping built from all unique image names across all sessions. The "omitted" image name is sorted to the end. Each interval gets the image_name from its stimulus presentation row.

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

iii. The sorted mapping ensures consistent codes across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because each time bin corresponds to one stimulus presentation interval, and the `image_name` column directly provides the image for that interval. Both neural data and image identity share the same interval-based indexing.

ii.
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
...
neural=neural_by_interval[:, stim_idx],
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. The interval-based design ensures perfect alignment between neural and stimulus data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. The `is_change` column is a pre-computed boolean flag in the stimulus table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is simply converted to integer (0/1). NaN values are filled with False.

ii.
```python
change_idx = trial.image_change.astype(np.int64)
```

iii. Minimal processing needed since the SDK provides the change flag directly.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: 0 = no_change, 1 = change. The `is_change` column from stimulus_presentations is used directly as a boolean, applying to both go and catch trials (unlike the reference which restricts change=1 to go trials only).

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
...
["no_change", "change"],
```

iii. The AI uses `is_change` directly without distinguishing go vs. catch trials for the change indicator.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — aligned via the shared stimulus presentation interval indexing.

ii. See 3-c.

iii. Same interval-based alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using the `speed` and `timestamps` columns.

ii.
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. Standard SDK interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each stimulus presentation interval using `interval_reduce_mean`, which computes a cumulative-sum-based mean over the running speed samples falling within each interval. The resulting per-interval values are then globally discretized into 5 quintile bins using `pd.qcut` on ranks.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
...
all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)

def rank_quintiles(values):
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)
```

iii. The agent uses interval-mean averaging consistent with its interval-based binning design, and rank-based quintiles for balanced discretization.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 quintile bins (q1-q5) using `pd.qcut` on rank-transformed values. This produces approximately equal counts in each bin.

ii.
```python
def rank_quintiles(values):
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)
```

iii. Rank-based quintiles ensure exactly equal bin sizes regardless of the value distribution.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same stimulus presentation intervals as the neural data, ensuring alignment.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
...
running_raw=running_by_interval[stim_idx],
```

iii. Both neural and running data use the same interval boundaries.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking["pupil_width"]`.

ii.
```python
pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
pupil_filled = fill_nan_by_time(timestamps=eye_t, values=eye_df["pupil_width"].to_numpy(dtype=np.float64))
```

iii. Same `pupil_width` field as the reference.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in pupil_width are filled by linear interpolation using `np.interp` (based on timestamps of finite values). No explicit blink removal via `likely_blink` is performed — instead, NaN values (which may include blinks) are interpolated through. The filled values are then averaged within stimulus intervals and discretized into 5 quintile bins.

ii.
```python
def fill_nan_by_time(timestamps, values):
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    if finite.all():
        return values
    values[~finite] = np.interp(
        timestamps[~finite], timestamps[finite], values[finite]
    ).astype(np.float32)
    return values

pupil_filled = fill_nan_by_time(timestamps=eye_t, values=eye_df["pupil_width"].to_numpy())
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
```

iii. The AI fills NaN values by interpolation without first removing blink frames, whereas the reference explicitly removes `likely_blink` frames before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 quintile bins using `pd.qcut` on rank-transformed values.

ii.
```python
pupil_bins = rank_quintiles(all_pupil)
```

iii. Same rank-based quintile approach.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same interval-based alignment as running speed and neural data.

ii.
```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
```

iii. Consistent interval-based approach.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

ii.
```python
def infer_trial_outcome(trial_row):
    if bool(trial_row["hit"]):
        return "hit"
    if bool(trial_row["miss"]):
        return "miss"
    if bool(trial_row["false_alarm"]):
        return "false_alarm"
    if bool(trial_row["correct_reject"]):
        return "correct_reject"
    raise ValueError(f"Could not infer trial outcome for trial {trial_row.name}")
```

iii. Same four outcome categories as the reference, with a strict check that raises an error if none match.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping. The code is constant across all time bins within a trial (static per-trial).

ii.
```python
TRIAL_OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
```

iii. Same approach as reference — fixed mapping repeated across all time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing events**: Experiments with no valid ROI events are skipped.
- **Missing eye tracking**: Experiments with empty eye tracking or all-NaN pupil width are skipped.
- **NaN pupil values**: Filled by linear interpolation via `fill_nan_by_time`.
- **Missing image names**: Filled with "unknown" via `fillna("unknown")`.
- **Empty trials**: Trials with no matching stimulus intervals are skipped.
- **Few trials**: Experiments with <2 valid trials are skipped.
- **Empty intervals in interval_reduce_mean**: If an interval contains no samples, the center timestamp is used with `np.interp` as fallback.

ii.
```python
if len(ds.events) == 0: return None
if ds.eye_tracking.empty: return None
if not np.isfinite(pupil_width).any(): return None
...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
...
if len(trial_data) < 2: return None
```

iii. The agent handles edge cases defensively to prevent crashes during batch processing.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads large NWB files from disk. The trajectory shows the full conversion took many minutes, with the agent noting "still consuming CPU" and "roughly 180 GB of NWB reads."

ii. N/A

iii. I/O-bound NWB file loading dominates the runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_experiment` iterates over each kept trial to extract stimulus intervals. The image name mapping loop (`[image_to_idx[name] for name in trial.image_names]`) could be vectorized with a pandas map or numpy lookup. The `interval_reduce_mean` and `interval_reduce_sum_matrix` functions are already vectorized.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...
```

iii. The per-trial filtering `stim[stim["trials_id"] == trial_id]` is O(n_stim * n_trials) and could be replaced by a groupby.

## 9-c. What processing does the code repeat multiple times?

i. The code processes running and pupil data per-experiment but computes global discretization bins from all concatenated values afterward, which requires iterating over all trial data twice (once for extraction, once for bin computation). The `fill_nan_by_time` function is called per-experiment, which is necessary but not repeated unnecessarily.

ii. N/A

iii. The two-pass approach (extract then discretize) is standard and not wastefully repeated.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `omission_count` and `interval_count` per session, and `interval_durations` per session, which are stored in metadata but not used in the decoder. The `trial_count_before_filter` is also tracked but only used for reporting. The `make_sample_dataset` function is run to create a separate sample pickle, which is additional processing.

ii.
```python
omission_count=int((stim["image_name"] == "omitted").sum()),
interval_count=int(len(stim)),
```

iii. These are metadata fields for documentation purposes.
