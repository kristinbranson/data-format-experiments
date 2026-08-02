# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the Allen SDK's `VisualBehaviorOphysProjectCache`. Rather than loading all experiments from the cache manifest, it first discovers which NWB files are physically present on disk, then filters the experiment table to only those locally available experiments. It further filters by `project_code` (both `VisualBehavior` and `VisualBehaviorMultiscope`), `behavior_type == "active_behavior"`, and `targeted_structure` in `{VISp, VISl}`. Each experiment is loaded individually via `cache.get_behavior_ophys_experiment()`.

ii.
```python
def get_local_experiment_ids(cache_dir: Path) -> set[int]:
    experiment_dir = cache_dir / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {
        int(path.stem.split("_")[-1])
        for path in experiment_dir.glob("behavior_ophys_experiment_*.nwb")
    }

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

iii. The AI documented in CONVERSION_NOTES.md that the local cache only contains a subset of experiments, so it restricts to locally available files. It includes `VisualBehaviorMultiscope` alongside `VisualBehavior` and filters to VISp/VISl and active behavior to match the paper's focus.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the selected experiments, sorted alphabetically.

ii.
```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` is the SDK's unique identifier for each animal. The AI found 38 unique mice in the locally available data.

## 1-c. How are the data split into sessions?

i. Each individual experiment (single imaging plane) is treated as a separate session. Multi-plane sessions are NOT grouped by `ophys_session_id`. Each `ophys_experiment_id` becomes its own session in the output.

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
    if session is not None:
        session_data.append(session)
```

iii. The AI processes each experiment independently. In the reference solution, experiments sharing an `ophys_session_id` are grouped together into a single session with neurons pooled across planes. The AI's approach results in 199 sessions (one per experiment) rather than grouping by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials are defined using `ds.trials` from the Allen SDK, filtered to go/catch trials that are not aborted and not auto-rewarded. Within each trial, the time axis is defined by stimulus presentation intervals from `stimulus_presentations` (filtered to the active `change_detection` block), not by raw ophys frames from `start_time` to `stop_time`. Each trial's time bins correspond to consecutive image-presentation intervals assigned to that trial via `trials_id`.

ii.
```python
trials = ds.trials.copy()
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
kept_trials = trials.loc[keep_trials].copy()

stim = make_change_detection_table(ds)
# ...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    if trial_stim.empty:
        continue
    stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
    # neural, running, pupil are all indexed by stim_idx
```

iii. The AI used stimulus presentation intervals as time bins rather than raw ophys frames, arguing this matches the paper's image-interval analysis style. Each trial contains ~10-17 intervals (one per image flash).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted trials, (2) excluding auto-rewarded trials, (3) requiring `go == True` or `catch == True`, (4) requiring matching stimulus presentation intervals exist in the `change_detection` block, (5) sessions with fewer than 2 valid trials are skipped. Additionally, entire experiments are skipped if they have no valid ROI events, no eye tracking data, or all-NaN pupil width.

ii.
```python
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
# ...
if len(trial_data) < 2:
    print(f"Skipping {experiment_id}: fewer than 2 kept trials")
    return None
# ...
if len(ds.events) == 0:
    return None
if ds.eye_tracking.empty:
    return None
```

iii. The AI's trial filtering matches the instructions (exclude aborted and auto-rewarded, keep go and catch). It adds additional session-level quality checks for missing eye tracking and empty neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `ds.events` (deconvolved calcium events), NOT from `ds.dff_traces` (dF/F fluorescence traces).

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. The AI chose `events` because the paper describes using "discrete calcium events" rather than raw dF/F traces. The CONVERSION_NOTES.md states: "Neural activity is taken from BehaviorOphysExperiment.events. This follows the paper's use of discrete calcium events rather than dF/F traces."

## 2-b. How is the `neural` data processed?

i. Neural event amplitudes are summed within each image-presentation interval using a cumulative-sum-based vectorized approach. Each time bin's neural value is the sum of all event magnitudes for each neuron within that interval's time window.

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

iii. The AI sums events within intervals, matching the paper's analysis style. The reference solution simply slices dF/F traces at native ophys frame resolution without any temporal aggregation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. The AI relies on the AllenSDK's built-in filtering in `events`/`cell_specimen_table`. Entire experiments are skipped if `ds.events` is empty.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. The CONVERSION_NOTES.md states: "The AllenSDK already excludes invalid ROIs from events/cell_specimen_table, so no additional ROI-quality filter was added."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation intervals. For each trial, the stimulus presentations assigned to that trial (via `trials_id`) define the time bins. Neural events are summed within each interval. The first interval of a trial corresponds to the first stimulus flash in that trial.

ii.
```python
stim = make_change_detection_table(ds)
# ...
trial_stim = stim[stim["trials_id"] == trial_id].copy()
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
neural=neural_by_interval[:, stim_idx].astype(np.float32)
```

iii. The AI aligned to stimulus intervals rather than ophys timestamps directly. The instructions say "temporally align based on ophys timestamp," which the reference interprets as using the ophys timestamp as the time axis at native frame rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~750ms per time bin (one image-presentation interval: 250ms image + 500ms grey). This is a significant rebinning from the native ophys frame rate (~11 Hz, ~90ms per frame). The time bin size is computed as the median interval duration across all intervals.

ii.
```python
"time_bin_size": float(np.median(interval_durations) * 1000.0),
```

iii. The AI chose to rebin to image-presentation intervals, arguing this matches the paper's analysis. The reference keeps the native ophys frame rate (~90ms bins) without rebinning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in `stimulus_presentations`, filtered to the active `change_detection` block.

ii.
```python
stim = make_change_detection_table(ds)
# ...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. The AI uses `image_name` from `stimulus_presentations` directly, which gives the image shown during each stimulus interval. This contrasts with the reference which uses `initial_image_name` and `change_image_name` from the trials table with a change-time switch.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping. The sorted order places `omitted` last (using a key that puts it after all `im*` names). The mapping includes 17 categories (16 natural images + `omitted`).

ii.
```python
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
# ...
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. The AI includes `omitted` as a category. The reference does not include `omitted` since it uses the trials table's `initial_image_name`/`change_image_name` fields which don't contain `omitted`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is naturally aligned because each time bin corresponds to one stimulus presentation interval, and the image name is the image shown during that interval. Both neural and image identity share the same `stim_idx` indexing.

ii.
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
# Both use same stim_idx:
neural=neural_by_interval[:, stim_idx]
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
```

iii. Since the time axis is defined by stimulus presentations, image identity is inherently aligned.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `stimulus_presentations`.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. The AI uses the SDK's pre-computed `is_change` flag rather than computing it from `change_time` and `go` columns.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is converted to integer (0/1). NaN values are filled with False. No additional windowing is applied -- it is 1 only at the single interval where the change occurs.

ii.
```python
change_idx = trial.image_change.astype(np.int64)
```

iii. The reference applies a 750ms window (one flash + grey period) for image change. The AI uses a single-interval indicator, which is approximately equivalent since each interval is ~750ms.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary: 0 = no_change, 1 = change. No thresholding is needed.

ii.
```python
["no_change", "change"]
```

iii. Both AI and reference use binary categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- aligned via stimulus presentation intervals using `stim_idx`.

ii. See 3-c.

iii. Same alignment mechanism as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `speed` column.

ii.
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. Same source as the reference.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is aggregated per image-presentation interval by computing the mean of all running speed samples within each interval, using a cumulative-sum-based approach. It is then discretized into 5 equal-frequency bins using rank-based quintiles (`pd.qcut` on ranks).

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
# ...
def rank_quintiles(values):
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)

running_bins = rank_quintiles(all_running)
```

iii. The reference interpolates to ophys timestamps and uses `np.percentile`/`np.digitize`. The AI averages within intervals and uses rank-based quintiles.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (labeled q1-q5) using rank-based quintiles computed globally across all sessions. This guarantees exactly equal bin counts.

ii.
```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
running_bins = rank_quintiles(all_running)
```

iii. The reference uses percentile-based edges with `np.digitize`, which approximately achieves equal counts. The AI's rank-based approach guarantees exactly equal counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same image-presentation intervals used for neural data, then indexed by the same `stim_idx`.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
# ...
running_raw=running_by_interval[stim_idx]
```

iii. Aligned by sharing the same interval-based time axis.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column.

ii.
```python
eye_df = ds.eye_tracking.copy()
pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
```

iii. Same source variable (`pupil_width`) as the reference.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in `pupil_width` are interpolated using `np.interp` over the eye tracking timestamps. Then pupil is averaged within each image-presentation interval using `interval_reduce_mean`. Finally, it is discretized into 5 bins using rank-based quintiles. Sessions with all-NaN pupil are skipped entirely.

ii.
```python
def fill_nan_by_time(timestamps, values):
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Series has no finite values")
    if finite.all():
        return values
    values[~finite] = np.interp(
        timestamps[~finite], timestamps[finite], values[finite],
    ).astype(np.float32)
    return values

pupil_filled = fill_nan_by_time(timestamps=eye_t, values=eye_df["pupil_width"]...)
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
```

iii. The reference explicitly removes blink frames (using `likely_blink`) before interpolation. The AI interpolates all NaN values, which includes blink frames (since the SDK sets blink frames to NaN). The approaches are functionally similar but differ in mechanism -- the reference explicitly filters by `likely_blink`, while the AI treats all NaN values uniformly.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed -- rank-based quintiles producing 5 bins (q1-q5) computed globally.

ii.
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
pupil_bins = rank_quintiles(all_pupil)
```

iii. Same approach as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed -- averaged within image-presentation intervals, indexed by `stim_idx`.

ii.
```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
pupil_raw=pupil_by_interval[stim_idx]
```

iii. Same alignment mechanism as all other signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

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

iii. Same source as the reference. The AI raises an error if none match, while the reference falls back to `'other'`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping. The integer code is repeated across all time bins within the trial.

ii.
```python
TRIAL_OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
```

iii. Same approach as the reference -- map to integer codes, repeat across time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Entire experiments with no eye tracking data are skipped.
- **All-NaN pupil**: Experiments with entirely NaN pupil width are skipped.
- **NaN values in pupil/running**: NaN pupil values are interpolated using `np.interp`. For running speed, if no samples fall within an interval, the center of that interval is interpolated from the running speed time series.
- **Empty events**: Experiments with no valid ROI events are skipped.
- **No valid trials**: Experiments with fewer than 2 valid trials after filtering are skipped.
- **Missing stimulus intervals**: Trials with no matching stimulus intervals are skipped.

ii.
```python
if len(ds.events) == 0:
    return None
if ds.eye_tracking.empty:
    return None
if not np.isfinite(pupil_width).any():
    return None
# ...
if len(trial_data) < 2:
    return None
```

iii. The AI's approach is more cautious than the reference, skipping entire experiments when key data streams are missing. The reference uses try/except to skip sessions that throw exceptions.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads large NWB files from disk. Additionally, building the stimulus presentation table and computing interval-based aggregations add computation.

ii. N/A

iii. Data loading is I/O bound and dominates runtime, same as in the reference.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_experiment` iterates over each kept trial to extract stimulus indices. However, the neural and behavioral data aggregation is already vectorized using cumulative sum approaches. The main loop body is lightweight (indexing into pre-computed arrays).

ii. N/A

iii. The AI's use of `interval_reduce_sum_matrix` and `interval_reduce_mean` with cumulative sums is already well-vectorized for the aggregation step. The per-trial loop is not a bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. `make_change_detection_table` is called once per experiment. The global discretization (rank_quintiles) is computed once across all data after all experiments are loaded. No significant processing is repeated.

ii. N/A

iii. The two-pass design (load all experiments, then discretize globally) avoids repeating work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `interval_durations`, `omission_count`, `interval_count`, `trial_count_before_filter`, and `trial_count_after_filter` as metadata/summary statistics in `SessionData`. These are used for documentation but not for downstream decoder analysis. The `omitted` image category is included in the output but may not be useful for decoding image identity in a change-detection context.

ii.
```python
return SessionData(
    ...
    interval_durations=stim["interval_duration"].to_numpy(dtype=np.float64),
    trial_count_before_filter=int(len(trials)),
    trial_count_after_filter=int(len(trial_data)),
    omission_count=int((stim["image_name"] == "omitted").sum()),
    interval_count=int(len(stim)),
)
```

iii. These statistics are useful for validation and documentation but are not directly used in the decoder pipeline.
