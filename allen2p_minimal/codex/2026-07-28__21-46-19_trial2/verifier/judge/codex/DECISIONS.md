# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI initializes `VisualBehaviorOphysProjectCache`, reads the SDK experiment table, restricts it to experiment IDs that already have local NWB files, and then further filters to active `VisualBehavior` or `VisualBehaviorMultiscope` experiments in `VISp` or `VISl`. It then loads each selected `ophys_experiment_id` individually with `get_behavior_ophys_experiment()`.

ii. 
```python
def get_cache(cache_dir: Path) -> VisualBehaviorOphysProjectCache:
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(cache_dir))

def get_local_experiment_ids(cache_dir: Path) -> set[int]:
    experiment_dir = cache_dir / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {
        int(path.stem.split("_")[-1])
        for path in experiment_dir.glob("behavior_ophys_experiment_*.nwb")
    }

selected = experiment_table[
    (experiment_table["ophys_experiment_id"].isin(local_ids))
    & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
    & (experiment_table["behavior_type"] == "active_behavior")
    & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
].copy()

for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
```

iii. In trajectory steps 71, 163, and 166, the agent said the local cache was only a partial, read-only subset of the Allen release, so it intentionally limited the cohort to experiments already present on disk and loaded them one-by-one as the only reproducible choice in that environment.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id`. After conversion, the subject list is the sorted set of `SessionData.mouse_id`, and each output session gets a `subject_idx` from that mapping.

ii.
```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.mouse_id])
```

iii. The trajectory repeatedly discusses cohort selection in terms of mice and reports cohort counts by mouse; once the experiment subset was chosen, the agent used the SDK's `mouse_id` as the subject identity.

## 1-c. How are the data split into sessions?

i. The AI effectively treats each `ophys_experiment_id` as a session. Although it stores `behavior_session_id`, it does not group multiple experiments by `ophys_session_id` or `behavior_session_id` before building the final dataset.

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
    if session is not None:
        session_data.append(session)

return SessionData(
    experiment_id=int(experiment_id),
    behavior_session_id=int(ds.metadata["behavior_session_id"]),
    ...
)
```

iii. In steps 71 and 166, the agent framed the task around converting all locally present experiments. Its rationale was about local availability, not reconstructing multi-experiment sessions, so the session unit remained the experiment object returned by `get_behavior_ophys_experiment()`.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`, but the AI does not slice raw ophys frames from `start_time` to `stop_time`. Instead, it keeps go/catch trials, pulls active `change_detection` stimulus presentations, links them back to trials through `stimulus_presentations.trials_id`, and represents each trial as the ordered sequence of stimulus intervals assigned to that trial.

ii.
```python
trials = ds.trials.copy()
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
kept_trials = trials.loc[keep_trials].copy()

stim = make_change_detection_table(ds)
valid_trial_ids = set(int(x) for x in kept_trials.index.to_numpy())
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()

for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...
    TrialData(
        neural=neural_by_interval[:, stim_idx],
        image_names=trial_stim["image_name"]...,
        ...
    )
```

iii. In steps 36, 48, 163, and 166, the agent explicitly said it did not want to use raw 2p frames as decoder bins. It justified trial segmentation through stimulus intervals because the paper's image-interval analysis style already mapped each flash or omission back to `trials_id`.

## 1-e. How are trials filtered based on quality controls?

i. A trial must be `go` or `catch`, not `aborted`, and not `auto_rewarded`. Trials with no associated `change_detection` stimulus intervals are dropped. Entire experiments are also skipped if they have no events, no eye tracking, all-NaN pupil width, no kept go/catch trials, no kept stimulus intervals, or fewer than two retained trials.

ii.
```python
if len(ds.events) == 0:
    return None
if ds.eye_tracking.empty:
    return None
if not np.isfinite(pupil_width).any():
    return None

keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
...
if trial_stim.empty:
    continue
...
if len(trial_data) < 2:
    return None
```

iii. In steps 163 and 166, the agent said this matched the instruction to keep go/catch trials while dropping aborted and auto-rewarded trials. The additional experiment-level skips were justified as necessary sanity checks for the locally available subset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from `ds.events["events"]`, not from `dff_traces`.

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. In steps 9, 36, 163, and 166, the agent said it wanted to follow the paper's use of discrete calcium events rather than raw dF/F traces, and it documented that choice explicitly in `CONVERSION_NOTES.md`.

## 2-b. How is the `neural` data processed?

i. The AI converts each neuron's event train into one value per stimulus interval by summing event magnitudes between the interval's start and end timestamps. No cross-experiment merging is done; each experiment stays separate.

ii.
```python
interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps,
    matrix=event_matrix,
    starts=interval_starts,
    ends=interval_ends,
)
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False)
```

iii. In steps 36, 48, 163, and 166, the agent justified this as matching the paper's "image-presentation interval" analysis style: one neural bin per roughly 750 ms flash/gray interval.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit per-neuron QC beyond what AllenSDK has already exposed in `ds.events`. The only extra neural-related filter is skipping an experiment if `ds.events` is empty.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. In step 163, the agent wrote that the AllenSDK already excludes invalid ROIs from the event table, so it did not add another ROI-quality filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The trial-level neural matrices are aligned to successive active `change_detection` stimulus intervals, not directly to trial start or to raw ophys frame indices. Trial membership comes from `stimulus_presentations.trials_id`.

ii.
```python
stim = ds.stimulus_presentations.copy()
block_mask = stim["stimulus_block_name"].astype(str).str.contains("change_detection")
active_mask = stim["active"].fillna(False).astype(bool)
stim = stim.loc[block_mask & active_mask].copy()
...
trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
neural=neural_by_interval[:, stim_idx]
```

iii. In steps 36, 48, 163, and 166, the agent said it was intentionally aligning everything to the task's 750 ms image-presentation intervals because image identity and change labels are naturally defined on those bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the median stimulus-interval duration, about 750 ms. Yes: the raw ophys event traces are rebinned from native frame times into one value per image-presentation interval.

ii.
```python
"time_bin_size": float(np.median(interval_durations) * 1000.0),
...
"image_interval_duration_median_s": float(np.median(interval_durations)),
```

iii. In steps 36, 48, 163, and 166, the agent explicitly justified moving from raw frame bins to the paper's 750 ms image intervals so the decoder would operate on a task-defined common time axis.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name` for active `change_detection` intervals.

ii.
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
```

iii. In steps 48, 163, and 166, the agent said the stimulus table already assigned every image flash or omission to a `trials_id`, so it used the table's per-interval image labels directly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Missing image labels are filled with `"unknown"`, the per-trial interval labels are collected, all unique image names across the dataset are globally sorted with `"omitted"` forced to the end, and each interval label is mapped to an integer category.

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

iii. In step 163, the agent justified retaining `"omitted"` as its own category because omissions are an explicit part of the Visual Behavior task and appear in the stimulus table.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned interval-by-interval using the exact same `stim_idx` slice that is used to extract the rebinned neural data for that trial.

ii.
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. In steps 36, 48, 163, and 166, the agent said all modalities should share the image-presentation interval axis, so the image labels and neural data use the same interval indexing.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` on the active `change_detection` table.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
```

iii. In steps 48, 163, and 166, the agent treated the stimulus table as the canonical per-interval task description and therefore used its direct change flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code fills missing `is_change` values with `False`, converts them to a boolean array for each trial, and later casts that array to integer category labels when building the final output tensor.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
...
change_idx = trial.image_change.astype(np.int64)
```

iii. The trajectory does not describe additional processing beyond using the stimulus-table change flag on the shared interval axis.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `False/0` becomes `"no_change"` and `True/1` becomes `"change"`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    TRIAL_OUTCOMES,
]
```

iii. This follows the decoder task directly; the trajectory does not give a separate justification beyond using the stimulus-table `is_change` flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same active `change_detection` stimulus intervals as the neural data, using the same trial-level interval selection.

ii.
```python
trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. In steps 36, 48, 163, and 166, the agent said all task outputs should live on the image-interval axis, so image change is aligned exactly as the neural intervals are.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed["speed"]` together with `ds.running_speed["timestamps"]`.

ii.
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. The trajectory treats the AllenSDK `running_speed` table as the standard locomotion source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each stimulus interval via `interval_reduce_mean`, concatenated across all trials and sessions, and then converted to five global discrete bins.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t,
    values=running_v,
    starts=interval_starts,
    ends=interval_ends,
)
...
all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)
```

iii. In step 163, the agent justified this as part of the shared image-interval representation: one running value per image interval, then global quintile binning for categorical decoding.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five equal-frequency bins by ranking all values with `rank(method="first")` and then applying `pd.qcut(..., q=5, labels=False)`.

ii.
```python
def rank_quintiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)
```

iii. In step 163, the agent said it wanted five balanced categories across the full dataset, and it documented the choice as rank-based global quintiles.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging the wheel-speed samples inside each stimulus interval and then slicing those interval-level values trial-by-trial on the same interval index set as the neural data.

ii.
```python
running_by_interval = interval_reduce_mean(..., starts=interval_starts, ends=interval_ends)
...
running_raw=running_by_interval[stim_idx].astype(np.float32, copy=False),
...
run_idx = running_bins[running_cursor : running_cursor + t]
```

iii. In steps 36, 48, 163, and 166, the agent justified this as part of the common 750 ms image-interval time axis shared with neural activity.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking["pupil_width"]` with `ds.eye_tracking["timestamps"]`.

ii.
```python
eye_df = ds.eye_tracking.copy()
eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
...
values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
```

iii. In step 163, the agent said `pupil_width` was the direct diameter-like quantity exposed by the SDK, so it used that rather than another eye metric.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code linearly interpolates over all NaN pupil-width samples within a session, averages the filled series inside each stimulus interval, concatenates the interval means across the full dataset, and discretizes the result into five global bins.

ii.
```python
def fill_nan_by_time(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    ...
    values[~finite] = np.interp(
        timestamps[~finite],
        timestamps[finite],
        values[finite],
    ).astype(np.float32)
    return values

pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
pupil_by_interval = interval_reduce_mean(..., starts=interval_starts, ends=interval_ends)
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
```

iii. In step 163, the agent justified this by saying the SDK exposes blink-filtered pupil as NaNs and that interpolating within the eye-tracking time base before interval averaging would give one value per image interval.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded with the same global rank-based `qcut` scheme used for running speed, yielding five equal-frequency bins.

ii.
```python
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    TRIAL_OUTCOMES,
]
```

iii. In step 163, the agent described both running and pupil as being discretized into five balanced global quintiles.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging the filled pupil signal within the same stimulus-interval boundaries used for the rebinned neural data, then slicing those interval values per trial.

ii.
```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t,
    values=pupil_filled,
    starts=interval_starts,
    ends=interval_ends,
)
...
pupil_raw=pupil_by_interval[stim_idx].astype(np.float32, copy=False),
```

iii. In steps 36, 48, 163, and 166, the agent said all modalities should be aggregated onto the common image-presentation interval axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
TRIAL_OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]

def infer_trial_outcome(trial_row: pd.Series) -> str:
    if bool(trial_row["hit"]):
        return "hit"
    if bool(trial_row["miss"]):
        return "miss"
    if bool(trial_row["false_alarm"]):
        return "false_alarm"
    if bool(trial_row["correct_reject"]):
        return "correct_reject"
```

iii. In steps 163 and 166, the agent described these four AllenSDK trial outcomes as the categories kept for the final decoder target.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps each inferred outcome string to a fixed integer code and repeats that code across every time bin in the trial.

ii.
```python
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
...
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
...
np.vstack([image_idx, change_idx, run_idx, pupil_idx, outcome_idx])
```

iii. In step 163, the agent justified this by saying the output arrays should all share shape `(d_output, T)`, so the static per-trial outcome was repeated across the trial's time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by selecting only locally present NWB files, skipping experiments with missing events or eye tracking, skipping all-NaN pupil sessions, interpolating NaN pupil values over time, interpolating interval values if no raw samples fall inside an interval, filling missing image labels with `"unknown"`, and filling missing `is_change` values with `False`.

ii.
```python
local_ids = get_local_experiment_ids(cache_dir)
...
if len(ds.events) == 0:
    return None
if ds.eye_tracking.empty:
    return None
if not np.isfinite(pupil_width).any():
    return None

values[~finite] = np.interp(...)
...
if (~valid).any():
    centers = (starts + ends) / 2.0
    out[~valid] = np.interp(centers[~valid], timestamps, values).astype(np.float32)

image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. In steps 71, 163, and 166, the agent explicitly justified the local-file restriction as an environment constraint. The trajectory also explains the NaN interpolation as a way to keep interval-level pupil and behavior outputs defined on the shared binning axis.

## 9-a. What are the most time-consuming steps of the code?

i. The code structure suggests the most time-consuming steps are loading each experiment through `get_behavior_ophys_experiment()`, materializing the full event matrix with `np.vstack(ds.events["events"].values)`, and reducing full-session neural/behavior streams into interval-level arrays.

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)

event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
neural_by_interval = interval_reduce_sum_matrix(...)
running_by_interval = interval_reduce_mean(...)
pupil_by_interval = interval_reduce_mean(...)
```

iii. The trajectory focuses more on scientific choices than profiling, but the agent repeatedly treated experiment loading as the main outer-loop operation and validated the final data by running the full converter and decoder over the whole selected cohort.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-trial loop in `process_experiment`, especially the repeated `stim[stim["trials_id"] == trial_id]` DataFrame filtering, and the per-trial output-construction loop in `build_decoder_dataset`.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...

for session in session_data:
    ...
    for trial in session.trials:
        image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
        ...
```

iii. The trajectory does not give an explicit optimization rationale here; it concentrates on matching the chosen scientific representation rather than vectorizing the remaining Python/DataFrame loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly filters the stimulus table by `trials_id` inside the per-trial loop, repeatedly converts image names to integer codes trial-by-trial, and computes both rank-based bins and separate quantile summaries for running and pupil.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...

image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
...
running_bins = rank_quintiles(all_running)
pupil_bins = rank_quintiles(all_pupil)
...
running_quantiles = np.quantile(all_running, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
pupil_quantiles = np.quantile(all_pupil, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
```

iii. There is no explicit trajectory justification for these repeated passes; the agent was optimizing for finishing a working full conversion and documenting its scientific choices.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several summary or bookkeeping quantities that are not used by the decoder itself, such as `selected_summary`, `session_info`, `trial_count_before_filter`, `trial_count_after_filter`, `omission_count`, `interval_count`, and metadata quantiles. It also appends `interval_duration_all` without using it later.

ii.
```python
trial_count_before_filter: int
trial_count_after_filter: int
omission_count: int
interval_count: int
...
interval_duration_all = []
...
interval_duration_all.append(t)
...
selected_summary = summarize_selected(selected)
...
"session_info": session_info,
"selected_experiment_summary": selected_summary,
"running_bin_quantiles": [float(x) for x in running_quantiles],
"pupil_bin_quantiles": [float(x) for x in pupil_quantiles],
```

iii. The trajectory does not justify these as decoder necessities; they appear to have been added for reporting, documentation, or sanity-check purposes rather than downstream analysis.
