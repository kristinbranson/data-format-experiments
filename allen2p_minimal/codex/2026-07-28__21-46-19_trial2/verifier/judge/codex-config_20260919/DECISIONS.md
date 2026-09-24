# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent opens an AllenSDK S3 cache, enumerates the experiment table, intersects it with NWB experiment IDs physically present in the local cache, and retains active `VisualBehavior` or `VisualBehaviorMultiscope` VISp/VISl experiments. It then loads every selected experiment individually with `get_behavior_ophys_experiment`. Thus “all” means all locally available experiments satisfying those filters, not every experiment listed by the SDK.

ii.
```python
experiment_table = cache.get_ophys_experiment_table().reset_index()
local_ids = get_local_experiment_ids(cache_dir)
selected = experiment_table[
    (experiment_table["ophys_experiment_id"].isin(local_ids))
    & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
    & (experiment_table["behavior_type"] == "active_behavior")
    & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
].copy()
...
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. The trajectory says the cache was a partial, read-only on-disk subset and attempts to materialize missing NWBs failed. The agent therefore constrained processing to locally present files. It also sought a paper-matched active VISp/VISl cohort, although its final filter was broader than its initially stated familiar multiscope plan.

## 1-b. How are the data split into subjects?

i. Subjects are unique SDK `mouse_id` strings among successfully processed experiments; they are sorted globally and each output session gets an integer subject index.

ii.
```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.mouse_id])
```

iii. The agent relied on the SDK mouse identifier as the animal identity. The trajectory reports checking metadata and paper mouse/session counts when selecting the cohort.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` is emitted as a separate decoder session. Simultaneously recorded imaging planes sharing an `ophys_session_id` are not combined; `behavior_session_id` is retained only as metadata.

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
    if session is not None:
        session_data.append(session)
...
"ophys_experiment_id": session.experiment_id,
"behavior_session_id": session.behavior_session_id,
```

iii. The trajectory describes a “200-session cohort” while iterating experiment files and frames the experiment-level cohort as the paper-matched neural cohort. It does not record a justification for treating a plane/experiment, rather than an `ophys_session_id`, as a session.

## 1-d. How are the data split into trials?

i. Trial membership comes from `ds.trials` and `stimulus_presentations.trials_id`. For each kept trial, the code gathers active change-detection stimulus intervals assigned that trial. A trial therefore becomes a variable-length sequence of image-presentation intervals rather than all native ophys frames from trial `start_time` through `stop_time`.

ii.
```python
valid_trial_ids = set(int(x) for x in kept_trials.index.to_numpy())
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
```

iii. The agent reasoned that the stimulus table already maps each flash/omission to `trials_id` and that the paper's 750 ms image-presentation interval was a clean shared time axis for neural and behavioral data.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials that are neither aborted nor auto-rewarded are kept. Trials without retained active change-detection intervals are dropped, and an experiment is dropped unless at least two trials remain. Entire experiments are also skipped for missing events or usable eye tracking.

ii.
```python
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

iii. The trajectory says the trial table encoded the requested inclusion mask cleanly. The minimum of two trials enforces the decoder contract; missing-modality skips prevent malformed outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the AllenSDK `ds.events["events"]` arrays for valid ROIs, not from dF/F traces.

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. The trajectory explicitly records the choice of AllenSDK events as the neural signal and describes it as matching the selected paper cohort, but gives no detailed comparison showing why events should replace the reference dF/F signal.

## 2-b. How is the `neural` data processed?

i. For each stimulus interval, event samples whose ophys timestamps fall in `[start, end)` are summed using cumulative sums. No cross-plane stacking occurs because every experiment is separate.

ii.
```python
start_idx = np.searchsorted(timestamps, starts, side="left")
end_idx = np.searchsorted(timestamps, ends, side="left")
csum = np.concatenate([np.zeros((matrix.shape[0], 1)),
                       np.cumsum(matrix, axis=1, dtype=np.float64)], axis=1)
reduced = csum[:, end_idx] - csum[:, start_idx]
```

iii. The agent chose 750 ms stimulus-interval bins as a shared axis and used sums because event traces represent activity/events accumulated in each interval.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code uses only ROIs exposed by `ds.events`; experiments with no events are skipped. It performs no further neuron-wise filtering or normalization.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. The “valid ROI events” message implies reliance on the SDK’s upstream ROI/event quality control. No additional neural QC rationale appears in the trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to successive active change-detection stimulus onsets, reduced over each onset-to-next-onset interval, and assigned to trials through `trials_id`. There is no fixed alignment to trial start or change time.

ii.
```python
interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
neural_by_interval = interval_reduce_sum_matrix(...)
...
neural=neural_by_interval[:, stim_idx]
```

iii. The trajectory calls these intervals the shared alignment axis and says the stimulus table’s trial mapping avoids reconstructing trial membership manually.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal bin is the median onset-to-onset stimulus interval, approximately 750 ms. Native ophys frames are explicitly rebinned and summed. Interval length may vary slightly, while metadata reports only the median as `time_bin_size`.

ii.
```python
stim["interval_duration"] = stim["interval_end"] - stim["start_time"]
...
"time_bin_size": float(np.median(interval_durations) * 1000.0),
```

iii. The agent checked that the paper’s 750 ms image-presentation interval aligned cleanly enough to serve as a common binning axis and validated the resulting shapes and decoder performance.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It comes directly from `image_name` in active change-detection `stimulus_presentations`, including omission labels and substituting `"unknown"` for missing values.

ii.
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
```

iii. The agent chose stimulus presentations because they directly describe each flash/omission and already carry trial IDs.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. All observed names are collected globally, sorted (with `omitted` last), and mapped to integer category indices consistently across experiments.

ii.
```python
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The implicit rationale is to produce deterministic, dataset-wide categorical labels required by the decoder. The trajectory emphasizes categorical output coherence but gives no separate explanation for omission/unknown categories.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. One image label is taken from the exact stimulus row defining each neural aggregation interval, so the label and neural column share the same `stim_idx` ordering.

ii.
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
neural=neural_by_interval[:, stim_idx]
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
```

iii. A shared stimulus-interval axis was the central alignment decision in the trajectory.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived directly from the `is_change` boolean in the active change-detection stimulus-presentation table.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
```

iii. The agent viewed the stimulus table as the canonical per-presentation source for image events and their trial assignment.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Missing flags become false, booleans are cast to integer 0/1 during assembly, and each change flag occupies its full stimulus interval.

ii.
```python
change_idx = trial.image_change.astype(np.int64)
```

iii. No additional processing was considered necessary because `is_change` already represents the event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated. The SDK boolean is encoded as 0 (`no_change`) or 1 (`change`), with missing values assigned 0.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```

iii. The raw variable is already categorical, so direct binary encoding satisfies the decoder requirement.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change flag and neural event sum come from the same stimulus-presentation interval and have identical column positions within a trial.

ii.
```python
neural=neural_by_interval[:, stim_idx]
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
```

iii. The agent justified all modalities through the shared stimulus-interval time axis.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the `speed` and `timestamps` columns of `ds.running_speed`.

ii.
```python
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. The agent traced the AllenSDK running API and used its timestamped standard speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running samples in each stimulus interval are averaged using cumulative sums. If an interval contains no samples, speed is linearly interpolated at its center. The resulting interval means are then globally rank-binned into quintiles.

ii.
```python
running_by_interval = interval_reduce_mean(...)
...
ranks = pd.Series(values).rank(method="first")
bins = pd.qcut(ranks, q=5, labels=False)
```

iii. The shared interval aggregation keeps behavior synchronized with neural bins; five equal-rank groups implement the requested equal-percentile categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. All retained interval values across the full converted dataset are ranked with ties broken by first occurrence, then `qcut` assigns exactly five approximately equal-count bins numbered 0–4.

ii.
```python
all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)
```

iii. Equal-rank quintiles were selected to satisfy “five equal percentile bins” and balance decoder classes.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running values are averaged over the same stimulus start/end boundaries used to sum neural events; slices are assigned to the same trial stimulus indices.

ii.
```python
running_by_interval = interval_reduce_mean(..., starts=interval_starts, ends=interval_ends)
...
running_raw=running_by_interval[stim_idx]
```

iii. The trajectory explicitly describes aggregating all modalities within common stimulus intervals using their native timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `pupil_width` and `timestamps` from `ds.eye_tracking`. The `likely_blink` flag is not used.

ii.
```python
eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
```

iii. The agent selected pupil width as the pupil-diameter proxy and checked for nonempty, finite eye data. It did not document why blink filtering was omitted.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaNs in the full pupil-width series are linearly interpolated in time; filled samples are averaged within stimulus intervals (or interpolated at the interval center if empty), then globally rank-binned into quintiles.

ii.
```python
values[~finite] = np.interp(timestamps[~finite], timestamps[finite], values[finite])
...
pupil_by_interval = interval_reduce_mean(...)
...
pupil_bins = rank_quintiles(all_pupil)
```

iii. Interpolation ensures a complete categorical stream, interval averaging creates the common temporal resolution, and quintiles provide balanced discrete targets.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. All interval-mean pupil values are globally ranked (first-occurrence tie breaking) and split with `qcut` into five approximately equal-count bins 0–4.

ii.
```python
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
```

iii. This directly implements five equal percentile bins and maintains a common category definition across experiments.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil samples are averaged over the same stimulus interval boundaries used for neural event sums, then indexed by the same per-trial stimulus rows.

ii.
```python
pupil_by_interval = interval_reduce_mean(..., starts=interval_starts, ends=interval_ends)
...
pupil_raw=pupil_by_interval[stim_idx]
```

iii. The common interval time axis was intended to align eye, running, stimulus, and ophys streams despite their distinct native timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns of `ds.trials`, checked in that order.

ii.
```python
if bool(trial_row["hit"]): return "hit"
if bool(trial_row["miss"]): return "miss"
if bool(trial_row["false_alarm"]): return "false_alarm"
if bool(trial_row["correct_reject"]): return "correct_reject"
```

iii. These are the SDK’s canonical outcomes for retained go/catch trials; the code raises rather than silently inventing a label if none is present.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A fixed global mapping converts the four names to 0–3, then repeats the trial’s code across every temporal bin in that trial.

ii.
```python
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
...
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
```

iii. Repetition makes the static trial label compatible with the common time-varying output matrix used by the decoder.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments are skipped if events, eye tracking, finite pupil data, eligible trials, eligible stimulus intervals, or two valid trials are absent. Missing pupil samples are interpolated; empty behavior intervals are center-interpolated; missing image names become `unknown`; missing change flags become false. The last stimulus interval receives a median-duration endpoint. Unlike the reference, exceptions during an experiment are not caught, blink frames are not removed, and missing running values are not explicitly repaired before cumulative averaging.

ii.
```python
if not np.isfinite(pupil_width).any(): return None
values[~finite] = np.interp(...)
out[~valid] = np.interp(centers[~valid], timestamps, values)
image_names=trial_stim["image_name"].fillna("unknown")
image_change=trial_stim["is_change"].fillna(False)
interval_end[-1] = start_times[-1] + median_dt
```

iii. The agent encountered missing local NWBs and an indexing bug, then narrowed to locally available data and tested a sample before the full run. Its code favors skipping unusable experiments and interpolation/default categories to keep arrays complete.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and parsing each NWB through `get_behavior_ophys_experiment` dominates. Full-session cumulative sums over every neuron/timepoint and the serial processing of roughly 200 experiments are the main compute costs.

ii.
```python
for ... experiment_id in enumerate(...):
    session = process_experiment(...)
...
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. The trajectory reports the full pass spending about 13 CPU-minutes in bulk NWB reads and identifies I/O as the expected bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Experiment loading is necessarily iterative, but the per-trial DataFrame filtering repeatedly scans `stim`; it could be replaced by a `groupby("trials_id")`. The image-name list comprehension could use a vectorized mapping. Sample selection and metadata collection loops could also be simplified, though they are minor. Interval reductions are already vectorized with searchsorted and cumulative sums.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
image_idx = np.array([image_to_idx[name] for name in trial.image_names])
```

iii. The trajectory does not discuss vectorization specifically; it focused optimization effort on reliable sample validation and letting the I/O-heavy full pass complete.

## 9-c. What processing does the code repeat multiple times?

i. `stim[stim["trials_id"] == trial_id]` repeatedly scans/copies the same stimulus table for each trial. The data are also traversed several times to concatenate behavior, collect image categories, build outputs, and compute summaries. A sample pickle is created by slicing the already-built full output, without reloading raw data.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
all_running = np.concatenate([...])
image_values = sorted({name for s in session_data for t in s.trials ...})
```

iii. No explicit justification appears. The multiple passes keep category construction and final assembly straightforward, and are likely small relative to NWB I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `interval_duration_all` is appended to for every trial but never read. Several counts and summaries are computed only for logging/metadata, and by default a separate sample dataset is copied and written even though the required downstream artifact is the full pickle. The rich session metadata is not needed by the decoder itself.

ii.
```python
interval_duration_all = []
...
interval_duration_all.append(t)
...
if args.sample_output is not None:
    sample = make_sample_dataset(...)
```

iii. The trajectory shows the sample artifact and summaries were used for sanity checking, validation, and reporting, so they served development diagnostics even though they are discarded by downstream model inputs. The unused list has no stated purpose.
