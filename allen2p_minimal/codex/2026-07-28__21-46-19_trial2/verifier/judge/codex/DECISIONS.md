# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the Allen experiment table from `VisualBehaviorOphysProjectCache`, then restricts to experiments that already have a local NWB file on disk, only keeps `active_behavior` experiments from project codes `VisualBehavior` or `VisualBehaviorMultiscope`, and further restricts to `VISp`/`VISl`. It then loads each kept `ophys_experiment_id` individually with `get_behavior_ophys_experiment()`. Trials are not loaded as whole-session trial windows; they are later reconstructed from `trials` plus `stimulus_presentations`.

ii. ```python
def select_experiments(
    cache: VisualBehaviorOphysProjectCache,
    cache_dir: Path,
) -> pd.DataFrame:
    experiment_table = cache.get_ophys_experiment_table().reset_index()
    local_ids = get_local_experiment_ids(cache_dir)
    selected = experiment_table[
        (experiment_table["ophys_experiment_id"].isin(local_ids))
        & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
        & (experiment_table["behavior_type"] == "active_behavior")
        & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
    ].copy()
```

```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    print(f"[{idx}/{len(selected)}] loading experiment {experiment_id}")
    session = process_experiment(
        cache=cache,
        experiment_id=int(experiment_id),
        max_trials_per_session=max_trials_per_session,
    )
```

iii. In `CONVERSION_NOTES.md`, the AI says the local cache is incomplete and read-only, so it deliberately defines the dataset as "all locally available active Visual Behavior ophys experiments" and treats that as the only reproducible choice in this environment.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values taken from the converted `SessionData` objects, then sorted globally.

ii. ```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.mouse_id])
```

iii. The AI did not give a separate justification beyond following Allen metadata conventions; `CONVERSION_NOTES.md` reports cohort counts in terms of mice/subjects and treats `mouse_id` as the subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each loaded `ophys_experiment_id` as one session in the output. It does not group multiple experiments that share an `ophys_session_id`.

ii. ```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    ...
    session = process_experiment(
        cache=cache,
        experiment_id=int(experiment_id),
        max_trials_per_session=max_trials_per_session,
    )
    if session is not None:
        session_data.append(session)
```

```python
return SessionData(
    experiment_id=int(experiment_id),
    behavior_session_id=int(ds.metadata["behavior_session_id"]),
    ...
)
```

iii. In the trajectory, the AI initially planned to work with experiment-level AllenSDK objects and kept that representation. In `CONVERSION_NOTES.md`, the final cohort is also counted at experiment/session level, confirming that each experiment was treated as a session.

## 1-d. How are the data split into trials?

i. The AI does not use trial `start_time` to `stop_time` windows. Instead, it first builds a table of active `change_detection` stimulus presentations, defines each stimulus interval as the time from one stimulus onset to the next, filters those intervals to kept trials via `trials_id`, and then groups intervals back into trial objects by `trials_id`.

ii. ```python
def make_change_detection_table(ds: Any) -> pd.DataFrame:
    stim = ds.stimulus_presentations.copy()
    block_mask = stim["stimulus_block_name"].astype(str).str.contains("change_detection")
    active_mask = stim["active"].fillna(False).astype(bool)
    stim = stim.loc[block_mask & active_mask].copy()
    ...
    interval_end[:-1] = start_times[1:]
    interval_end[-1] = start_times[-1] + median_dt
    stim["interval_end"] = interval_end
```

```python
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    if trial_stim.empty:
        continue
```

iii. `CONVERSION_NOTES.md` explicitly says the AI "did not use raw 2p frames as decoder bins" and instead matched what it called the paper's "image-interval analysis style," using one decoder bin per image-presentation interval and assigning intervals to trials with `stimulus_presentations.trials_id`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if `go` or `catch` is true and both `aborted` and `auto_rewarded` are false. After that, the AI also requires that the trial have at least one associated kept stimulus interval, and it drops any experiment with fewer than two kept trials. It does not explicitly require non-null `change_time`.

ii. ```python
trials = ds.trials.copy()
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
kept_trials = trials.loc[keep_trials].copy()
```

```python
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
...
if trial_stim.empty:
    continue
...
if len(trial_data) < 2:
    print(f"Skipping {experiment_id}: fewer than 2 kept trials with stimulus intervals")
    return None
```

iii. `CONVERSION_NOTES.md` says this was chosen to "match the task instruction to include Go/Catch and exclude Aborted/Auto-rewarded trials." The extra interval-presence and minimum-two-trials filters are implementation guards required by the chosen interval representation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from `BehaviorOphysExperiment.events`, specifically the per-ROI `events` traces.

ii. ```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
...
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says this follows the paper's use of discrete calcium events rather than dF/F traces and states that neural activity is taken from `BehaviorOphysExperiment.events`.

## 2-b. How is the `neural` data processed?

i. The AI converts continuous event traces into one value per stimulus interval by summing event magnitudes across all ophys timestamps that fall inside each image-presentation interval. It does not merge multiple planes within a behavioral session because it treats one experiment as one session.

ii. ```python
def interval_reduce_sum_matrix(
    timestamps: np.ndarray,
    matrix: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate(
        [
            np.zeros((matrix.shape[0], 1), dtype=np.float64),
            np.cumsum(matrix, axis=1, dtype=np.float64),
        ],
        axis=1,
    )
    reduced = csum[:, end_idx] - csum[:, start_idx]
    return reduced.astype(np.float32)
```

```python
neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps,
    matrix=event_matrix,
    starts=interval_starts,
    ends=interval_ends,
)
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
```

iii. `CONVERSION_NOTES.md` says the intent was to "match the paper's image-interval analysis style" and therefore sum AllenSDK event magnitudes within each 750 ms interval.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no explicit per-neuron filter beyond whatever AllenSDK has already placed in `ds.events`. However, it skips an entire experiment if `ds.events` is empty.

ii. ```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. `CONVERSION_NOTES.md` says the AllenSDK already excludes invalid ROIs from `events`/`cell_specimen_table`, so the AI did not add any extra ROI-quality filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural activity to successive stimulus-presentation intervals from the active `change_detection` block, not to a single trial event such as `start_time` or `change_time`. Trial matrices are built by selecting the interval bins whose `trials_id` matches each kept trial.

ii. ```python
"temporal_alignment_event": (
    "Successive image-presentation intervals defined by active "
    "change_detection stimulus onsets and assigned to trials via trials_id; "
    "modalities aggregated using ophys/running/eye timestamps within each interval."
),
```

```python
interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
...
trial_stim = stim[stim["trials_id"] == trial_id].copy()
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as using the paper's 750 ms image-presentation cadence as the shared alignment axis because image identity and change labels are "naturally defined on these intervals."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use one bin per image-presentation interval, with bin size set from the median interval duration, about 750 ms. This is a substantial temporal rebinning from raw ophys frames and from the native running/eye sampling rates.

ii. ```python
start_times = stim["start_time"].to_numpy(dtype=np.float64)
dt = np.diff(start_times)
median_dt = float(np.median(dt))
...
stim["interval_duration"] = stim["interval_end"] - stim["start_time"]
```

```python
"time_bin_size": float(np.median(interval_durations) * 1000.0),
"image_interval_duration_median_s": float(np.median(interval_durations)),
```

iii. `CONVERSION_NOTES.md` says the AI intentionally chose one decoder time bin per image-presentation interval and reports a median interval duration of `0.75061 s`, which it considers consistent with the paper/whitepaper cadence.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `Image identity` is derived from `stimulus_presentations.image_name` within the filtered active `change_detection` stimulus table, not from the trial-table `initial_image_name` / `change_image_name` columns.

ii. ```python
trial_data.append(
    TrialData(
        ...
        image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
        ...
    )
)
```

iii. `CONVERSION_NOTES.md` says omissions are an explicit part of the task and therefore image identity was taken directly from the stimulus table's per-interval image labels rather than reconstructed from trial metadata.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI keeps one image label per stimulus interval, fills missing names with `"unknown"`, collects all unique image names globally, sorts them with `"omitted"` placed last, and converts each trial's interval labels to integer category codes.

ii. ```python
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` justifies retaining `"omitted"` as its own category because omissions are part of the Visual Behavior task and present in the SDK stimulus table.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned at the same stimulus-interval resolution as the neural data. Each interval's `image_name` is stored in the same bin index as the summed interval neural activity.

ii. ```python
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

```python
session_output.append(
    np.vstack(
        [
            image_idx,
            change_idx,
            run_idx,
            pupil_idx,
            outcome_idx,
        ]
    ).astype(np.int64, copy=False)
)
```

iii. The AI's stated rationale in `CONVERSION_NOTES.md` is that image identity is "naturally defined" on the same 750 ms image-presentation intervals used for the neural representation.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `Image change` is derived from `stimulus_presentations.is_change` in the active `change_detection` stimulus table.

ii. ```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. `CONVERSION_NOTES.md` says image change was taken directly from `stimulus_presentations.is_change`, consistent with the AI's interval-based representation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI keeps `is_change` as a per-interval boolean signal, fills missing values with `False`, and later converts it to integer 0/1 when constructing the decoder outputs. There is no extra 750 ms post-change window computation because each interval is already a 750 ms bin.

ii. ```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
...
change_idx = trial.image_change.astype(np.int64)
```

iii. The interval-bin design in `CONVERSION_NOTES.md` is the implicit justification: once each bin is an image-presentation interval, a binary `is_change` column already defines whether that interval is a change interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is applied. The boolean `is_change` values are simply encoded as two categories: `0 = no_change`, `1 = change`.

ii. ```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    TRIAL_OUTCOMES,
],
```

```python
change_idx = trial.image_change.astype(np.int64)
```

iii. `CONVERSION_NOTES.md` states exactly this category mapping: `0 = no_change`, `1 = change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned one-for-one with the same stimulus-interval bins as the neural data and image identity.

ii. ```python
trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. The AI's general alignment rationale in `CONVERSION_NOTES.md` is that all outputs should live on the same image-interval axis used to aggregate neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `Running speed` is derived from `ds.running_speed`, specifically the `timestamps` and `speed` columns.

ii. ```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` identifies `dataset.running_speed["speed"]` as the source and treats it as the AllenSDK locomotion signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI averages running speed over each stimulus interval using `interval_reduce_mean`, concatenates all interval values across the full converted dataset, and then discretizes them with global rank-based quintiles.

ii. ```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t,
    values=running_v,
    starts=interval_starts,
    ends=interval_ends,
)
```

```python
all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)
```

iii. `CONVERSION_NOTES.md` says the AI wanted variables expressed on image intervals and wanted five balanced categories, so it used interval means followed by "rank-based global quintiles."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-frequency bins by ranking all values with `rank(method="first")` and then applying `pd.qcut(..., q=5)`. This is not percentile-edge digitization; it is rank-based quintiling with deterministic tie-breaking.

ii. ```python
def rank_quintiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as guaranteeing "five balanced categories even with ties."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by reducing the running trace within exactly the same stimulus-interval boundaries used to sum neural events; each interval gets one mean running-speed category.

ii. ```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t,
    values=running_v,
    starts=interval_starts,
    ends=interval_ends,
)
...
run_idx = running_bins[running_cursor : running_cursor + t]
```

iii. The alignment rationale is the same as for the other outputs in `CONVERSION_NOTES.md`: all modalities are aggregated into shared image-presentation intervals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `Pupil diameter` is derived from `ds.eye_tracking`, specifically the `timestamps` and `pupil_width` columns. The AI does not explicitly filter rows by `likely_blink`; instead it operates on the raw `pupil_width` series and fills NaNs.

ii. ```python
eye_df = ds.eye_tracking.copy()
eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
```

iii. `CONVERSION_NOTES.md` says `pupil_width` is the diameter-like variable exposed by the SDK and asserts that blink-filtered samples already appear as NaNs, so interpolation over NaNs is sufficient.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI linearly fills NaNs in `pupil_width` over eye-tracking time, averages the filled signal within each stimulus interval, concatenates all interval values across the dataset, and converts them to global rank-based quintile bins.

ii. ```python
def fill_nan_by_time(timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    ...
    values[~finite] = np.interp(
        timestamps[~finite],
        timestamps[finite],
        values[finite],
    ).astype(np.float32)
    return values
```

```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t,
    values=pupil_filled,
    starts=interval_starts,
    ends=interval_ends,
)
...
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
```

iii. `CONVERSION_NOTES.md` justifies this as using the SDK pupil signal, interpolating over blink/missing gaps, and then discretizing by global quintiles on the interval representation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded with the same rank-based global quintile procedure as running speed, yielding five integer bins.

ii. ```python
pupil_bins = rank_quintiles(all_pupil)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    TRIAL_OUTCOMES,
],
```

iii. `CONVERSION_NOTES.md` says the AI used rank-based quintiles so the bins would be balanced even when many values tie.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging it over the same stimulus-interval boundaries used for the neural event sums; each trial's pupil output shares the neural trial's bin count and bin boundaries.

ii. ```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t,
    values=pupil_filled,
    starts=interval_starts,
    ends=interval_ends,
)
...
pupil_idx = pupil_bins[pupil_cursor : pupil_cursor + t]
```

iii. As with running speed, the justification in `CONVERSION_NOTES.md` is the choice to put all modalities on the same image-presentation interval grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
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

iii. `CONVERSION_NOTES.md` lists exactly these four outcome categories as the trial outcome representation.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI infers one categorical outcome string per kept trial, maps outcomes to integer codes in the fixed `TRIAL_OUTCOMES` order, and repeats that code across all time bins in the trial.

ii. ```python
outcome = infer_trial_outcome(trial_row)
...
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
...
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says the per-trial outcome is repeated across all bins so it can live in the same `(5, T)` output array as the time-varying variables.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues explicitly: it skips experiments with no events, no eye-tracking table, or all-NaN pupil width; fills missing pupil values by linear interpolation in time; fills missing `image_name` with `"unknown"`; fills missing `is_change` with `False`; falls back to interval-center interpolation if an interval contains no running or pupil samples; and drops experiments with fewer than two kept trials containing stimulus intervals.

ii. ```python
if len(ds.events) == 0:
    ...
if ds.eye_tracking.empty:
    ...
if not np.isfinite(pupil_width).any():
    ...
```

```python
values[~finite] = np.interp(
    timestamps[~finite],
    timestamps[finite],
    values[finite],
).astype(np.float32)
```

```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

```python
if (~valid).any():
    centers = (starts + ends) / 2.0
    out[~valid] = np.interp(centers[~valid], timestamps, values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` frames these choices as practical cleanup for an incomplete local cache and for AllenSDK eye-tracking gaps, with the main goal of preserving interval-aligned trials rather than adding more manual curation.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps appear to be loading every selected Allen experiment from disk/cache and reducing the full event matrix, running trace, and pupil trace over every stimulus interval. The code materializes a cumulative sum over the entire neuron-by-time event matrix for each experiment, which is also a substantial cost.

ii. ```python
session = process_experiment(
    cache=cache,
    experiment_id=int(experiment_id),
    max_trials_per_session=max_trials_per_session,
)
```

```python
csum = np.concatenate(
    [
        np.zeros((matrix.shape[0], 1), dtype=np.float64),
        np.cumsum(matrix, axis=1, dtype=np.float64),
    ],
    axis=1,
)
reduced = csum[:, end_idx] - csum[:, start_idx]
```

iii. The AI did not explicitly discuss runtime hotspots in `CONVERSION_NOTES.md`; this is inferred from the implementation and from the fact that the pipeline loads every NWB-backed experiment and processes full-session arrays before trial assembly.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the per-trial loop that repeatedly filters `stim` with `stim["trials_id"] == trial_id`, the per-trial list comprehension that maps image names to indices, and the second pass over all trials during decoder-dataset assembly.

ii. ```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...
```

```python
for trial in session.trials:
    t = trial.neural.shape[1]
    image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
    ...
```

iii. The AI did not give a written efficiency justification here. The current implementation favors straightforward dataframe- and list-based assembly over pre-grouping or more fully vectorized indexing.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats trial-wise dataframe filtering (`stim[stim["trials_id"] == trial_id]`) for every kept trial, traverses all trials once to concatenate raw running/pupil values for quintiles and again to assemble integer-coded outputs, and runs decoder-format verification twice when a sample subset is requested.

ii. ```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
```

```python
all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
...
for session in session_data:
    ...
    for trial in session.trials:
```

```python
valid, errors, warnings_list = verify_data_format(data)
...
valid_sample, errors_sample, warnings_sample = verify_data_format(sample)
```

iii. There is no explicit justification in the notes. The repeated passes seem to come from staging the pipeline into "extract raw interval-level data" and then "globally discretize and assemble outputs."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps and computes several bookkeeping items that the downstream decoder does not need: `trial_id` per trial, trial/interval count fields in `SessionData`, selection/cohort summaries for notes, and `interval_duration_all`, which is appended to but never used. It also performs sample-subset generation and validation that are not part of the final full dataset consumed downstream.

ii. ```python
@dataclass
class TrialData:
    ...
    trial_id: int
```

```python
@dataclass
class SessionData:
    ...
    trial_count_before_filter: int
    trial_count_after_filter: int
    omission_count: int
    interval_count: int
```

```python
interval_duration_all = []
...
interval_duration_all.append(t)
```

iii. The notes justify the extra bookkeeping as documentation and sanity-check support rather than decoder necessity; for example, `CONVERSION_NOTES.md` reports cohort counts, omission fractions, and interval-duration summaries that depend on these auxiliary computations.
