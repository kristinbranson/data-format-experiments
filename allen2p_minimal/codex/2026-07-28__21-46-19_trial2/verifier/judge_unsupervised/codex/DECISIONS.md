# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads Allen Visual Behavior data through `VisualBehaviorOphysProjectCache`, starts from the Allen experiment table, intersects it with locally present NWB files, and then loads each selected `ophys_experiment_id` one by one with `cache.get_behavior_ophys_experiment(...)`. It additionally restricts the cohort to active-behavior `VisualBehavior` and `VisualBehaviorMultiscope` experiments in `VISp` or `VISl`.

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

def select_experiments(cache, cache_dir):
    experiment_table = cache.get_ophys_experiment_table().reset_index()
    local_ids = get_local_experiment_ids(cache_dir)
    selected = experiment_table[
        (experiment_table["ophys_experiment_id"].isin(local_ids))
        & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
        & (experiment_table["behavior_type"] == "active_behavior")
        & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
    ].copy()

for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
```

iii. In `CONVERSION_NOTES.md`, the agent says the cache was only partially present and read-only, so it intentionally used "the experiments that are already present locally on disk" as "the only reproducible choice in this environment." In the trajectory, it originally planned a more paper-matched familiar `VisualBehaviorMultiscope` subset, then broadened the cohort because of local-availability constraints.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. The converter collects unique `mouse_id` values from `SessionData`, sorts them, and stores a `subject_idx` for each top-level session entry.

ii. 
```python
return SessionData(
    ...
    mouse_id=str(ds.metadata["mouse_id"]),
    ...
)

subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.mouse_id])
```

iii. The notes justify the cohort at the experiment/session level, but subject handling itself is implicit: the Allen metadata already exposes `mouse_id`, so the agent simply used the released identifier as the mouse split.

## 1-c. How are the data split into sessions?

i. The agent treats each loaded `ophys_experiment_id` as one decoder "session". It does not merge multiple imaging planes/experiments that belong to the same Allen `ophys_session_id`; instead each experiment becomes its own `SessionData` and its own top-level entry in `data["neural"]`, `data["input"]`, and `data["output"]`.

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

for session in session_data:
    ...
    neural.append(session_neural)
    decoder_input.append(session_input)
    output.append(session_output)
```

iii. The explicit justification is mostly indirect. The whitepaper excerpt the agent read says a multi-plane recording session can have up to 8 experiments, but the implemented converter still uses `BehaviorOphysExperiment` as the unit of loading and storage. The notes frame this as a practical cohort choice rather than a theoretically preferred one.

## 1-d. How are the data split into trials?

i. Trials are split using the Allen `ds.trials` table after filtering. For each retained row in `kept_trials`, the converter collects all active `change_detection` stimulus presentations whose `trials_id` matches that Allen trial index, and stores them as one `TrialData` object.

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
    if trial_stim.empty:
        continue
    ...
    trial_data.append(TrialData(..., trial_id=int(trial_id)))
```

iii. In the trajectory the agent says "The trial table already encodes the inclusion mask cleanly, and the stimulus table already maps every image flash or omission back to a `trials_id`." That is the core justification for using `ds.trials` plus `stimulus_presentations.trials_id`.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is rule-based rather than metric-based: keep only `go` or `catch` trials, and drop any `aborted` or `auto_rewarded` trials. Trials with no retained stimulus intervals are also effectively dropped.

ii. 
```python
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)
kept_trials = trials.loc[keep_trials].copy()
if kept_trials.empty:
    print(f"Skipping {experiment_id}: no kept go/catch trials")
    return None

stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
...
if trial_stim.empty:
    continue
```

iii. The notes state this was chosen to match the explicit task instruction: "include Go/Catch and exclude Aborted/Auto-rewarded trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the AllenSDK `events` table and `ophys_timestamps`. The code stacks `ds.events["events"]` into a neuron-by-time matrix and uses `ds.ophys_timestamps` to assign those event values into decoder bins.

ii. 
```python
ophys_timestamps = ds.ophys_timestamps.astype(np.float64)
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
...
neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps,
    matrix=event_matrix,
    starts=interval_starts,
    ends=interval_ends,
)
```

iii. The notes explicitly justify this as following "the paper's use of discrete calcium events rather than dF/F traces."

## 2-b. How is the `neural` data processed?

i. The agent converts the per-cell event arrays into a matrix, defines one decoder time bin per active image-presentation interval, and sums each neuron's event magnitudes over all ophys timestamps that fall within each interval. Trial matrices are then slices of this interval-by-interval neural matrix.

ii. 
```python
def interval_reduce_sum_matrix(timestamps, matrix, starts, ends):
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate(
        [np.zeros((matrix.shape[0], 1), dtype=np.float64),
         np.cumsum(matrix, axis=1, dtype=np.float64)],
        axis=1,
    )
    reduced = csum[:, end_idx] - csum[:, start_idx]
    return reduced.astype(np.float32)

neural_by_interval = interval_reduce_sum_matrix(...)
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False)
```

iii. The notes say the agent intentionally matched a "paper's image-interval analysis style" and chose one bin per image-presentation interval because image identity/change are naturally defined there and the interval duration matches the 250 ms image + 500 ms gray cadence.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is almost no additional neural QC in the converter itself. The only explicit checks are to skip experiments with no `ds.events`, and otherwise rely on the AllenSDK's already-filtered released data. No extra per-neuron thresholding is applied.

ii. 
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
...
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` states: "The AllenSDK already excludes invalid ROIs from `events`/`cell_specimen_table`, so no additional ROI-quality filter was added."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are not kept on the native ophys frame axis. Instead, they are aligned to active `change_detection` stimulus-presentation intervals, with interval starts defined by successive stimulus onsets and trial membership defined by `trials_id`. Within each trial, the neural matrix columns correspond to those per-image intervals.

ii. 
```python
stim = ds.stimulus_presentations.copy()
block_mask = stim["stimulus_block_name"].astype(str).str.contains("change_detection")
active_mask = stim["active"].fillna(False).astype(bool)
stim = stim.loc[block_mask & active_mask].copy()
...
interval_end[:-1] = start_times[1:]
interval_end[-1] = start_times[-1] + median_dt
...
trial_stim = stim[stim["trials_id"] == trial_id].copy()
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
neural=neural_by_interval[:, stim_idx]
```

iii. The trajectory says the agent believed the "paper's 750 ms image presentation interval" should be used as the shared binning axis, and the notes repeat that this was chosen instead of raw 2p frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The effective time bin is one image-presentation interval, with median duration about 750.61 ms. Yes: the code rebins neural, running, and pupil signals from their native timestamps into these larger per-image intervals.

ii. 
```python
median_dt = float(np.median(dt))
interval_end[:-1] = start_times[1:]
interval_end[-1] = start_times[-1] + median_dt
stim["interval_duration"] = stim["interval_end"] - stim["start_time"]
...
"time_bin_size": float(np.median(interval_durations) * 1000.0),
"image_interval_duration_median_s": float(np.median(interval_durations)),
```

iii. The notes explicitly justify this with the flashed-image cadence from the whitepaper/paper and describe it as matching the task at the image-interval level.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations["image_name"]` after the table has been restricted to active `change_detection` presentations and mapped onto retained trials.

ii. 
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
...
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
```

iii. The notes justify this as the natural per-interval stimulus label and say omissions were kept because they are an explicit part of the Visual Behavior task.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code fills missing `image_name` with `"unknown"`, stores each per-trial interval label as strings, then creates a global categorical vocabulary across the dataset. Categories are sorted alphabetically with `"omitted"` forced to the end. Each trial's labels are then integer-encoded.

ii. 
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
...
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. The notes say omissions were retained as a real category rather than dropped, because the omission condition is present in the Allen stimulus table and is part of the task structure.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned one-to-one with the neural interval bins. The exact same `stim_idx` interval selection used for `trial.neural` is also used to define `trial.image_names`, and the integer-coded `image_idx` becomes the first row of the per-trial `output` matrix.

ii. 
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
...
neural=neural_by_interval[:, stim_idx]
image_names=trial_stim["image_name"]...
...
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

iii. The trajectory justifies the shared interval axis by saying the behavioral and stimulus targets are "more consistent with the paper" when expressed on each 750 ms image interval.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `stimulus_presentations["is_change"]` for the active `change_detection` presentations assigned to each trial.

ii. 
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
```

iii. The notes say image change was "taken directly from `stimulus_presentations.is_change`."

## 4-b. What processing is involved in computing `output` *Image change*?

i. Processing is minimal: missing values are filled with `False`, the result is cast to boolean, and then later converted to `int64` when constructing the output matrix.

ii. 
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
...
change_idx = trial.image_change.astype(np.int64)
```

iii. The notes present this as a direct translation from the Allen stimulus table rather than a derived signal.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous signal. The code directly encodes the boolean change flag into two categories: `0 = no_change`, `1 = change`.

ii. 
```python
change_idx = trial.image_change.astype(np.int64)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    TRIAL_OUTCOMES,
],
```

iii. The notes state the categories explicitly: `0 = no_change`, `1 = change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same per-image interval axis as the neural data. Each column of `change_idx` corresponds to the same `stim_idx` interval column used in `trial.neural`.

ii. 
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
...
neural=neural_by_interval[:, stim_idx]
image_change=trial_stim["is_change"]...
...
change_idx = trial.image_change.astype(np.int64)
```

iii. The shared-bin justification is the same as for image identity: the agent explicitly chose a single image-interval axis for all modalities.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed["speed"]` and its associated `ds.running_speed["timestamps"]`.

ii. 
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. The notes explicitly identify the source as `dataset.running_speed["speed"]`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code averages running speed over each active image interval using `interval_reduce_mean`, then later discretizes the concatenated per-interval values across the full dataset.

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

iii. The notes justify interval averaging as part of the common image-interval representation, and global quintiles as a way to guarantee five balanced categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The agent does a global five-bin equal-frequency discretization. It ranks all interval-level running values with `rank(method="first")`, applies `pd.qcut(..., q=5)`, and names the bins `q1` through `q5`.

ii. 
```python
def rank_quintiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)

RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
run_idx = running_bins[running_cursor : running_cursor + t]
```

iii. The notes say both running and pupil were discretized into "five equal-frequency bins across the full converted dataset by rank-based global quintiles" to guarantee balanced categories even with ties.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first reduced to the same stimulus intervals as the neural data, then each trial takes the same `stim_idx` slice. The resulting quintile labels are inserted into the same per-trial `output` matrix columns as the neural bins.

ii. 
```python
running_by_interval = interval_reduce_mean(..., starts=interval_starts, ends=interval_ends)
...
running_raw=running_by_interval[stim_idx].astype(np.float32, copy=False)
...
run_idx = running_bins[running_cursor : running_cursor + t]
```

iii. The agent's stated design principle was one shared 750 ms interval axis for neural and behavioral signals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from `ds.eye_tracking["pupil_width"]` and `ds.eye_tracking["timestamps"]`, not from a dedicated diameter field.

ii. 
```python
pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
...
eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
```

iii. The notes justify this by saying `pupil_width` is "the direct diameter-like quantity exposed by the SDK."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code skips sessions whose pupil width is entirely NaN, linearly interpolates NaNs in the remaining sessions with `fill_nan_by_time`, averages the interpolated signal over each image interval, and then discretizes the pooled interval values into global quintiles.

ii. 
```python
if not np.isfinite(pupil_width).any():
    print(f"Skipping {experiment_id}: pupil width is entirely NaN")
    return None

def fill_nan_by_time(timestamps, values):
    ...
    values[~finite] = np.interp(
        timestamps[~finite],
        timestamps[finite],
        values[finite],
    ).astype(np.float32)
    return values

pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t,
    values=pupil_filled,
    starts=interval_starts,
    ends=interval_ends,
)
```

iii. The notes say the eye-tracking table is already blink-filtered by AllenSDK, so NaNs were treated as blink/missing segments and linearly interpolated before interval averaging.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly like running speed: a global rank-based quintile binning into `q1`-`q5`.

ii. 
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
...
pupil_idx = pupil_bins[pupil_cursor : pupil_cursor + t]
```

iii. The notes give the same justification as for running speed: balanced equal-frequency categories across the full converted dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is interpolated and then reduced onto the same active image intervals as the neural data. Trial slicing uses the same `stim_idx`, so each pupil bin label lines up column-for-column with each neural interval.

ii. 
```python
pupil_by_interval = interval_reduce_mean(..., starts=interval_starts, ends=interval_ends)
...
pupil_raw=pupil_by_interval[stim_idx].astype(np.float32, copy=False)
...
pupil_idx = pupil_bins[pupil_cursor : pupil_cursor + t]
```

iii. The justification is the same common image-interval axis described in the notes and trajectory.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns in the Allen `trials` table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
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

iii. The notes list those four categories as the retained trial outcomes after filtering to go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent infers one categorical label per retained trial, then repeats that same category across all time bins in the trial so it can live in the same `(5, T)` output array as the time-varying variables.

ii. 
```python
outcome = infer_trial_outcome(trial_row)
...
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
...
"output_values": [
    ...,
    TRIAL_OUTCOMES,
]
```

iii. `CONVERSION_NOTES.md` explicitly says the per-trial outcome is repeated across all bins in that trial so it fits the unified output array shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are handled with a mix of skipping, interpolation, and defaults. Entire sessions are skipped if they have no ROI events, no eye-tracking table, all-NaN pupil width, or fewer than two usable trials. Within retained sessions, pupil NaNs are linearly interpolated, empty running/pupil intervals are filled by interpolation at the interval center, missing `image_name` becomes `"unknown"`, and missing `is_change` becomes `False`.

ii. 
```python
if len(ds.events) == 0: ...
if ds.eye_tracking.empty: ...
if not np.isfinite(pupil_width).any(): ...
...
values[~finite] = np.interp(...)
...
if (~valid).any():
    centers = (starts + ends) / 2.0
    out[~valid] = np.interp(centers[~valid], timestamps, values).astype(np.float32)
...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
```

iii. The notes justify the session skips as necessary quality gates for required outputs, and justify pupil interpolation by saying the SDK has already marked blink periods as NaN. There is no explicit deeper justification for the `"unknown"` and interval-center interpolations beyond making the conversion robust.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each experiment through AllenSDK/NWB I/O, stacking the event arrays, reducing the full neuron-by-time event matrix across all stimulus intervals, and looping through every selected experiment in the full cohort.

ii. 
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(...)

ds = cache.get_behavior_ophys_experiment(int(experiment_id))
...
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
neural_by_interval = interval_reduce_sum_matrix(...)
```

iii. The trajectory comments on runtime directly: the agent noted the "full conversion has been running for about 13 CPU-minutes" and treated the loader/I/O path as the expected bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are still Python-level and could be reduced: iterating experiment-by-experiment, filtering `stim` separately for every trial with `stim[stim["trials_id"] == trial_id]`, per-trial list-comprehension encoding of `image_idx`, and repeated per-trial appends into nested Python lists.

ii. 
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    ...

for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    ...

image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. The agent did not explicitly document vectorization opportunities; this is an evaluation of the implemented code shape.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats the same interval-reduction pattern for multiple modalities, repeatedly filters the same stimulus table by `trials_id`, and performs several repeated `astype(...)` conversions and list-to-array constructions at both session and trial levels.

ii. 
```python
running_by_interval = interval_reduce_mean(...)
pupil_by_interval = interval_reduce_mean(...)
...
trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False)
running_raw=running_by_interval[stim_idx].astype(np.float32, copy=False)
pupil_raw=pupil_by_interval[stim_idx].astype(np.float32, copy=False)
```

iii. There is no explicit justification in the notes; the agent prioritized clarity and correctness checks over aggressive refactoring.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is `interval_duration_all.append(t)`, which is never used later. The code also stores some fields that are not consumed by the decoder (`trial_id`, several `SessionData` bookkeeping counts, quantile metadata, and the generated sample subset during a full run), though some of that is useful for notes/debugging rather than downstream analysis.

ii. 
```python
interval_duration_all = []
...
interval_duration_all.append(t)

@dataclass
class TrialData:
    ...
    trial_id: int

@dataclass
class SessionData:
    ...
    trial_count_before_filter: int
    trial_count_after_filter: int
    omission_count: int
    interval_count: int
```

iii. The notes emphasize validation and documentation, so some extra bookkeeping appears to have been kept for sanity checks rather than for the decoder itself.
