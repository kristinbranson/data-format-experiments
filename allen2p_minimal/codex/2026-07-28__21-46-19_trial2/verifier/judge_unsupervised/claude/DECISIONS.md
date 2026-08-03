# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir)` to access the Allen SDK cache. It scans the local disk for available NWB files and filters the experiment table to only process experiments with files present locally. Each experiment is loaded via `cache.get_behavior_ophys_experiment(experiment_id)`, which returns a `BehaviorOphysExperiment` object containing neural data (`events`), trials, stimulus presentations, running speed, and eye tracking.

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

# In process_experiment:
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. The agent noted that the manifest lists far more experiments than are actually present on disk and the data directory is read-only, so only locally available NWB files are processed. This was described as "the only reproducible choice in this environment."

## 1-b. How are the data split into subjects (mice)?

i. Subjects (mice) are identified by `mouse_id` from the experiment metadata. Each session's `mouse_id` is extracted from `ds.metadata["mouse_id"]`. A sorted list of unique subjects is built, and `subject_idx` maps each session to its subject index.

ii.
```python
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[session.mouse_id])
```

iii. The agent relied on the SDK's metadata to identify mice, which is the standard approach.

## 1-c. How are the data split into sessions?

i. Each ophys experiment ID corresponds to one session (one imaging plane from one recording). The experiment table is filtered for local availability, `active_behavior`, `VISp`/`VISl` targeted structures, and `VisualBehavior`/`VisualBehaviorMultiscope` project codes. Each passing experiment is loaded individually and becomes one session in the output.

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

iii. The agent documented that it uses all locally available active Visual Behavior experiments in VISp/VISl rather than trying to replicate the exact paper cohort, because the paper's exact subset was not fully available locally.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`, the SDK's trial table. Each trial is identified by `trials_id`. Stimulus presentation intervals are assigned to trials via the `trials_id` column in `stimulus_presentations`. A trial's time series data consists of all stimulus presentation intervals belonging to that trial.

ii.
```python
trials = ds.trials.copy()
# ...
kept_trials = trials.loc[keep_trials].copy()
# ...
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
# ...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
    # Neural, running, pupil data indexed by stim_idx
```

iii. The agent used the SDK's built-in trial definitions and the `trials_id` foreign key in `stimulus_presentations` to group intervals into trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, sessions are skipped if they have no valid ROI events, no eye tracking data, entirely NaN pupil width, or fewer than 2 kept trials.

ii.
```python
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (
    trials["go"] | trials["catch"]
)

# Session-level skips:
if len(ds.events) == 0: return None
if ds.eye_tracking.empty: return None
if not np.isfinite(pupil_width).any(): return None
if len(trial_data) < 2: return None
```

iii. The trial filter directly matches the instructions: "Include both the Go and Catch trials, but exclude the Aborted and Auto-rewarded trials." The session-level skips are practical quality controls.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `ds.events`, the AllenSDK's inferred calcium events (L0 event detection), not from `ds.dff_traces`.

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
```

iii. The agent stated: "This follows the paper's use of discrete calcium events rather than dF/F traces." The `events` property provides sparse, non-negative values at the rise time of each calcium transient.

## 2-b. How is the `neural` data processed?

i. The events matrix (n_cells x n_ophys_timestamps) is summed within each image-presentation interval using cumulative sum indexing. Each interval is defined by successive stimulus onset times in the active change_detection block. The result is one value per cell per interval.

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

iii. The agent chose to bin at the image-presentation interval level (~750 ms) rather than per-ophys-frame (~93 ms), arguing that "the paper explicitly assigns behavioral events to each 750 ms image presentation interval" and "image identity and change labels are naturally defined on these intervals."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural quality filtering is applied beyond what the AllenSDK already does. Sessions with zero valid ROIs (`len(ds.events) == 0`) are skipped entirely. Trials with all-zero neural activity are retained.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. The agent stated: "The AllenSDK already excludes invalid ROIs from events/cell_specimen_table, so no additional ROI-quality filter was added." Zero-activity trials were kept because "they are scientifically plausible with event-based calcium data."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets (image flashes) in the active change_detection block. The `ophys_timestamps` are used to map events into each image-presentation interval via `np.searchsorted`. Intervals are assigned to trials via `stimulus_presentations.trials_id`.

ii.
```python
stim = make_change_detection_table(ds)
# ...
interval_starts = stim["start_time"].to_numpy(dtype=np.float64)
interval_ends = stim["interval_end"].to_numpy(dtype=np.float64)
neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps, matrix=event_matrix,
    starts=interval_starts, ends=interval_ends,
)
```

iii. The agent described this as "Successive image-presentation intervals defined by active change_detection stimulus onsets and assigned to trials via trials_id; modalities aggregated using ophys/running/eye timestamps within each interval."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one time bin per image-presentation interval, approximately 750 ms. This is a rebinning from the native ophys frame rate (~10.7 Hz, ~93 ms per frame) to the stimulus presentation cadence. The reported median bin duration is 0.75061 s.

ii.
```python
"time_bin_size": float(np.median(interval_durations) * 1000.0),
# Results in ~750.6 ms
```

iii. The agent noted that median interval duration of 0.75061 s matches the whitepaper's 250 ms image + 500 ms gray cadence. This is a significant rebinning from the native frame rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations["image_name"]`, which contains string identifiers like `im061`, `im035`, `omitted`, etc.

ii.
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
# ...
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The agent retained "omitted" as a category, stating "omissions are an explicit part of the Visual Behavior task."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected from stimulus presentations, converted to sorted integer indices (with `omitted` sorted last), and stored as integer-coded categorical time series. NaN image names are filled with "unknown".

ii.
```python
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. Straightforward categorical encoding of the SDK's image name field.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is naturally aligned because both neural data and image identity are indexed by the same stimulus presentation intervals. Each interval has one image name and one neural activity vector.

ii.
```python
# Both use the same stim_idx indexing:
neural=neural_by_interval[:, stim_idx],
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. The alignment is inherent in the interval-based binning approach.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations["is_change"]`, a boolean column indicating whether the image changed at that presentation.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. Direct use of the SDK's `is_change` field.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean is cast to integer (0 or 1). NaN values are filled with False (no change).

ii.
```python
change_idx = trial.image_change.astype(np.int64)
```

iii. Minimal processing - just type conversion from boolean to integer.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed - `is_change` is already binary. It is stored as two categories: `no_change` (0) and `change` (1).

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    # ...
]
```

iii. Binary variable directly from the SDK.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - aligned through the shared stimulus presentation interval indexing.

ii.
```python
# Both neural and image_change use the same stim_idx
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. Inherent alignment via interval-based approach.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, specifically the `speed` column (in cm/s) with its own `timestamps`.

ii.
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. The `running_speed` property provides 10 Hz low-pass filtered running speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each image-presentation interval using `interval_reduce_mean()`. The mean is computed via cumulative sum for efficiency. If no running speed samples fall within an interval, the value is interpolated from surrounding timestamps.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
```

iii. Mean aggregation within each ~750 ms interval.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-frequency bins (quintiles) computed globally across all sessions and all time bins. Rank-based qcut is used to ensure balanced bins even with ties.

ii.
```python
def rank_quintiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)

all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)
```

iii. The agent stated: "This guarantees five balanced categories even with ties."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned through the same interval-based approach. Running speed samples are averaged within each image-presentation interval, producing one value per interval matching the neural data bins.

ii.
```python
running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v,
    starts=interval_starts, ends=interval_ends,
)
# Then indexed by stim_idx per trial
running_raw=running_by_interval[stim_idx],
```

iii. Aligned via shared interval boundaries.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking["pupil_width"]`, which is the width of the DeepLabCut-fitted pupil ellipse in pixels.

ii.
```python
eye_df = ds.eye_tracking.copy()
eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
pupil_filled = fill_nan_by_time(
    timestamps=eye_t,
    values=eye_df["pupil_width"].to_numpy(dtype=np.float64),
)
```

iii. The agent chose `pupil_width` because "the task requested pupil diameter; pupil_width is the direct diameter-like quantity exposed by the SDK."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values in `pupil_width` (from blink detection) are linearly interpolated over eye-tracking timestamps. Then pupil width is averaged within each image-presentation interval. Finally, quintile binning is applied globally.

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

pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
```

iii. Linear interpolation fills blink-related NaNs, then interval averaging reduces to one value per time bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-frequency bins (quintiles) computed globally across all sessions using rank-based qcut.

ii.
```python
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
```

iii. Global quintile binning ensures balanced categories.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same interval-based alignment as running speed. Pupil width samples are averaged within each interval.

ii.
```python
pupil_by_interval = interval_reduce_mean(
    timestamps=eye_t, values=pupil_filled,
    starts=interval_starts, ends=interval_ends,
)
pupil_raw=pupil_by_interval[stim_idx],
```

iii. Aligned via shared interval boundaries.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `ds.trials`.

ii.
```python
def infer_trial_outcome(trial_row: pd.Series) -> str:
    if bool(trial_row["hit"]): return "hit"
    if bool(trial_row["miss"]): return "miss"
    if bool(trial_row["false_alarm"]): return "false_alarm"
    if bool(trial_row["correct_reject"]): return "correct_reject"
    raise ValueError(f"Could not infer trial outcome for trial {trial_row.name}")
```

iii. Standard trial outcome categories from the Visual Behavior task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is a static per-trial value. Since the output format requires `(n_output, T)` arrays, the outcome integer index is broadcast (repeated) across all time bins within the trial.

ii.
```python
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
```

iii. The agent tiled the static value across time to fit the required output shape, as trial outcome is "Static per-trial" per the instructions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several NaN-handling strategies are used:
- **Pupil width NaNs** (blinks): linearly interpolated over time within each session.
- **Missing eye tracking**: entire session is skipped.
- **All-NaN pupil**: session is skipped.
- **`is_change` NaN**: filled as False.
- **`image_name` NaN**: filled as "unknown".
- **Running speed with no samples in interval**: interpolated from surrounding timestamps.
- **Zero-activity neural trials**: retained (not filtered).
- **No valid ROI events**: session is skipped.

ii.
```python
# Pupil NaN interpolation
pupil_filled = fill_nan_by_time(timestamps=eye_t, values=...)
# is_change NaN fill
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy()
# image_name NaN fill
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy()
# Session skips
if ds.eye_tracking.empty: return None
if not np.isfinite(pupil_width).any(): return None
```

iii. The agent documented each NaN-handling strategy in CONVERSION_NOTES.md and justified keeping zero-activity trials as "scientifically plausible with event-based calcium data."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB experiment via `cache.get_behavior_ophys_experiment()`, which involves reading large NWB files from disk. With 202 experiments to process, this dominates the runtime. The per-experiment processing (interval aggregation, trial splitting) is relatively fast due to vectorized operations.

ii.
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    print(f"[{idx}/{len(selected)}] loading experiment {experiment_id}")
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
```

iii. Each NWB file load involves parsing HDF5 data, constructing DataFrames, and running SDK-internal processing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop inside `process_experiment()` iterates over `kept_trials` to extract per-trial stimulus intervals. The inner loop in `build_decoder_dataset()` iterates over trials within sessions to build image index arrays. The image index mapping `[image_to_idx[name] for name in trial.image_names]` is a Python list comprehension that could use vectorized lookup.

ii.
```python
# Per-trial loop in process_experiment
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    # ...

# Per-trial loop in build_decoder_dataset
for trial in session.trials:
    image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. These loops could potentially be replaced with groupby operations or vectorized pandas/numpy operations.

## 9-c. What processing does the code repeat multiple times?

i. The full running speed and pupil diameter aggregation is computed for ALL intervals in the session, but then only the intervals belonging to kept trials are actually used. The `interval_reduce_mean` and `interval_reduce_sum_matrix` functions process all intervals upfront, then per-trial indexing selects the relevant subsets. Additionally, the format validation (`verify_data_format`) is run multiple times - once during conversion and again during verification.

ii.
```python
# All intervals computed
neural_by_interval = interval_reduce_sum_matrix(...)
running_by_interval = interval_reduce_mean(...)
pupil_by_interval = interval_reduce_mean(...)
# Then indexed per trial
neural=neural_by_interval[:, stim_idx]
running_raw=running_by_interval[stim_idx]
```

iii. Computing over all intervals first and then indexing is actually a reasonable design choice (vectorized computation), but it processes intervals from dropped trials unnecessarily.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that may not be used downstream:
- The `interval_durations` are computed per session and stored but only used for metadata statistics.
- The `omission_count` and `interval_count` per session are tracked but only used for summary statistics.
- `trial_count_before_filter` is computed but only used for logging.
- The `SessionData` dataclass stores metadata fields (behavior_session_id, cre_line, imaging_depth, session_type) that are placed into `session_info` metadata but not directly used by the decoder.
- Running speed and pupil values are computed for ALL intervals (including those belonging to excluded trials) before filtering.
- The `selected_summary` statistics are computed and stored in metadata.

ii.
```python
@dataclass
class SessionData:
    # ...
    interval_durations: np.ndarray        # only used for metadata
    trial_count_before_filter: int         # only used for logging
    trial_count_after_filter: int          # only used for logging
    omission_count: int                    # only used for summary
    interval_count: int                    # only used for summary
```

iii. These are mostly metadata/diagnostic fields that support documentation and sanity checking but are not used by the decoder itself.
