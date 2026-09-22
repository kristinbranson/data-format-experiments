# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the Allen SDK cache. It enumerates locally available NWB files from `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, joins them to `ophys_experiment_table.csv`, drops passive experiments, and then opens each NWB directly with `h5py`. Within each retained file it separately reads the trial table, task stimulus presentations, event-detection traces, running speed, and pupil data.

ii. ```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)

    experiment_dir = data_root / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    available_files = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
    }

    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
```

```python
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. The justification is stated in `CONVERSION_NOTES.md` Steps 4-6: the agent believed the local AllenSDK/NWB stack could not instantiate these NWB files, so it chose direct `h5py` reads of the processed NWB contents. It also explicitly decided to use only locally available active experiment files.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the retained session metadata. The final subject list is sorted, and each converted session stores an integer index into that list.

ii. ```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes repeatedly describe `mouse_id` as the subject identifier, mirroring the project metadata. No alternative subject grouping is introduced.

## 1-c. How are the data split into sessions?

i. The agent treats each `behavior_ophys_experiment_<ophys_experiment_id>.nwb` file as one converted session. It does not group multiple experiments by shared `ophys_session_id`; instead, each `SessionMeta` row is converted independently.

ii. ```python
sessions.append(
    SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        ...
        filepath=available_files[int(row.ophys_experiment_id)],
    )
)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session=session, ...)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent makes this explicit: “Treat each `ophys_experiment_id` file as one converted session,” arguing that this matches the `BehaviorOphysExperiment` object granularity and keeps one neuron set / one targeted structure per converted session.

## 1-d. How are the data split into trials?

i. Trials come from the processed NWB `intervals/trials` table. After filtering, each trial is represented by a new regular 30 Hz grid whose bin centers run from `start_time` to `stop_time`.

ii. ```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id", "start_time", "stop_time", "go", "catch", "aborted",
        "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "initial_image_name", "change_image_name",
    ]
    trials = read_interval_table(f["intervals"]["trials"], columns)
```

```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
    if centers.size == 0:
        continue
```

iii. The notes say the NWB `trials` table is taken as the authoritative Allen-processed trial definition and that trial windows should span full `start_time` to `stop_time` so outputs can vary across the trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are labeled `go` or `catch`, not `aborted`, and not `auto_rewarded`. Trials with empty/invalid windows are dropped, and whole sessions are dropped if they end up with fewer than 2 kept trials. Sessions are also skipped earlier if they lack task stimulus presentations or eye tracking.

ii. ```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    ...
```

```python
presentations = get_task_presentations(f)
if presentations.empty:
    ...
...
if centers.size == 0:
    continue
```

```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

iii. The justification in Steps 4-5 is that contingent trials are GO and CATCH only, aborted and auto-rewarded trials should be excluded, pupil output is required so sessions missing eye tracking must be removed, and each session must retain at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from the NWB event-detection stream, specifically `processing/ophys/event_detection/data` and its timestamps, not from dF/F.

ii. ```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The agent justifies this in Steps 4-5 of `CONVERSION_NOTES.md`: the paper’s neural analyses were read as using calcium events, so it chose raw event magnitudes instead of dF/F.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly resampled from native ophys timestamps onto each trial’s 30 Hz bin centers. The stored trial matrix is neuron-by-time because `linear_resample_matrix` interpolates a time-by-neuron matrix and then transposes it. No extra normalization or denoising is applied.

ii. ```python
def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
neural_trials.append(neural_trial.astype(np.float32, copy=False))
```

iii. The notes say the event traces are already processed upstream and that the extra processing decision was to place every stream on a common 30 Hz trial grid because the local dataset mixes native sampling rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter in `convert_data.py`. The code uses whatever rows are present in the NWB cell table / event matrix and only assigns brain-region indices by cell count. Neural inclusion is therefore effectively “all cells present in the file,” while session-level filters (eye tracking, trial count, task presentations) can exclude whole sessions.

ii. ```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

```python
events = np.asarray(event_group["data"][:], dtype=np.float32)
...
brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
```

iii. The notes state that no additional ROI curation was applied because the agent believed the listed local cells were already valid after Allen preprocessing. That justification appears in the consistency-review sections rather than in the code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data is aligned to trial start. The code builds `centers` relative to each trial’s `start_time`, samples neural events at those absolute times, and records metadata saying the alignment event is `trial start`.

ii. ```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "trial start",
    "off_start": 0.0,
    "off_end": None,
    ...
}
```

iii. The justification in Steps 4-5 is that all streams should first be aligned in absolute ophys time and then cut into trial windows, with trial start used as the trial-alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a fixed 30 Hz grid (`DT = 1/30`), so each time bin is 33.333... ms. Yes: temporal rebinning / resampling is applied to native ophys, running, and pupil streams.

ii. ```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    ...
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Step 4 and Step 5 of the notes explicitly justify a common 30 Hz grid: the agent wanted one bin size for all sessions despite mixed native acquisition rates and considered 30 Hz close to the behavioral / eye-tracking sampling rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from task stimulus-presentation interval tables, not from the trial table’s `initial_image_name` / `change_image_name` pair. The code reads per-presentation `image_name`, `omitted`, `is_change`, `trials_id`, and presentation times.

ii. ```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    columns = [
        "start_time", "stop_time", "image_name", "omitted", "is_change",
        "trials_id", "stimulus_block_name", "active", "duration",
    ]
    ...
    return out.sort_values("start_time").reset_index(drop=True)
```

```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes justify this by saying the stimulus-presentation tables preserve actual flashed-image timing, omissions, and change labels, while non-task stimulus blocks can be filtered out by `stimulus_block_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, image identity starts as `gray` for all bins. The code then overwrites bins covered by stimulus presentations: omitted presentations stay `gray`, non-omitted presentations take their `image_name`. A global categorical mapping is built over all encountered image names, with `gray` forced to index 0 if present.

ii. ```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. The justification from Step 5 is explicit: the agent chose to “encode gray/omission periods explicitly” and to construct the image-identity trace from stimulus presentations rather than only from pre/post change trial labels.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by assigning values on the exact same 30 Hz `centers` array used to create the neural trial matrix. Presentation start/stop times are converted into boolean masks over those bin centers.

ii. ```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes justify alignment by saying all outputs are resampled or sampled onto the same trial grid as the neural data after absolute-time alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag plus each presentation’s start/stop interval. The trial-table `change_time` is read into memory but is not used to generate the final `image_change` output.

ii. ```python
columns = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "trials_id", "stimulus_block_name", "active", "duration",
]
```

```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. In Step 5 the agent says image-change should come from stimulus-presentation rows and not from a reconstructed trial-level window, because those rows directly encode the flashed change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes `image_change` to 0 for the full trial and sets it to 1 only on bins whose centers fall inside a stimulus-presentation interval marked `is_change`.

ii. ```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. The notes describe this as a per-bin binary trace that is 1 during the change-image flash interval and 0 otherwise.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No threshold beyond binary coding is used. The variable has two categories: 0 for no change and 1 for change.

ii. ```python
output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
],
```

iii. This follows the decoder task directly; the notes describe image change as a binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned on the same 30 Hz trial grid as the neural data by applying presentation-time masks to the shared `centers` vector.

ii. ```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The stated rationale is the same shared-grid alignment used for all outputs: first define trial bin centers, then place each behavioral / stimulus variable onto those bin centers.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running stream `processing/running/speed`, specifically its `timestamps` and `data` arrays.

ii. ```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes treat this as the processed Allen running-speed stream and do not derive running from lower-level wheel signals themselves.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to each trial’s 30 Hz centers and then discretized with global quantile edges computed over all finite running samples from all kept trials in pass 1.

ii. ```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
running_all = np.concatenate(running_values).astype(np.float64, copy=False)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The notes justify global percentile binning so category semantics stay consistent across sessions, and justify interpolation because all outputs are being mapped to a common trial grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five dataset-wide equal-quantile bins.

ii. ```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. This follows the task requirement for five equal percentile bins, and the notes explicitly say the bins are computed globally rather than per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned with neural data by evaluating both on the same per-trial 30 Hz centers array.

ii. ```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The agent’s stated alignment rule is shared absolute-time sampling onto one common grid for all streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking timestamps plus the pupil ellipse `width` and `height` arrays under `acquisition/EyeTracking/pupil_tracking`. The agent computes diameter as `2 * max(width, height)` per sample.

ii. ```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
```

iii. In Step 5 the notes say the pupil signal will be computed from width/height and that sessions missing eye tracking should be excluded because pupil is a required decoder output.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code fills missing pupil samples by time interpolation within the raw eye-tracking stream, then linearly resamples that continuous signal to the 30 Hz trial centers and discretizes it with global quantile edges. It does not use the SDK `likely_blink` mask.

ii. ```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. The notes justify this as interpolation through blink-related NaNs rather than dropping blink frames, and again rely on global percentile bins for consistent categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five dataset-wide equal-quantile bins.

ii. ```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. This is the same quantile-binning strategy used for running speed and is described that way in the notes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned with neural data by interpolation onto the same trial-center grid used for the resampled neural matrix.

ii. ```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes describe pupil alignment exactly the same way as running alignment: all streams share the 30 Hz trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]


def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
```

iii. The notes treat these as the mutually exclusive Allen trial-outcome labels for kept contingent trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The output is encoded as an integer category 0-3 using that fixed order, and the chosen category is repeated across every time bin of the trial as a constant trace.

ii. ```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
...
output_trial = np.vstack(
    [
        image_identity,
        image_change,
        running_bin,
        pupil_bin,
        outcome_trace,
    ]
)
```

iii. Step 5 of the notes explicitly says trial outcome will be represented as a static per-trial label repeated across bins so all outputs share the same `(n_output, T)` format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles data problems mostly by skipping sessions or interpolating missing continuous values. Missing eye tracking triggers a `KeyError` and the session is excluded in pass 1. Sessions with no task presentations or fewer than 2 kept trials are excluded. Individual trials with invalid or empty time windows are skipped. Pupil NaNs are filled by interpolation in time; if all pupil values are NaN, `fill_nan_by_time` raises and the session would fail. There is no special handling for missing `change_time` because the final outputs come from stimulus presentations rather than from trial `change_time`.

ii. ```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

```python
if len(trials) < 2:
    ...
if presentations.empty:
    ...
if centers.size == 0:
    continue
```

```python
def fill_nan_by_time(time_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("All values are NaN")
    ...
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
```

iii. The notes justify these choices by saying pupil is required, so sessions without eye tracking should be removed; modest within-session missingness can be interpolated; and structurally unusable sessions/trials should be skipped rather than forcing malformed output.

## 9-a. What are the most time-consuming steps of the code?

i. The agent explicitly identifies whole-file I/O and the two-pass design as the dominant runtime costs. In practice, pass 1 scans each retained NWB once to gather running/pupil statistics and pass 2 reopens each file to do the full conversion, with large event matrices being read both times.

ii. ```python
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. Step 6 and Step 7 of `CONVERSION_NOTES.md` say the conversion is I/O-heavy because each NWB event matrix must be read from disk and that the two-pass design means files are read twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest expensive interpolation loop was already vectorized for neural data, but several other loops remain scalar / row-wise: string decoding, interval-table column collection, the per-trial pass-1 accumulation loop, and the per-trial / per-presentation loops in `convert_session`.

ii. ```python
def decode_str_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in values:
        ...
```

```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    ...
    running_values.append(running_trial)
    pupil_values.append(pupil_trial)
```

```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        ...
```
```python
def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
```

iii. The notes explicitly say the agent vectorized matrix interpolation because that was the important hot path, but the overall workflow still contains trial-by-trial and presentation-by-presentation Python loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats work across two passes. It opens every retained file once in `collect_global_statistics` and again in `convert_session`. Running and pupil are interpolated once in pass 1 just to compute global quantile edges and then interpolated again in pass 2 for the final output.

ii. ```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

```python
# pass 1
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
# pass 2
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The justification is again in Step 6: the two-pass design was chosen so the running and pupil bin edges could be computed globally without retaining all per-trial converted outputs in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several fields that are not used in the final converted dataset. Most notably, the trial table includes `change_time`, `initial_image_name`, and `change_image_name`, but the output construction uses stimulus-presentation tables instead. `get_task_presentations` also reads `stimulus_block_name`, `active`, and `duration`, though only `image_name`, `omitted`, `is_change`, `trials_id`, and timing matter downstream. `SessionMeta` stores `session_type`, `experience_level`, and `project_code`, but those metadata are not propagated into the saved dataset.

ii. ```python
columns = [
    "id", "start_time", "stop_time", "go", "catch", "aborted",
    "auto_rewarded", "hit", "miss", "false_alarm", "correct_reject",
    "change_time", "initial_image_name", "change_image_name",
]
```

```python
columns = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "trials_id", "stimulus_block_name", "active", "duration",
]
```

```python
class SessionMeta:
    ...
    session_type: str
    experience_level: str
    project_code: str
```

iii. This is not presented as a deliberate scientific choice in the notes; it is a by-product of reading broader metadata tables than the final conversion actually needs.
