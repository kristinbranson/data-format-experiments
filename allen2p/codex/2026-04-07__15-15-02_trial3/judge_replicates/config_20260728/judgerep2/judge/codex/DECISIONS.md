# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not use the Allen SDK cache path used in the human reference. It loaded a local metadata CSV, found locally present NWB files, filtered out passive experiments, and then opened each NWB directly with `h5py`. Trials, neural data, running data, pupil data, and stimulus presentations were then read from NWB groups on demand.

ii.
```python
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
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(
        session=session,
        running_edges=running_edges,
        pupil_edges=pupil_edges,
        image_value_to_idx=image_value_to_idx,
        region_to_idx=region_to_idx,
    )
```

iii. In `CONVERSION_NOTES.md`, the agent justified direct NWB reads because it believed the local AllenSDK/NWB stack could not load these files cleanly. It also explicitly chose to operate on the locally available active experiment files rather than the full project cache.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by unique `mouse_id` values from the retained session metadata. The final `subjects` list is built after pass-1 filtering.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes describe `mouse_id` as the subject identifier taken from the experiment metadata table, so the subject split follows the NWB/CSV metadata rather than being inferred from filenames or sessions.

## 1-c. How are the data split into sessions?

i. The agent treated each local `behavior_ophys_experiment_<ophys_experiment_id>.nwb` file as one converted session. It did not group multiple experiments by shared `ophys_session_id`.

ii.
```python
sessions.append(
    SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        behavior_session_id=int(row.behavior_session_id),
        mouse_id=str(row.mouse_id),
        targeted_structure=str(row.targeted_structure),
        ...
        filepath=available_files[int(row.ophys_experiment_id)],
    )
)
```

```python
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session=session, ...)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent explicitly said it would treat each `ophys_experiment_id` file as one session because that matched the local NWB granularity and gave one imaging plane per converted session.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB `intervals/trials` table. For each kept trial, the agent uses the Allen-produced `start_time` and `stop_time` bounds and constructs a variable-length per-trial time grid between those two times.

ii.
```python
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

iii. The notes say the NWB `trials` table was treated as the authoritative Allen-processed trial definition, avoiding reimplementation of trial logic from lower-level files.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Zero-length trial grids are skipped. Sessions are also dropped if they have fewer than 2 kept trials, no task stimulus presentations, or no eye-tracking acquisition.

ii.
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    ...
presentations = get_task_presentations(f)
if presentations.empty:
    ...
```

```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

iii. The justification in the notes was that the task explicitly asked for Go and Catch trials while excluding aborted and auto-rewarded trials, and that sessions lacking required pupil output should be excluded entirely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from NWB event-detection output, specifically `processing/ophys/event_detection/timestamps` and `processing/ophys/event_detection/data`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. In Steps 4 and 5 of `CONVERSION_NOTES.md`, the agent said it chose event traces instead of dF/F because the paper’s neural analyses used calcium events and the NWB already contained event-detection outputs.

## 2-b. How is the `neural` data processed?

i. The agent linearly resampled the event matrix from native ophys timestamps onto a uniform 30 Hz per-trial grid. It transposed the time-by-neuron event matrix into neuron-by-time trial matrices. It did not merge multiple imaging planes because each NWB file was treated as its own session.

ii.
```python
DT = 1.0 / 30.0

def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
neural_trials.append(neural_trial.astype(np.float32, copy=False))
```

iii. The notes justify this as a way to put all sessions on a common bin size while staying tied to ophys time. The agent also described the vectorized interpolation as a speed optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality-control filter in the conversion code. The agent uses the event matrix as stored in the NWB and assigns one brain-region index per listed cell specimen.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

```python
ophys_time, events = get_neural_data(f)
brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
```

iii. In the notes, the agent argued that the NWB contents already reflected upstream Allen processing and that no extra neuron filtering was needed in the conversion step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned by absolute ophys time within each trial window. For each trial, the agent builds bin centers from `start_time` to `stop_time` and samples neural activity onto those centers. Metadata then labels the alignment event as `trial start`.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    return centers[valid]
```

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
"temporal_alignment_event": "trial start",
```

iii. The notes say the agent wanted to align all streams on absolute ophys time first and then express each trial on a shared trial-local 30 Hz grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 30 Hz bins (`33.333...` ms), and the code explicitly rebins/resamples neural and behavioral streams onto that grid.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

```python
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "sampling_grid_hz": 30.0,
    ...
}
```

iii. The justification in `CONVERSION_NOTES.md` was that native acquisition rates differed across rigs and that a shared 30 Hz grid would satisfy the decoder-format requirement of a common time bin size across all sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the task stimulus-presentation interval tables, using `image_name`, `omitted`, `start_time`, `stop_time`, `trials_id`, and block labels to select the change-detection block.

ii.
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "trials_id", "stimulus_block_name", "active", "duration",
]
df = read_interval_table(group, columns)
...
out["image_name"] = out["image_name"].astype(str)
out["omitted"] = out["omitted"].fillna(0).astype(bool)
out["trials_id"] = out["trials_id"].astype(np.int64)
```

```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes say the agent preferred stimulus presentations over the trial table for image identity because they provide explicit flash timing and make it possible to represent gray and omitted periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent initializes each trial to `gray`, then overwrites bins covered by stimulus presentations with the presented image name unless the presentation is omitted, in which case the bins remain `gray`. A global categorical mapping is built from all observed image names plus `gray`.

ii.
```python
image_names = set(["gray"])
...
image_names.update(x for x in presentations["image_name"].unique() if x != "omitted")
...
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The agent justified this in the notes by saying the task is a flashed-image paradigm with inter-stimulus gray periods, so it wanted the output trace to distinguish actual image flashes from gray/omitted time bins.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same 30 Hz trial-bin centers used for neural data. Each stimulus presentation paints categorical values into the same `centers` array that neural activity was resampled onto.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes describe all outputs as being built on the common trial grid so they remain exactly time-aligned with the converted neural matrix.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation `is_change` flag and each presentation’s `start_time` and `stop_time`.

ii.
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "trials_id", "stimulus_block_name", "active", "duration",
]
```

```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say the agent used the processed stimulus-presentation table because it directly marks change flashes and avoids reconstructing them from trial-level fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent initializes a zero vector and sets bins to 1 only for time bins lying inside stimulus presentations whose `is_change` flag is true.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The justification in the notes was that the stimulus table already encodes the actual change flashes, so the code can use that processed annotation directly rather than inferring a post-change window from `change_time`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: `0` for no change and `1` for change. No further thresholding is performed.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
],
```

iii. The task asked for a binary image-change variable, so the agent kept the representation as a direct binary indicator.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The image-change trace is filled on the same per-trial 30 Hz `centers` array used for neural resampling.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. As with image identity, the notes describe image change as being expressed on the common trial grid so every output shares the same time axis as the neural matrix.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB `processing/running/speed` dataset: its `timestamps` and `data` arrays.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The agent’s notes describe this as the processed running-speed stream already stored in the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz bin centers, then discretized using global quantile edges computed across all retained sessions and trials.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

```python
running_all = np.concatenate(running_values).astype(np.float64, copy=False)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. The notes justify this as a global five-quantile discretization on the same shared time grid used for neural data and other outputs.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The agent computes five equal-quantile bins over all finite running-speed samples from retained sessions, then digitizes each trial’s running trace into integer bins `0` through `4`.

ii.
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    ...
    return edges
```

```python
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```

iii. The notes explicitly say running speed should be discretized globally into five equal-percentile bins so category semantics stay consistent across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same 30 Hz trial-bin centers used for the neural data.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes say all modalities were mapped to the common trial grid so decoder inputs and outputs are exactly synchronized with the converted neural matrices.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking acquisition data, specifically `eye_tracking/timestamps` plus `pupil_tracking/width` and `pupil_tracking/height`.

ii.
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. In the notes, the agent states it read eye-tracking values directly from NWB because pupil output was required and some sessions lacked this acquisition entirely.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes diameter as `2 * max(width, height)`, fills NaNs over time by interpolation, resamples onto the 30 Hz trial grid, and discretizes using global quantile edges.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
return timestamps, diameter
```

```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes justify this as a way to keep sessions with blink-related missingness by interpolating within-session and then applying the same global five-bin discretization as for running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. As with running speed, the agent computes five global quantile bins over all finite pupil-diameter samples and digitizes each trial’s resampled pupil trace into bins `0` through `4`.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes say pupil diameter, like running speed, was meant to be a five-level percentile-binned categorical output consistent across the full converted dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same 30 Hz trial-bin centers used for the neural data.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes describe a common time grid for neural, running, pupil, and stimulus outputs, so pupil alignment follows the same mechanism as the other continuous streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trial table.

ii.
```python
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

iii. The notes treat these four mutually exclusive Allen trial labels as the canonical source for per-trial outcome.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the four boolean outcome labels to fixed integer categories and repeats the resulting category across all time bins in a trial.

ii.
```python
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

iii. The notes justify this as a static per-trial output represented in time-varying format for consistency with the rest of the decoder outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases: missing eye-tracking acquisition causes the session to be skipped; missing/NaN pupil samples are interpolated over time; missing `id` in the trial table is synthesized; zero-length trial windows are skipped; sessions with no task presentations or fewer than two usable trials are skipped; duplicate quantile edges are perturbed upward by machine epsilon.

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
diameter = fill_nan_by_time(timestamps, diameter)
```

```python
if "id" not in trials:
    trials["id"] = np.arange(len(trials), dtype=np.int64)
...
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

```python
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. The notes say these choices were made to keep the pipeline robust on imperfect local NWB files while still enforcing the required decoder outputs, especially pupil diameter.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming operations are repeated NWB file I/O and trial-wise resampling, especially loading the event matrices during both pass 1 and pass 2.

ii.
```python
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. In the notes, the agent explicitly called full conversion “I/O-heavy” and said the two-pass design requires reading each file twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized interpolation across neurons in `linear_resample_matrix`, but the code still contains Python-level loops over sessions, trials, stimulus presentations, and string decoding that could be further reduced or cached.

ii.
```python
for idx, session in enumerate(kept_sessions, start=1):
    ...
    for trial_idx, trial in trials.iterrows():
        ...
        for row in trial_presentations.itertuples(index=False):
            ...
```

```python
def decode_str_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in values:
        ...
```

iii. The notes emphasize that the main vectorization effort was the neural interpolation itself; the remaining loops were left in place for clarity and because the larger cost was still file I/O.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats file opening and data extraction across two passes. It first resamples running and pupil traces for every kept trial to compute global bin edges, then reopens the same files and repeats per-trial loading/resampling during final conversion.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
    running_values.append(running_trial)
    pupil_values.append(pupil_trial)
```

iii. The notes explicitly acknowledge this duplication and justify it as the price of computing global percentile edges before producing the final categorical outputs.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion code reads several fields that are not ultimately used in the saved decoder arrays, including `change_time`, `initial_image_name`, and `change_image_name` from the trials table, and some stimulus-presentation metadata columns such as `active`, `duration`, and `stimulus_block_name` after filtering. It also constructs optional diagnostic plots that are not consumed downstream.

ii.
```python
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
if show_processing and not plotted:
    ...
    make_processing_plot(...)
```

iii. The notes frame some of this as intentional debugging or sanity-check support, but these pieces are not needed by the final `converted_data.pkl` consumer.
