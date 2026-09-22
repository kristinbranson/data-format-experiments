# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the AllenSDK cache. It reads local NWB files and metadata CSVs directly with `h5py`/`pandas`, discovers sessions from `ophys_experiment_table.csv`, keeps only rows whose NWB files are present locally, drops passive sessions, then opens each NWB file in two passes.

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
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. In `CONVERSION_NOTES.md`, the agent says it read NWBs directly with `h5py` because the local AllenSDK/NWB stack could not load these files, and that it intentionally restricted itself to locally available active experiment files.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment metadata.

ii. 
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes describe `mouse_id` as the session-to-subject key from the metadata table, so the agent uses it directly rather than inferring subjects from filenames or session IDs.

## 1-c. How are the data split into sessions?

i. Each local `behavior_ophys_experiment_<ophys_experiment_id>.nwb` file is treated as one session. The agent does not group multiple experiment files that share an `ophys_session_id`.

ii. 
```python
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ophys_session_id=int(row.ophys_session_id),
            behavior_session_id=int(row.behavior_session_id),
            ...
            filepath=available_files[int(row.ophys_experiment_id)],
        )
    )
```

```python
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent explicitly says to “treat each `ophys_experiment_id` file as one converted session” because that matches the local NWB/Allen object granularity.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each kept trial, the agent uses the Allen-processed `start_time` and `stop_time` bounds and builds a variable-length 30 Hz grid spanning that interval.

ii. 
```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = [
        "id", "start_time", "stop_time", "go", "catch",
        "aborted", "auto_rewarded", "hit", "miss",
        "false_alarm", "correct_reject", "change_time",
        "initial_image_name", "change_image_name",
    ]
    trials = read_interval_table(f["intervals"]["trials"], columns)
```

```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
    if centers.size == 0:
        continue
```

iii. The notes say the NWB `trials` table is the authoritative Allen-processed trial definition, so the agent avoids reconstructing trials from lower-level logs.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only `go` or `catch` trials, removes `aborted` and `auto_rewarded` trials, skips zero-length trial windows, and later rejects sessions with fewer than two usable trials.

ii. 
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. In Steps 4 and 5 of the notes, the agent justifies this as matching “contingent trials” from the Allen task while excluding behaviorally invalid aborted and auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB `processing/ophys/event_detection` group: specifically its `timestamps` and `data` datasets.

ii. 
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The notes state that the paper’s neural analyses used calcium event traces, so the agent chose stored event magnitudes instead of dF/F.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from native ophys timestamps onto a per-trial 30 Hz grid. The raw stored matrix is time-by-neuron; the interpolation routine returns neuron-by-time output for each trial.

ii. 
```python
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
```

iii. The notes say the agent wanted a common 30 Hz grid across all sessions and streams, while keeping “ophys time” as the alignment reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level filtering in the conversion code. The agent uses all rows present in the NWB event matrix and creates a brain-region index array whose length is the total number of entries in `cell_specimen_table`.

ii. 
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

```python
ophys_time, events = get_neural_data(f)
...
brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
```

iii. In the notes, the agent argues that ROI curation had already been applied upstream in the Allen pipeline and, for its local subset, that listed cells were effectively already valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. For each trial, the agent creates bin centers from `start_time` to `stop_time` and samples the neural event trace on those absolute-time centers.

ii. 
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    ...
    return centers[valid]
```

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes say the agent “aligns by absolute ophys time, then cuts into trials,” with trial start used as the per-trial zero point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 30 Hz grid with `DT = 1/30 s` (`33.333... ms` per bin). Yes: all signals, including neural events, are resampled to that grid.

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
    "neural_signal": "ophys event magnitudes from NWB event_detection",
}
```

iii. The notes justify this as a way to force a common bin size across sessions because the agent believed native ophys frame rates varied across rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the task stimulus-presentation interval tables under `intervals/*`, using `image_name`, `omitted`, `trials_id`, `start_time`, and `stop_time`. It is not derived from the trial table’s `initial_image_name`/`change_image_name` in the final implementation.

ii. 
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name",
    "active", "duration",
]
df = read_interval_table(group, columns)
```

```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. In Step 5 of the notes, the agent says stimulus presentation tables were used to build a time-varying image-identity signal and to represent gray/omission periods explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent initializes every time bin to a `"gray"` category, then overwrites bins covered by non-omitted stimulus presentations with the presented image name. A global codebook is built across all kept sessions, with `"gray"` forced to index 0.

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
    ...
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes say the agent wanted image identity to be fully time-varying and to distinguish actual image flashes from gray or omitted periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same per-trial 30 Hz `centers` array used for neural interpolation. Each stimulus presentation writes codes into the bins whose centers fall within that presentation interval.

ii. 
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
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

iii. The notes justify this by saying all outputs should share the same common 30 Hz trial grid as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag plus each presentation’s `start_time` and `stop_time`.

ii. 
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name",
    "active", "duration",
]
```

```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes describe `is_change` in the presentation table as the way to mark the actual change flash on the resampled grid.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent starts with zeros and sets bins to 1 only for time bins that fall inside stimulus-presentation intervals labeled `is_change`.

ii. 
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. The notes say image change should mark the change-image flash itself rather than using a broader post-change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already a binary categorical signal: `0` for no change and `1` for change. No additional thresholding is applied.

ii. 
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
]
```

iii. The task asked for a binary “image change” output, and the notes frame this as a direct change-vs-no-change indicator.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is written onto the same per-trial 30 Hz `centers` array used for the neural data.

ii. 
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say all outputs were intentionally sampled onto the same shared trial grid as the neural signal.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed`, specifically its `timestamps` and `data` arrays.

ii. 
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes describe this as the processed Allen running-speed stream already stored in the NWB.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The running trace is linearly interpolated onto each trial’s 30 Hz grid, then discretized with global 5-quantile edges computed in pass 1 across all kept trials.

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

iii. The notes justify this as a dataset-wide five-bin categorical output with consistent semantics across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-percentile bins using global quantile edges.

ii. 
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
```

```python
running_bin = digitize_with_edges(running_trial, running_edges)
...
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
```

iii. In Step 5, the agent says running and pupil bins should be global equal-percentile bins so categories are consistent across the dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz per-trial `centers` grid used for neural interpolation.

ii. 
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes say all streams were aligned in absolute time and then sampled onto one common ophys-referenced trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking acquisition group, using `pupil_tracking/width`, `pupil_tracking/height`, and `eye_tracking/timestamps`.

ii. 
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The notes say the agent wanted an explicit diameter-like quantity from the raw eye-tracking ellipse fit rather than using a precomputed `pupil_width` column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent computes `2 * max(width, height)` per eye-tracking sample, fills NaNs by linear interpolation in the native eye-tracking timebase, then interpolates the result onto each trial’s 30 Hz grid and discretizes it into five global quantile bins.

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

iii. The notes justify session-level pupil inclusion and interpolation as necessary because pupil diameter is a required decoder output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-percentile bins using global quantile edges.

ii. 
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
```

iii. As with running speed, the notes say the bins are global so the category labels mean the same thing in every session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating onto the same per-trial 30 Hz `centers` grid used for neural, image, and running signals.

ii. 
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes repeatedly describe the conversion as building one shared trial grid for all outputs and neural activity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes identify those four Allen trial labels as the desired categorical outcomes for kept go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The booleans are mapped to integer category IDs in fixed order, and that category is repeated across all time bins within the trial.

ii. 
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes say all outputs were stored in a uniform `(n_output, T)` time-varying format, so the static trial outcome is broadcast across the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code skips sessions with missing eye tracking or missing task presentations, skips zero-length trials, linearly fills NaNs in pupil width/height-derived diameter, and requires at least two usable trials per session. Running and pupil resampling otherwise use `np.interp`, which also extrapolates edge values rather than leaving NaNs.

ii. 
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

```python
def fill_nan_by_time(time_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("All values are NaN")
    if finite.all():
        return values
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
```

iii. The notes justify skipping sessions that cannot support a required output and interpolating pupil gaps because pupil diameter is mandatory in the target format.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are opening each NWB file and reading/resampling large event matrices twice: once in pass 1 for global running/pupil statistics and once in pass 2 for full trial conversion.

ii. 
```python
for idx, session in enumerate(sessions, start=1):
    with h5py.File(session.filepath, "r") as f:
        ...
```

```python
print("Pass 1: collecting global running/pupil statistics")
...
print("Pass 2: converting sessions")
```

iii. In Step 6 and Step 7 of the notes, the agent explicitly calls the conversion I/O-heavy and says the two-pass design makes the code read each file twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over sessions, then over trials within each session, and then over stimulus presentations within each trial. The most obvious remaining vectorization targets are the per-trial resampling/binning loop and the per-presentation mask loop used to write image identity and change.

ii. 
```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    ...
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

```python
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
```

iii. The notes say the agent already vectorized neural interpolation with `searchsorted` plus broadcasting, but accepted the remaining Python loops because disk I/O was the main bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats session loading and some resampling work because it uses a two-pass design. Pass 1 loads every session to collect global running/pupil samples and image names; pass 2 reloads every kept session to build neural/output trial matrices.

ii. 
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly call out the two-pass design as a tradeoff: exact global quantile binning without keeping all full-session arrays in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads some trial-table and presentation-table columns that are not used in the final outputs, and the optional processing-plot path performs extra diagnostics that are not part of the saved dataset. It also carries `stimulus_block_name` through the presentation table after filtering even though the final pickle does not use it.

ii. 
```python
columns = [
    "id", "start_time", "stop_time", "go", "catch",
    "aborted", "auto_rewarded", "hit", "miss",
    "false_alarm", "correct_reject", "change_time",
    "initial_image_name", "change_image_name",
]
```

```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name",
    "active", "duration",
]
...
if show_processing and not plotted:
    make_processing_plot(...)
```

iii. The notes frame these as debugging and interpretability aids rather than part of the downstream decoder input/output representation.
