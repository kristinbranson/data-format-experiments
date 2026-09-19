# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads the local metadata CSV `ophys_experiment_table.csv`, finds locally present `behavior_ophys_experiment_*.nwb` files, drops passive sessions, and then opens each retained NWB directly with `h5py`. It performs two passes over those files: one pass to collect global running/pupil statistics and a second pass to convert sessions.

ii.
```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)
    ...
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    exp_table = exp_table.sort_values("ophys_experiment_id")
```

```python
for idx, session in enumerate(sessions, start=1):
    with h5py.File(session.filepath, "r") as f:
        ...
```

iii. In `CONVERSION_NOTES.md`, the AI says it switched to direct local NWB reads because the local AllenSDK/NWB stack had version-mismatch issues and because only a local subset of NWBs was available. It also explicitly chose to exclude passive sessions as outside the active task.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the filtered local experiment table.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The notes justify this as using the metadata table’s canonical subject identifier after session-level filtering.

## 1-c. How are the data split into sessions?

i. The AI treats each local `ophys_experiment_id` NWB file as one converted session. It does not merge multiple imaging planes belonging to the same `ophys_session_id`.

ii.
```python
for row in exp_table.itertuples(index=False):
    sessions.append(
        SessionMeta(
            ophys_experiment_id=int(row.ophys_experiment_id),
            ophys_session_id=int(row.ophys_session_id),
            ...
            filepath=available_files[int(row.ophys_experiment_id)],
        )
    )
```

iii. The notes state this was deliberate: one `behavior_ophys_experiment` file was treated as one session because it matches `BehaviorOphysExperiment` granularity and yields a single plane / neuron set / region per converted session.

## 1-d. How are the data split into trials?

i. Trials are taken from the processed NWB `intervals/trials` table. For each kept trial, the AI constructs a variable-length trial grid from `start_time` to `stop_time` at 30 Hz and uses that grid for all outputs and neural data.

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
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The notes say the processed NWB trial table was used as the authoritative Allen-processed trial definition, to avoid reimplementing trial logic and to bypass the broken high-level loader.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Trials with zero valid bins are skipped. Sessions are skipped if they have fewer than 2 kept trials, no task stimulus presentations, or no eye-tracking data.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    ...
if presentations.empty:
    ...
```

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
```

iii. The notes justify keeping only contingent trials (`go` and `catch`) and excluding aborted and auto-rewarded trials to match the task. They also justify dropping sessions without eye tracking because pupil diameter is a required decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from NWB `processing/ophys/event_detection/data` and its corresponding `timestamps`, not from dF/F.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. The notes justify this by citing the paper and whitepaper’s use of calcium event traces, and explicitly say to use raw event magnitudes rather than `filtered_events` or recomputed dF/F.

## 2-b. How is the `neural` data processed?

i. The AI linearly interpolates the event matrix from native ophys timestamps onto the 30 Hz trial grid. The raw NWB matrix is time-by-neuron, and the interpolation routine returns neuron-by-time arrays.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    ...
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes justify this as a common 30 Hz representation across sessions while keeping ophys-based timing. They also note the interpolation was vectorized for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level QC in `convert_data.py`. The code uses every row in `event_detection/data`, and the neuron count comes from all rows in `cell_specimen_table`.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The notes say the NWB files already reflect Allen processing/ROI curation and report that included sessions had all listed cells valid, so no extra filtering was added in the converter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned using absolute ophys timestamps, but trial matrices begin at each trial’s `start_time`. The per-trial bins are centered within `[start_time, stop_time)`.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes describe this as “align by absolute ophys time, then cut into trials,” with `trial start` recorded as the alignment event in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 30 Hz bins (`33.333... ms`). Native neural sampling is resampled onto that grid, so temporal rebinning/interpolation is applied.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

```python
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. The notes justify this as a way to standardize across sessions with mixed native rates while staying close to the behavioral and eye-tracking sampling rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the task stimulus-presentation interval tables, especially `image_name`, `omitted`, `trials_id`, `start_time`, `stop_time`, and `stimulus_block_name`.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    ...
    columns = [
        "start_time", "stop_time", "image_name", "omitted",
        "is_change", "trials_id", "stimulus_block_name", "active", "duration",
    ]
```

```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The notes justify using stimulus presentations rather than only trial-level image names because presentations explicitly represent flashes, omissions, and gray periods on the time axis.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI starts every time bin as `gray`, then overwrites bins covered by stimulus presentations with a globally coded image identity. Omitted presentations also map to `gray`. The mapping is global and deterministic, with `gray` forced to index 0.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes justify the explicit `gray` state by saying the trial contains inter-stimulus gray periods and occasional omissions, so those periods should be represented rather than left undefined.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is written onto the same 30 Hz trial grid used for neural data, using trial-specific presentation intervals to choose which bins receive which image code.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes justify this as using the same per-trial time grid for every modality so stimulus labels and neural activity are binwise aligned.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag, together with `trials_id`, `start_time`, and `stop_time`.

ii.
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted",
    "is_change", "trials_id", "stimulus_block_name", "active", "duration",
]
```

```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as using the stimulus table’s explicit change annotations instead of reconstructing change epochs only from `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The trace is initialized to 0 and set to 1 for every 30 Hz bin that falls within a presentation interval marked `is_change=True`. The code does not add an extra post-flash gray window.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes describe this as a per-bin trace built from stimulus-presentation rows, with the change flag attached to the change-image flash itself.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding step is applied beyond binary coding. The categories are directly `0 = no_change` and `1 = change`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
],
```

iii. No separate thresholding rationale was documented because the source variable was already binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is written onto the same trial `centers` grid as the neural data.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes justify this as keeping every decoded output on the same common time base as the neural matrix.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from NWB `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. The notes identify this as the processed Allen running-speed stream already present in the NWB.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to each trial’s 30 Hz bins. In pass 1, all trialwise running samples from retained sessions are concatenated to compute global 5-quantile edges; in pass 2, each trial is digitized using those edges.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The notes justify global equal-frequency binning so that running categories have consistent meaning across sessions and are approximately balanced overall.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global quantile bins, labeled `q1` through `q5`.

ii.
```python
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
```

iii. The notes explicitly say five equal-percentile bins are computed globally across the included dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 30 Hz bin centers used for neural data within each trial.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes justify this as a common per-trial time grid shared by all modalities.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from NWB eye-tracking timestamps plus `pupil_tracking/width` and `pupil_tracking/height`.

ii.
```python
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The notes justify this as using the raw eye-tracking ellipse measurements available in the NWB when a processed `pupil_width` table could not be loaded through the AllenSDK path.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI computes `2 * max(width, height)` as a diameter surrogate, fills NaNs by time interpolation, interpolates to the 30 Hz trial bins, and discretizes globally into five quantile bins.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes justify interpolation-through-NaNs as a way to preserve sessions with modest blink-related missingness while still requiring eye tracking at the session level.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global quantile bins, labeled `q1` through `q5`.

ii.
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. The notes explicitly say pupil bins are global five-way equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial centers as the neural data.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The notes justify this as a single shared time base for neural, behavior, and eye-tracking streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the NWB trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes justify this as using the Allen-processed trial outcomes directly from the NWB trial table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true outcome flag is mapped to indices `0..3` in fixed order and then repeated across every time bin in that trial, making the output time-varying but constant within trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The notes justify this as keeping a uniform `(n_output, T)` format for all decoder outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter handles several edge cases: string arrays are decoded manually; trial IDs are synthesized if absent; zero-length trials are skipped; sessions with missing eye tracking or missing task presentations are skipped; pupil NaNs are linearly filled in time; and sessions with fewer than 2 retained trials are dropped.

ii.
```python
if "id" not in trials:
    trials["id"] = np.arange(len(trials), dtype=np.int64)
...
if centers.size == 0:
    continue
...
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
```

iii. The notes justify skipping missing-eye-tracking sessions because pupil is a required output, and justify retaining all-zero neural trials because raw NWB spot checks showed they were genuine source-data trials rather than conversion bugs.

## 9-a. What are the most time-consuming steps of the code?

i. The main runtime costs are opening each NWB file twice, reading the full event-detection matrix from disk, and resampling neural data trial by trial during pass 2.

ii.
```python
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly say full conversion is I/O-heavy because each NWB event matrix must be read from disk, and they estimate pass 2 as the slower phase.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining Python loops are the string-decoding loop, the pass-1 and pass-2 per-trial loops, and the per-trial loop over stimulus-presentation rows. The largest neural interpolation step was already vectorized.

ii.
```python
for value in values:
    ...
```

```python
for trial in trials.itertuples(index=False):
    ...
```

```python
for row in trial_presentations.itertuples(index=False):
    ...
```

iii. The notes explicitly mention that event interpolation was vectorized with `searchsorted` and broadcasting; they do not document any further vectorization because the remaining loops were left as straightforward Python loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats trial-table loading/filtering, task-presentation loading, and file opening across two passes. Running/pupil interpolation is also done once in pass 1 to compute global quantiles and again in pass 2 to build outputs.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes describe the converter as a two-pass pipeline by design: pass 1 estimates global running/pupil bin edges and filters sessions, and pass 2 builds the final trial matrices.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes optional diagnostic plotting and reads/stores some metadata and trial columns that are not needed by downstream decoding, such as `session_type`, `experience_level`, `project_code`, `change_time`, and trial-level image-name fields used mainly for QC or documentation rather than final output construction.

ii.
```python
class SessionMeta:
    ...
    session_type: str
    experience_level: str
    project_code: str
```

```python
if show_processing and not plotted:
    ...
    make_processing_plot(...)
```

iii. The notes justify these extras as sanity-check and review aids; they are not required by the final pickle structure.
