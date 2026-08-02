# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the metadata table `ophys_experiment_table.csv`, maps each `ophys_experiment_id` to one NWB file under `behavior_ophys_experiments/`, excludes passive sessions and sessions missing eye-tracking groups, and then reads each selected session directly with `h5py`. Within each NWB it loads the trials table, a stimulus-presentation table, ophys event timestamps and data, running speed, and pupil-tracking streams.

ii. ```python
def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
...
def read_session_raw(session: SessionInfo, load_events: bool = True) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
...
        stim_group = choose_task_presentation_group(h5f)
...
        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
        )
        events = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
        )
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
        pupil_width = np.asarray(
            h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
        )
```

iii. In `CONVERSION_NOTES.md`, the agent says it followed the SDK field definitions but used direct HDF5 reads because `BehaviorOphysExperiment.from_nwb_path` was blocked by a local `pynwb/hdmf` incompatibility. The notes justify using one NWB experiment file as one decoder session and treating the local disk contents as a subset of the release.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from the experiment metadata table. The converter builds a unique `subjects` list and a per-session `subject_idx` by first occurrence order.

ii. ```python
SessionInfo(
    ophys_experiment_id=int(row.ophys_experiment_id),
    path=file_map[int(row.ophys_experiment_id)],
    mouse_id=str(int(row.mouse_id)),
...
subject_to_idx: Dict[str, int] = {}
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
...
data["subject_idx"].append(subject_idx)
```

iii. The notes map NWB/metadata `mouse_id` to `subjects`/`subject_idx` and explicitly state that mouse identifiers should be used as the subject labels.

## 1-c. How are the data split into sessions?

i. Sessions are split at the NWB file / `ophys_experiment_id` level. The code treats each experiment file as one session, filters to active sessions in the metadata, and later skips any selected session with fewer than two valid trials.

ii. ```python
file_map = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
}
...
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    if int(raw["keep_mask"].sum()) < 2:
        ... continue
```

iii. `CONVERSION_NOTES.md` calls out the distinction between `behavior_session`, `ophys_session`, and `ophys_experiment`, and justifies using each NWB experiment file as one decoder session because the neural traces are experiment-specific.

## 1-d. How are the data split into trials?

i. Trials are split using the NWB `intervals/trials` table. For each kept trial, the converter uses `start_time` and `stop_time` to define the full trial window and builds a regular time grid over that interval.

ii. ```python
trials = read_interval_table(
    trial_group,
    [
        "go", "catch", "aborted", "auto_rewarded",
        "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "start_time", "stop_time",
    ],
)
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The notes say trial boundaries come directly from the NWB `intervals/trials` table and that trial windows run from `start_time` to `stop_time`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by keeping only `go` or `catch` trials and excluding `aborted` and `auto_rewarded` trials. Sessions with fewer than two remaining trials are skipped.

ii. ```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    ... continue
...
if len(session_neural) < 2:
    ... continue
```

iii. The notes justify this as matching the task instructions and the SDK/paper trial taxonomy: include standard go/catch trials, exclude aborted resets and free-reward / auto-reward trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the precomputed calcium event matrix `processing/ophys/event_detection/data` and its timestamps `processing/ophys/event_detection/timestamps`.

ii. ```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The notes explicitly say the neural signal should be precomputed `events`, not recomputed dF/F, because the reference paper’s analyses used detected calcium events.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from its native ophys timestamps onto each trial’s common 30 Hz grid, then transposed to `(n_neurons, n_timepoints)` and stored as `float32`.

ii. ```python
def interpolate_matrix(source_t, source_values, query_t) -> np.ndarray:
    right = np.searchsorted(source_t, query_t, side="left")
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify this by saying the paper interpolates event-triggered responses onto common 30 Hz timestamps and the target format requires one shared bin size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neuron QC is filtering to `valid_roi` if the event matrix still matches the full ROI table width. No extra neuron/session filtering is applied beyond that and the session-level minimum-trial rule.

ii. ```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say the SDK already filters invalid ROIs and that the converter should not add extra ad hoc neuron filtering beyond the reference QC reflected in the released files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned by absolute ophys timestamps onto a per-trial 30 Hz grid spanning each trial’s `start_time` to `stop_time`. The metadata labels the alignment event as `"trial start"`, even though the notes describe the grid as being derived from ophys timestamps.

ii. ```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
"temporal_alignment_event": "trial start",
```

iii. The notes say all streams are aligned in absolute experiment time and resampled to a common 30 Hz grid from raw trial `start_time` / `stop_time`, and earlier notes describe that as a grid “derived from ophys timestamps.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 30 Hz bin size (`1/30` s, `33.333...` ms). Yes: the code linearly resamples native ophys, running, and pupil streams onto that common grid.

ii. ```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
...
grid = session_grid(start, stop)
running_cont = interpolate_vector(...)
pupil_cont = interpolate_pupil(...)
neural_trial = interpolate_matrix(...)
```

iii. The notes justify 30 Hz as the only common rate compatible with the task format and with the paper’s use of common 30 Hz timestamps.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `image_identity` is derived from the task stimulus-presentation interval table, specifically `start_time`, `stop_time`, `image_name`, and `omitted`.

ii. ```python
stim = read_interval_table(
    stim_group,
    [
        "start_time",
        "stop_time",
        "image_name",
        "is_change",
        "omitted",
        "trials_id",
        "active",
        "flashes_since_change",
    ],
)
...
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]
```

iii. The notes map `stimulus_presentations` image intervals to the time-varying image identity output and state that the active change-detection block is the intended source.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The converter builds a global vocabulary of image names across sessions, prepends an explicit `gray` class, and then uses a piecewise-constant lookup on each trial grid: bins inside a non-omitted image presentation get that image code, otherwise they stay `gray`.

ii. ```python
image_names.update(
    str(x)
    for x, omitted in zip(stim["image_name"], stim["omitted"])
    if (not omitted) and str(x) not in ("", "None", "nan")
)
image_values = ["gray"] + sorted(image_names)
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes justify the explicit `gray` class because the task includes gray inter-stimulus periods and omission flashes extend gray rather than showing an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. `image_identity` is sampled on the same per-trial 30 Hz grid used for neural interpolation, so each label row is time-aligned to the neural bins.

ii. ```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. The notes say all streams are aligned in absolute experiment time and resampled to one common grid per trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the stimulus-presentation table fields `is_change`, `start_time`, `stop_time`, and `omitted`.

ii. ```python
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The notes say the converter uses the raw stimulus table’s `is_change` intervals rather than only the trial table’s `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The converter does a piecewise-constant interval lookup on the trial grid and marks bins as `1` during non-omitted changed-image presentation windows. Earlier trajectory notes show the agent changed this from a one-bin impulse at `change_time` to a full presentation-window label.

ii. ```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
if np.any(in_interval):
    sub_idx = idx_valid[in_interval]
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    assign = np.flatnonzero(valid)[in_interval]
    codes[assign] = changed.astype(np.int64)
```

iii. In the trajectory, the agent says the original one-bin impulse was “too sparse” and replaced it with a presentation-window label that it viewed as still faithful to “right after a change.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: `0` for `no_change`, `1` for `change`. The underlying raw boolean `is_change` is cast directly to integer codes.

ii. ```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
...
codes[assign] = changed.astype(np.int64)
```

iii. The notes describe `image_change` as a binary time-varying label from stimulus change intervals.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `image_change` is computed on the same per-trial 30 Hz grid as the neural matrix, so it is bin-aligned with the interpolated neural data.

ii. ```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(..., grid).T.astype(np.float32)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The notes say output streams and neural streams share the same trial grids and absolute timestamp frame.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii. ```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes say this uses the SDK-style filtered running-speed stream rather than recomputing running from raw encoder voltages.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The converter linearly interpolates the released running-speed timeseries onto each trial’s 30 Hz grid and uses the interpolated continuous values only as an intermediate before binning.

ii. ```python
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
```

iii. The notes justify using the already processed running speed because that matches the Allen reference processing.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The converter pools interpolated running values across all kept trials in all included sessions, computes robust global quintile edges at the 20/40/60/80 percentiles, and digitizes each trial’s running values into five bins.

ii. ```python
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
...
running_edges = robust_quintile_edges(running_all)
...
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The notes explicitly say continuous outputs are discretized globally, not per session, so categories are shared across the dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same per-trial 30 Hz grid as the neural data and then digitized bin-by-bin, preserving alignment.

ii. ```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(..., grid).T.astype(np.float32)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The notes say all streams are resampled into one common per-trial grid in absolute experiment time.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width`, `acquisition/EyeTracking/pupil_tracking/height`, and `acquisition/EyeTracking/eye_tracking/timestamps`.

ii. ```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64
)
```

iii. The notes describe the source as the eye-tracking stream and say the pupil output comes from pupil width/height plus blink-related invalid-frame handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The implemented code defines pupil diameter as `max(width, height)`, treats finite samples as valid, and linearly interpolates them onto the per-trial 30 Hz grid. The notes claim blink-masked values were used, but the code does not read or apply any blink flag or outlier mask.

ii. ```python
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
...
def interpolate_pupil(pupil_t, pupil_diameter, query_t) -> np.ndarray:
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
```

iii. The notes justify the intended approach as “blink-masked pupil diameter” with interpolation across valid samples, but the actual script only implements the `max(width, height)` plus finite-value interpolation part.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. As with running speed, pupil values are pooled across all kept trials in all included sessions, global 20/40/60/80 percentile edges are computed, and each trial’s interpolated pupil values are digitized into five bins.

ii. ```python
pupil_values.append(
    interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
)
...
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes say pupil diameter should be discretized globally into quintiles so the categories are shared across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same per-trial 30 Hz grid used for the neural data, then binned on that grid.

ii. ```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(..., grid).T.astype(np.float32)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes say pupil, running, stimulus, and neural streams all share a common aligned grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial outcome flags in `intervals/trials`: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
trials = read_interval_table(
    trial_group,
    [
        "go", "catch", "aborted", "auto_rewarded",
        "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "start_time", "stop_time",
    ],
)
...
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. The notes map the SDK trial outcome taxonomy directly to the categorical decoder output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The converter maps the first true outcome flag to an integer class and then broadcasts that static class across all bins in the trial.

ii. ```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The notes say static trial outcome is intentionally repeated across time to fit the requested output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code excludes active sessions that entirely lack eye-tracking groups, skips sessions with fewer than two valid trials, interpolates pupil values only over finite samples, nudges duplicate percentile edges upward by `1e-6`, raises errors for missing HDF5 columns, and tolerates sparse all-zero neural trials without removing them.

ii. ```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
...
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
...
if int(raw["keep_mask"].sum()) < 2:
    ... continue
```

iii. The notes say the only sessions excluded for missing data were the three active sessions without eye tracking, and they argue that all-zero neural trials reflect genuine event sparsity rather than conversion errors.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the two full passes over the NWB sessions and, within conversion, the per-trial interpolation of the neural event matrix onto the 30 Hz grid. Global statistics collection also repeatedly interpolates running and pupil data for every kept trial.

ii. ```python
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
        running_values.append(interpolate_vector(..., grid))
        pupil_values.append(interpolate_pupil(..., grid))
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
        neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. The notes explicitly identify neural interpolation as the dominant cost and describe the conversion as a two-pass pipeline.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over sessions, over trials within each session, and over per-bin assignments inside `stimulus_identity_codes`. The neural interpolation itself is vectorized across neurons, but the converter still does one interpolation call per trial and separately recomputes stimulus lookup for each output row.

ii. ```python
for idx, session in enumerate(sessions, start=1):
...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes mention trial-level interpolation is already vectorized over neurons, implying the remaining inefficiency is the repeated Python-level per-trial work.

## 9-c. What processing does the code repeat multiple times?

i. The script reads each session twice, rebuilds trial grids twice, and re-interpolates running and pupil data in both the global-statistics pass and the conversion pass. It also searches stimulus intervals separately for image identity and image change on the same grids.

ii. ```python
raw = read_session_raw(session, load_events=False)
...
running_values.append(interpolate_vector(..., grid))
pupil_values.append(interpolate_pupil(..., grid))
...
raw = read_session_raw(session)
...
running_cont = interpolate_vector(...)
pupil_cont = interpolate_pupil(...)
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The notes describe the design as deliberately two-pass to compute global bin edges first, so this repeated work is intentional rather than accidental.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and logs `native_dt_by_session`, `valid_trial_counts`, and processing plots for optional inspection, but those products are not used by downstream decoder training. It also constructs continuous running/pupil traces only to discard them after digitization, and builds sample plotting payloads that are only for debugging.

ii. ```python
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
...
running_cont = interpolate_vector(...)
pupil_cont = interpolate_pupil(...)
running_bins = digitize_with_edges(running_cont, running_edges)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
if show_processing and (sess_num <= 2) and not plotted:
    make_processing_plot(...)
```

iii. The notes explicitly frame these as sanity-checking, profiling, or visualization steps rather than part of the final decoder representation.
