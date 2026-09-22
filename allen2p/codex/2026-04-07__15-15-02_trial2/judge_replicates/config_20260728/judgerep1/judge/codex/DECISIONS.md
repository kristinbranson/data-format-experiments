# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads the local release metadata CSV (`ophys_experiment_table.csv`) and local NWB files directly with `h5py`. It restricts processing to experiment files present on disk, drops passive sessions up front, then drops active sessions that lack eye-tracking groups before any trial extraction.

ii. 
```python
def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    ...
    filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. In its notes, the AI justified direct NWB reads as a workaround for a `pynwb/hdmf` incompatibility while claiming it would mirror SDK field semantics. It also justified excluding passive and eye-tracking-missing sessions because the requested outputs include trial outcome and pupil-derived labels.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the metadata table, stored as strings.

ii. 
```python
SessionInfo(
    ...
    mouse_id=str(int(row.mouse_id)),
    ...
)
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The notes say the mouse identifier in the metadata is the subject identifier to preserve for `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (`ophys_experiment_id`) is treated as one decoder session. The AI does not group multiple experiments by shared `ophys_session_id`.

ii. 
```python
@dataclass(frozen=True)
class SessionInfo:
    ophys_experiment_id: int
    path: Path
    mouse_id: str
    targeted_structure: str
    session_type: str
    project_code: str
    passive: bool
...
"session_ophys_experiment_ids": [int(s.ophys_experiment_id) for s in sessions],
```

iii. The notes explicitly justify this by saying one NWB file corresponds to one experiment-specific neural recording and that experiments should therefore be treated as sessions even when multiple experiments share one behavior/ophys session.

## 1-d. How are the data split into trials?

i. Trials are read from `intervals/trials`. For every kept trial, the code uses the raw `start_time` and `stop_time` and creates a regular 30 Hz time grid spanning that interval.

ii. 
```python
trial_group = h5f["intervals"]["trials"]
trials = read_interval_table(
    trial_group,
    [..., "change_time", "start_time", "stop_time"],
)
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The notes say trial boundaries should come directly from the NWB trial table and should run from `start_time` to `stop_time` after go/catch filtering.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept when `(go or catch) and not aborted and not auto_rewarded`. Sessions with fewer than 2 kept trials are skipped. Separately, the dataset is pre-filtered to active sessions that have eye-tracking groups.

ii. 
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(
        f"[pass2] skipping session {session.ophys_experiment_id} because it has "
        f"{int(raw['keep_mask'].sum())} valid trials"
    )
    continue
```

iii. The notes justify excluding aborted and auto-rewarded trials from the operant task, excluding passive sessions, and requiring at least two trials for the decoder. The notes also mention dropping sessions missing eye tracking so pupil outputs can be defined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` and its paired timestamps, not from dF/F.

ii. 
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
...
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The notes repeatedly justify this by saying the strategy paper used detected calcium events and that using precomputed `events` better matches that paper than using dF/F.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from the raw ophys timestamps onto each trial’s 30 Hz grid, then transposed to `(n_neurons, n_timepoints)` for storage. Because sessions are per-experiment, no multi-plane stacking occurs.

ii. 
```python
def interpolate_matrix(
    source_t: np.ndarray, source_values: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify the 30 Hz interpolation by arguing that the target format requires one common bin size, eye tracking and behavior are naturally 30 Hz, and the paper interpolates event-aligned signals onto common 30 Hz timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code mostly accepts the released event traces as-is, but if the NWB cell table exposes a `valid_roi` mask and the event matrix still includes those invalid ROIs, it filters columns by that mask.

ii. 
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
...
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say the released data already reflect Allen ROI QC and that the converter should keep valid ROIs without adding new ad hoc neural filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial is aligned to trial start: the trial grid starts at `start_time`, ends at `stop_time`, and neural activity is interpolated onto that grid in absolute experiment time.

ii. 
```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
```

iii. The notes say trial start is the alignment event and that all streams should share the same absolute trial grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz bin size (`1/30 s`, `33.33 ms`) for all sessions. Yes: the neural data are resampled by interpolation onto this common grid.

ii. 
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. The notes explicitly defend a common 30 Hz grid as the “most defensible common grid” because eye tracking and behavior are 30 Hz and because the paper used common 30 Hz event-aligned traces.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the active stimulus-presentation table, specifically stimulus `start_time`, `stop_time`, `image_name`, and `omitted`. It is not derived from trial-level `initial_image_name` / `change_image_name`.

ii. 
```python
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
```

iii. The notes justify this as using the actual flashed-image intervals from the active task block, including explicit gray gaps and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds a global image vocabulary from all non-omitted stimulus image names, prepends a `gray` category, then labels each trial time bin by the active stimulus interval containing that time; bins outside a non-omitted image interval get `gray`.

ii. 
```python
image_values = ["gray"] + sorted(image_names)
...
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. The notes say `gray` must be an explicit class because the task includes 500 ms gray periods and omission periods with no image shown.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on exactly the same 30 Hz per-trial grid as the neural data, by querying stimulus intervals against `grid`.

ii. 
```python
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify this as a shared common trial grid for all streams.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` fields, rather than from trial `change_time`.

ii. 
```python
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The notes justify this as using the changed-image presentation interval itself, not a one-bin impulse, because that yields a less degenerate decoder target while staying close to task structure.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each time bin, the code finds the current stimulus interval and emits `1` if that interval is a non-omitted `is_change` presentation; otherwise it emits `0`.

ii. 
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
assign = np.flatnonzero(valid)[in_interval]
codes[assign] = changed.astype(np.int64)
```

iii. The notes say this replaced an earlier single-bin `change_time` marker because the latter was too sparse for the decoder.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary categorical output with categories `0 = no_change` and `1 = change`. There is no additional threshold beyond the boolean `is_change` interval.

ii. 
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    outcome_values,
],
...
codes[assign] = changed.astype(np.int64)
```

iii. The notes frame this as a categorical time-varying target defined directly by change-presentation intervals.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is computed on the same 30 Hz trial grid as the neural data and image identity.

ii. 
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify a common trial grid for all streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps in the NWB file.

ii. 
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes say this mirrors the SDK running-speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The continuous running trace is interpolated onto each trial’s 30 Hz grid, global quintile edges are computed across all included trials, and each value is digitized into one of five bins.

ii. 
```python
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
...
running_edges = robust_quintile_edges(running_all)
...
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The notes justify global quintiles as shared class definitions across sessions and say the running signal should be interpolated onto the common 30 Hz grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five percentile bins using global 20th, 40th, 60th, and 80th percentile edges.

ii. 
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    ...
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The notes explicitly say running speed should use global quintile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz per-trial grid as the neural data before binning.

ii. 
```python
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify a shared absolute trial grid for all streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The implemented code derives pupil diameter from `pupil_tracking/width`, `pupil_tracking/height`, and eye-tracking timestamps, using `max(width, height)` as the diameter proxy.

ii. 
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64
)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The notes justify pupil processing in terms of blink-masked eye-tracking data from the SDK, but the implemented code actually uses width/height directly and does not read a blink flag.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code takes `max(width, height)`, drops only non-finite samples, linearly interpolates the remaining values onto each trial’s 30 Hz grid, computes global quintile edges, and digitizes each sample into five bins.

ii. 
```python
def interpolate_pupil(
    pupil_t: np.ndarray, pupil_diameter: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_values.append(
    interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes claim this should reflect blink-masked pupil interpolation and global quintile binning, but only the interpolation and quintile-binning parts appear in the code.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five global percentile bins using the same 20/40/60/80 percentile scheme as running speed.

ii. 
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
"output_values": [
    list(image_values),
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    outcome_values,
],
```

iii. The notes explicitly plan global quintile bins for pupil diameter.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz per-trial grid as the neural data before discretization.

ii. 
```python
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify this with the same common-grid alignment argument used for neural data and running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
```

iii. The notes say these are the canonical go/catch outcome labels and should be used directly.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps those outcomes to integer labels in the fixed order `hit`, `miss`, `false_alarm`, `correct_reject`, then broadcasts the trial label across every time bin in the trial.

ii. 
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The notes justify this as turning a static per-trial variable into a time-varying row so every output shares the same trial matrix format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye-tracking sessions are excluded before conversion. For pupil interpolation, only finite samples are used. Sessions with fewer than two valid trials are skipped. There is no per-session `try/except` recovery in the final converter; failures would stop execution.

ii. 
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if int(raw["keep_mask"].sum()) < 2:
    ...
    continue
```

iii. The notes justify excluding sessions with missing pupil streams so the required outputs remain well-defined and describe the all-zero neural warnings as genuine data sparsity rather than conversion error.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies neural interpolation onto the common trial grid as the dominant expected cost in the final implementation, with full-session HDF5 reads as additional overhead.

ii. 
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
raw = read_session_raw(session)
```

iii. The notes explicitly say “Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid.”

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes interpolation across neurons within a trial, but it still loops over sessions and trials in Python. The AI’s notes focus on that per-trial resampling loop as the main remaining cost center.

ii. 
```python
for sess_num, session in enumerate(sessions, start=1):
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
        neural_trial = interpolate_matrix(
            raw["ophys_timestamps"], raw["events"], grid
        ).T.astype(np.float32)
```

iii. The notes say a speedup already added was that “trial-level interpolation is vectorized over neurons within each trial,” implying the remaining Python loops were left in place.

## 9-c. What processing does the code repeat multiple times?

i. The converter rereads every session twice: once in `collect_global_statistics(..., load_events=False)` to compute global image/running/pupil statistics, and again in `convert_sessions(...)` to load the full session and build trial outputs. Running and pupil interpolation are therefore repeated across both passes.

ii. 
```python
raw = read_session_raw(session, load_events=False)
...
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
pupil_values.append(
    interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
)
...
raw = read_session_raw(session)
```

iii. The notes justify this as a deliberate two-pass design to keep memory bounded while still computing global discretization edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints per-session `native_dt` and valid-trial counts that are not stored in the output, loads several stimulus columns that are not used downstream (`trials_id`, `active`, `flashes_since_change`), and includes optional plotting/debug machinery that is not part of the converted dataset.

ii. 
```python
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
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(
    sessions
)
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
```

iii. The notes frame most of this as auditability and debugging support rather than as data needed by downstream decoding.
