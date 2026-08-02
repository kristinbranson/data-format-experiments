# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads local NWB files directly with `h5py` rather than using the Allen SDK cache. It first reads `ophys_experiment_table.csv`, keeps rows whose experiment ids have local NWB files, drops passive sessions, then opens each NWB file and reads trial, stimulus, neural, running, and eye-tracking arrays from HDF5 groups.

ii. 
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
...
with h5py.File(session.path, "r") as h5f:
    trial_group = h5f["intervals"]["trials"]
    ...
    stim_group = choose_task_presentation_group(h5f)
    ...
    events = np.asarray(
        h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
    )
```

iii. In `CONVERSION_NOTES.md`, the AI says it switched to direct HDF5 loading because the local `pynwb/hdmf` stack could not instantiate these NWB files through the Allen SDK, and that direct reads would still mirror SDK field semantics.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the metadata table, stored as strings and indexed globally.

ii.
```python
sessions = [
    SessionInfo(
        ...
        mouse_id=str(int(row.mouse_id)),
        ...
    )
    for row in exp_table.itertuples(index=False)
]
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The notes state that mouse identifiers from metadata are the natural subject split and should be carried through unchanged.

## 1-c. How are the data split into sessions?

i. Each kept `ophys_experiment_id` NWB file is treated as one session. The AI does not reconstruct multi-plane `ophys_session_id` sessions from multiple experiments; it also excludes passive sessions and active sessions lacking eye-tracking groups.

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
exp_table = exp_table[~exp_table["passive"]].copy()
...
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The notes explicitly justify this as a local-data decision: one NWB file was treated as one decoder session, and passive sessions were excluded because the requested outputs require active-task trial outcomes.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each kept trial, the AI uses the trial `start_time` and `stop_time` to build a variable-length trial on a synthetic 30 Hz grid.

ii.
```python
trial_group = h5f["intervals"]["trials"]
trials = read_interval_table(
    trial_group,
    ["go", "catch", "aborted", "auto_rewarded", ...,
     "change_time", "start_time", "stop_time"],
)
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The notes say trial boundaries should come directly from the NWB trial table and that `start_time` to `stop_time` best preserves the full go/catch trial structure.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are `go` or `catch` and are not `aborted` or `auto_rewarded`. The AI also drops sessions with fewer than two kept trials.

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

iii. The notes justify this as matching the task statement to include go and catch trials but exclude aborted and auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from precomputed calcium `event_detection/data` and `event_detection/timestamps` in each NWB file, not from dF/F.

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

iii. The notes explicitly say the strategy paper used detected calcium events, so the AI chose events instead of dF/F to better match the paper’s analyses.

## 2-b. How is the `neural` data processed?

i. Neural activity is linearly interpolated from native ophys timestamps onto a common 30 Hz trial grid and then transposed to `(n_neurons, n_timepoints)` for each trial.

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

iii. The notes justify the common 30 Hz grid as a way to unify mixed native ophys frame rates and align neural data with running and pupil streams that are naturally around 30 Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI conditionally filters neural columns by `valid_roi` from the NWB cell specimen table when the table dimensions line up, so only valid ROIs are kept.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
...
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say this mirrors the Allen SDK’s `exclude_invalid_rois=True` behavior while bypassing SDK object loading.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start. The AI builds each trial grid from `start_time` and treats time 0 as the start of the trial.

ii.
```python
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
```

iii. The notes say trial segmentation should use the trial table’s `start_time`/`stop_time` bounds, so all outputs can be defined over the full trial relative to start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. 33.33 ms bins, and all trial-level streams are resampled/interpolated onto that grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. The notes explicitly defend 30 Hz as the common time base because local sessions mix ~31 Hz and ~11 Hz ophys, while running and eye tracking are near 30 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, primarily `start_time`, `stop_time`, `image_name`, and `omitted`, not from the trial table’s initial/change image names.

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
def stimulus_identity_codes(
    stimulus: Dict[str, np.ndarray], query_t: np.ndarray, image_to_code: Dict[str, int]
) -> np.ndarray:
```

iii. The notes say the stimulus table gives the actual flashed-image intervals and allows an explicit gray class during gray screens and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI creates a global image vocabulary `["gray"] + sorted(image_names)`. It then uses a search over stimulus intervals to assign each 30 Hz bin either gray or the currently presented image code.

ii.
```python
image_values = ["gray"] + sorted(image_names)
...
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes justify the explicit gray category because the task includes 500 ms gray periods and omission periods where no image is shown.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same 30 Hz `grid` used for the interpolated neural trial, so labels and neural activity share identical trial bins.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. The notes say all modalities were intentionally mapped to one common grid to remove cross-stream timing mismatches.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` fields, not from trial `change_time`.

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
        ...
    ],
)
...
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
```

iii. The notes say using the stimulus table marks the actual changed-image presentation interval rather than a point event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 30 Hz bin, the AI finds the current stimulus interval and sets the label to 1 when that interval has `is_change=True` and is not omitted; otherwise it sets 0.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
assign = np.flatnonzero(valid)[in_interval]
codes[assign] = changed.astype(np.int64)
```

iii. In the notes, the AI says this yields a less degenerate decoder target than a single-bin impulse while staying tied to task structure.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary. The AI uses category `0` for `no_change` and `1` for `change`, with no additional thresholding.

ii.
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    ...
]
...
codes = np.zeros(query_t.shape, dtype=np.int64)
codes[assign] = changed.astype(np.int64)
```

iii. The code and notes both treat image change as a native binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same per-trial 30 Hz `grid` as the neural data, so it is bin-aligned to the interpolated event traces.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The notes justify one shared time base for all streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` with its matching `timestamps`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes say this is the released SDK-equivalent running-speed signal rather than reconstructing it from raw wheel voltages.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz grid, then discretized into five global bins using quintile edges computed across all included sessions.

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

iii. The notes justify global quintile bins as shared category definitions across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses four global quintile cut points at the 20th, 40th, 60th, and 80th percentiles, with a small monotonicity fix if adjacent percentiles tie. `np.digitize` maps values to bins `0` through `4`.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The AI did not separately justify the epsilon tie-break, but it is consistent with the goal of always producing five ordered bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 30 Hz trial grid as the neural data.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
```

iii. The notes describe the common grid as the main alignment strategy across modalities.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/width`, `pupil_tracking/height`, and eye-tracking timestamps. The AI defines diameter as `max(width, height)`.

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

iii. The notes justify this as a direct diameter proxy available in the released eye-tracking arrays.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI interpolates finite pupil-diameter samples onto the 30 Hz trial grid, then discretizes them into five global quintile bins.

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

iii. The notes say blink-related invalid samples should be masked by invalid values and then interpolated over when building the common grid.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter uses the same quintile-binning scheme as running speed: four global cut points and five categories `0` through `4`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
[f"bin_{i}" for i in range(5)]
```

iii. The notes group running and pupil together as continuous variables requiring global five-way discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid used for neural interpolation.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
```

iii. The notes describe common-grid resampling as the alignment method for all behavioral outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial flags `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
```

iii. The notes state these are the SDK trial-outcome labels and should be mapped directly.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps each trial outcome to an integer code and broadcasts that code across every time bin in the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The notes explicitly say even static outputs should be represented as time-varying by repeating them across the trial grid.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness by excluding sessions without required eye-tracking groups, filtering to finite pupil samples before interpolation, optionally filtering invalid ROIs via `valid_roi`, skipping sessions with fewer than two valid trials, and erroring if there are no valid pupil samples.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    ...
    events = events[:, valid_roi]
...
if int(raw["keep_mask"].sum()) < 2:
    ...
    continue
```

iii. The notes justify these choices as conservative data-curation steps needed to keep all requested outputs defined and auditable.

## 9-a. What are the most time-consuming steps of the code?

i. The code is structured as a two-pass pipeline. The most expensive work is repeatedly opening NWB files and, in pass 2, interpolating the full neural event matrix for every kept trial.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
    ...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    neural_trial = interpolate_matrix(
        raw["ophys_timestamps"], raw["events"], grid
    ).T.astype(np.float32)
```

iii. The notes explicitly say neural interpolation is the dominant expected cost and that direct HDF5 loading was chosen partly to reduce SDK object-construction overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over sessions in both passes, over all valid trials within each session, and over assigned stimulus intervals in `stimulus_identity_codes`. The matrix interpolation itself is already vectorized across neurons, but trial-level processing is not.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes say the major speedup already applied was vectorizing neural interpolation over neurons and using a two-pass design, implying the remaining loops were left in place for simplicity.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats session reads across two passes, recomputes each trial’s time grid in both passes, and interpolates running and pupil once in pass 1 for global-statistics collection and again in pass 2 for final outputs.

ii.
```python
raw = read_session_raw(session, load_events=False)
...
grid = session_grid(start, stop)
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
pupil_values.append(
    interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
)
...
raw = read_session_raw(session)
...
grid = session_grid(start, stop)
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
```

iii. The notes acknowledge the two-pass design explicitly and present it as a tradeoff to compute global bins without storing all neural trials in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and stores several fields used only for setup, diagnostics, or optional plots: `active`, `flashes_since_change`, `project_code`, `session_type`, `native_dt`, `valid_trial_counts`, and plotting data. It also computes summary statistics printed to stdout but not used in the saved dataset.

ii.
```python
stim = read_interval_table(
    stim_group,
    [
        ...,
        "active",
        "flashes_since_change",
    ],
)
...
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(
    sessions
)
print(
    "[setup] native ophys dt range:",
    min(native_dt.values()),
    max(native_dt.values()),
)
...
if show_processing and (sess_num <= 2) and not plotted:
    make_processing_plot(...)
```

iii. The notes say these extra computations were added for sanity checks, bottleneck detection, and optional visualization rather than for the final decoder representation itself.
