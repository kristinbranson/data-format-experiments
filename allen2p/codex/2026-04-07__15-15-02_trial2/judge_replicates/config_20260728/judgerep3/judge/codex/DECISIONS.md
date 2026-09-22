# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It discovers local NWB files from `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, matches them to rows in `ophys_experiment_table.csv`, drops passive sessions, drops sessions missing eye-tracking groups, and then reads each session directly with `h5py`.

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
def read_session_raw(
    session: SessionInfo,
    load_events: bool = True,
) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
```

iii. In `CONVERSION_NOTES.md`, the AI justified direct HDF5 loading as a workaround for a `pynwb/hdmf` incompatibility with the local NWB files, and treated the local disk contents as a subset of the release rather than fetching additional experiments.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by unique `mouse_id` values from `ophys_experiment_table.csv`, stored as strings.

ii. 
```python
SessionInfo(
    ophys_experiment_id=int(row.ophys_experiment_id),
    path=file_map[int(row.ophys_experiment_id)],
    mouse_id=str(int(row.mouse_id)),
    targeted_structure=str(row.targeted_structure),
    session_type=str(row.session_type),
    project_code=str(row.project_code),
    passive=bool(row.passive),
)
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The notes treat `mouse_id` as the canonical subject identifier exposed by the release metadata, with no extra subject-level filtering beyond session inclusion/exclusion.

## 1-c. How are the data split into sessions?

i. Each local NWB experiment file is treated as one session. The AI does not group multiple experiments by `ophys_session_id`; `ophys_experiment_id` is effectively the session key.

ii. 
```python
@dataclass(frozen=True)
class SessionInfo:
    ophys_experiment_id: int
    path: Path
    mouse_id: str
...
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
```

iii. The notes explicitly say the agent resolved the experiment-vs-session ambiguity by treating each NWB experiment file as one decoder session because the neural traces are experiment-specific.

## 1-d. How are the data split into trials?

i. Trials are split using the NWB `intervals/trials` table. For each kept trial, the AI uses the trial `start_time` and `stop_time` and builds a regular 30 Hz grid spanning that interval.

ii. 
```python
trial_group = h5f["intervals"]["trials"]
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

iii. The notes say the trial boundaries should come directly from the NWB `intervals/trials` table and should cover the full `start_time` to `stop_time` window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept when they are `go` or `catch` and not `aborted` or `auto_rewarded`. Sessions are also dropped if they are passive, if required eye-tracking groups are missing, or if fewer than two valid trials remain.

ii. 
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
...
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(
        f"[pass2] skipping session {session.ophys_experiment_id} because it has "
        f"{int(raw['keep_mask'].sum())} valid trials"
    )
    continue
```

iii. The notes justify excluding passive sessions because the decoder needs trial outcomes from the active task, and justify the go/catch vs aborted/auto-rewarded filter as the common interpretation across the paper, SDK, and task instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/data` and `processing/ophys/event_detection/timestamps` in each NWB file, not from `dff_traces`.

ii. 
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
...
if load_events:
    events = np.asarray(
        h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
    )
```

iii. The notes repeatedly justify this as following the strategy paper more closely, because that paper discusses analyses based on detected calcium events rather than dF/F.

## 2-b. How is the `neural` data processed?

i. The event matrix is optionally filtered to valid ROIs, then linearly interpolated from native ophys timestamps onto a common 30 Hz trial grid and transposed to `(n_neurons, n_timepoints)` for each trial.

ii. 
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify the 30 Hz common grid by saying the target format needs one shared bin size, the behavioral streams are naturally 30 Hz, and the paper used interpolation onto a common behavioral time base for event-triggered analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies only the `valid_roi` mask from the NWB cell table when needed; it does not add extra neuron-level QC beyond that.

ii. 
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say this is meant to mirror the SDK’s built-in ROI filtering and avoid adding ad hoc curation beyond the reference pipeline’s own QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start in the saved metadata, but neural activity is sampled from the event timestamps over the full `start_time` to `stop_time` trial window using a 30 Hz grid.

ii. 
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. The notes frame this as using ophys timestamps as the source clock while keeping full trial windows so all time-varying outputs can be aligned on the same grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz time base (`33.33 ms` bins). Yes: all streams, including neural events, are resampled/interpolated to that common grid.

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

iii. The notes explicitly defend the common 30 Hz grid as a deliberate cross-session resampling choice.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the chosen active stimulus-presentation table, specifically `start_time`, `stop_time`, `image_name`, and `omitted`, rather than from the trial table’s `initial_image_name` and `change_image_name`.

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
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
image_names = stimulus["image_name"]
omitted = stimulus["omitted"]
```

iii. The notes justify this as a way to capture the real flashed-image stream, including gray periods and omissions, instead of forcing a simple pre-change/post-change trial-table description.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a global vocabulary of image names from all non-omitted stimulus presentations, prepends a `"gray"` class, then assigns each 30 Hz trial bin to the active stimulus interval; bins outside a flashed-image interval or in omitted intervals get the `"gray"` code.

ii. 
```python
image_names.update(
    str(x)
    for x, omitted in zip(stim["image_name"], stim["omitted"])
    if (not omitted) and str(x) not in ("", "None", "nan")
)
...
image_values = ["gray"] + sorted(image_names)
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. In the notes, the explicit `"gray"` class is justified because the trials include gray-screen intervals and omissions that otherwise would have no faithful image-identity label.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same 30 Hz per-trial grid used for neural interpolation, so the categorical output and neural data share exactly the same time bins.

ii. 
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. The notes describe this as putting all streams on one shared grid derived from the source timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` columns.

ii. 
```python
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
```

iii. The notes say this was chosen so the label marks the changed-image presentation window itself rather than a single instant tied only to `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 30 Hz trial bin, the AI locates the active stimulus interval and sets the label to `1` if that interval is a non-omitted change presentation; otherwise it is `0`.

ii. 
```python
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
...
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    assign = np.flatnonzero(valid)[in_interval]
    codes[assign] = changed.astype(np.int64)
```

iii. The notes explicitly say the agent changed this from a sparser point-event style target to a full changed-image interval because the impulse version was too sparse.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value. It is directly represented as a binary categorical output with `0 = no_change` and `1 = change`.

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
codes = np.zeros(query_t.shape, dtype=np.int64)
```

iii. The justification is implicit in the task design and the notes: `image_change` is treated as a binary event label rather than a discretized continuous quantity.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `image_change` is computed on the same 30 Hz trial grid used for neural data.

ii. 
```python
grid = session_grid(start, stop)
...
image_change = stimulus_change_codes(raw["stimulus"], grid)
...
output_trial = np.vstack(
    [
        image_codes.astype(np.int64),
        image_change,
        running_bins,
        pupil_bins,
        trial_outcome,
    ]
)
```

iii. The notes describe this as joint alignment of all streams on one common grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps in each NWB file.

ii. 
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes cite the Allen SDK running-speed processing as the semantic reference for this stream, even though the implementation reads the NWB datasets directly.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed to each trial’s 30 Hz grid, pools those values across all kept trials and sessions to compute global quintile edges, then digitizes each trial into five bins.

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

iii. The notes justify global quintile binning so category definitions are shared across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five percentile-style bins using global edges at the 20th, 40th, 60th, and 80th percentiles, then labeled `bin_0` through `bin_4`.

ii. 
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles

def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The notes describe this as global quintile binning, with a small monotonicity fix so repeated percentile values do not create invalid edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same 30 Hz grid as the neural data, trial by trial.

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

iii. The notes justify this by using the synchronized timestamps from the NWB file and a shared target grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width`, `.../height`, and `eye_tracking/timestamps`; the code defines diameter as `max(width, height)`.

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

iii. The notes say this was intended to mirror the eye-tracking processing while using a simple diameter proxy from the raw ellipse fits; the notes also mention blink masking, although the final code does not use a blink flag explicitly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code interpolates the `max(width, height)` signal onto each trial’s 30 Hz grid using only finite samples, then computes global quintile edges and digitizes each trial into five bins.

ii. 
```python
def interpolate_pupil(
    pupil_t: np.ndarray, pupil_diameter: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    if valid.sum() == 1:
        return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes justify global discretization and interpolation similarly to running speed, but the written justification in the notes is stricter than the final implementation because it describes blink masking that the script does not actually perform.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global quintile bins using the same 20/40/60/80 percentile scheme as running speed.

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

iii. The notes describe this as keeping the continuous outputs in shared, balanced categorical bins across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto the same 30 Hz per-trial grid used for neural activity.

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

iii. The notes justify this with the same shared-grid alignment argument used for running and image variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. The notes treat these as the canonical active-task outcome labels from the SDK/NWB trial logic.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome labels to integer codes and broadcasts the selected code across every time bin in the trial.

ii. 
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The notes justify making all outputs time-varying for a uniform decoder format, even when the underlying variable is static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or imperfect data mainly by exclusion or interpolation: it drops sessions without required pupil-tracking groups, filters pupil interpolation to finite samples, raises an error if a trial has no valid outcome label or a session has no valid pupil samples, uses a constant fill if only one pupil sample is finite, and skips sessions with fewer than two valid trials.

ii. 
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
if valid.sum() == 1:
    return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
...
if int(raw["keep_mask"].sum()) < 2:
    print(
        f"[pass2] skipping session {session.ophys_experiment_id} because it has "
        f"{int(raw['keep_mask'].sum())} valid trials"
    )
    continue
```

iii. The notes specifically mention excluding a few active sessions that lacked usable pupil data so both passes would operate on the same session subset.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are the two full passes over all sessions and the per-trial interpolation of the neural event matrices onto the 30 Hz grid.

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

iii. The notes explicitly identify neural interpolation as the dominant cost after the agent switched to a two-pass direct-HDF5 design.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over every kept trial in both passes, and `stimulus_identity_codes` still has a Python loop over interval assignments. Those are the clearest vectorization opportunities.

ii. 
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
    ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes acknowledge that trial-by-trial interpolation remains the main bottleneck even after some vectorization over neurons inside each trial.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats session loading and behavioral interpolation across two passes: pass 1 recomputes per-trial running and pupil signals only to estimate global bin edges, and pass 2 rereads the same sessions and recomputes those interpolations to build the final dataset.

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
...
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
```

iii. The notes explicitly describe the converter as a two-pass design and note that this avoids storing all neural arrays at once, at the cost of repeated reading and interpolation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script reads and keeps some stimulus-table columns it never uses downstream (`trials_id`, `active`, `flashes_since_change`), computes native ophys dt and valid-trial summaries mostly for logging, and includes optional plotting code that is not part of the saved dataset.

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
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
...
if show_processing and (sess_num <= 2) and not plotted:
    make_processing_plot(...)
```

iii. The notes frame some of this as sanity-check and debugging support rather than essential downstream conversion logic.
