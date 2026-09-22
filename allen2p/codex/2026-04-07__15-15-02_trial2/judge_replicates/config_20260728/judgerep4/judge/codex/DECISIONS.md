# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the Allen SDK cache or `VisualBehaviorOphysProjectCache`. It reads the local release subset directly from `/app/data/visual-behavior-ophys-1.1.0`, loads `ophys_experiment_table.csv`, maps each listed `ophys_experiment_id` to an NWB file, drops passive sessions, and further excludes sessions missing eye-tracking groups. Session contents are then read directly from each NWB with `h5py`.

ii.
```python
DATA_ROOT = Path("/app/data")
RELEASE_ROOT = DATA_ROOT / "visual-behavior-ophys-1.1.0"
EXPERIMENT_DIR = RELEASE_ROOT / "behavior_ophys_experiments"
METADATA_DIR = RELEASE_ROOT / "project_metadata"

def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
```

```python
def read_session_raw(session: SessionInfo, load_events: bool = True) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        ...
        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
        )
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the agent says it switched to direct `h5py` reads because `BehaviorOphysExperiment.from_nwb_path` failed with a `pynwb/hdmf` `external_resources` error. The notes also say passive sessions were excluded because the decoder requires trial outcomes, and sessions without eye-tracking pupil data were excluded so pupil outputs could be computed.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from `ophys_experiment_table.csv`. Each `mouse_id` is converted to a string and assigned a global subject index when a session is added.

ii.
```python
SessionInfo(
    ...
    mouse_id=str(int(row.mouse_id)),
    ...
)
```

```python
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The notes describe `mouse_id` as the subject identifier exposed by the release metadata and map it directly to `subjects` / `subject_idx`.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file / `ophys_experiment_id` as one session. It does not group multiple experiments by shared `ophys_session_id`; instead it processes the filtered `ophys_experiment_table.csv` rows in sorted `ophys_experiment_id` order.

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
```

```python
exp_table = exp_table.sort_values("ophys_experiment_id")
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
```

iii. In Step 4 of `CONVERSION_NOTES.md`, the agent explicitly resolves the experiment/session ambiguity by saying each NWB experiment file will be treated as one decoder session because neural traces are experiment-specific.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For every kept trial, the code uses `start_time` and `stop_time` as trial boundaries and constructs a per-trial time grid from start to stop. This gives variable-length trials, but at a common 30 Hz bin size rather than native ophys frames.

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
```

```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. In Step 5 of the notes, the agent states it will segment trials directly from the NWB trial table using `start_time` to `stop_time` after filtering to valid go/catch trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if `(go or catch) and not aborted and not auto_rewarded`. The code does not additionally require non-null `change_time`. Sessions with fewer than 2 kept trials are skipped.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
```

```python
if int(raw["keep_mask"].sum()) < 2:
    print(
        f"[pass2] skipping session {session.ophys_experiment_id} because it has "
        f"{int(raw['keep_mask'].sum())} valid trials"
    )
    continue
```

iii. The notes justify this as matching the task’s instruction to include go and catch trials but exclude aborted and auto-rewarded trials. The agent also documented a sanity check that active sessions in the local subset all had at least two valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from `processing/ophys/event_detection/data` with timestamps from `processing/ophys/event_detection/timestamps`, not from dF/F traces.

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

iii. In Step 4 and Step 5 of the notes, the agent says it chose events rather than dF/F because the strategy paper’s neural analyses used detected calcium events, and because those events were already present in the NWB files.

## 2-b. How is the `neural` data processed?

i. The code uses raw event matrices from each NWB file, optionally filters to `valid_roi`, and then linearly interpolates the full event matrix onto each trial’s 30 Hz grid. Trials are stored as `(n_neurons, n_timepoints)` after transposing the interpolated `(T, N)` array.

ii.
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
```

```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify this as recreating SDK ROI filtering when needed and putting every session on a common 30 Hz grid because the target format requires a shared bin size while the dataset mixes ~31 Hz and ~11 Hz imaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent attempts to filter neural traces to ROIs marked `valid_roi` in the NWB `cell_specimen_table`, but only when the event matrix width matches the table width and the count of valid ROIs differs from the current trace width.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. Step 1 and Step 4 of the notes say the SDK normally uses `exclude_invalid_rois=True`, so the agent tried to preserve that QC rule when bypassing the SDK and reading HDF5 directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start. For each trial, the code builds a grid beginning at `start_time` and ending just before `stop_time`, then interpolates the event traces onto that grid using absolute ophys timestamps.

ii.
```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

```python
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The metadata record `"temporal_alignment_event": "trial start"`, and the notes repeatedly describe the common-grid interpolation as being tied to raw trial `start_time` / `stop_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid (`1/30 s`, `33.33 ms`) for all sessions and trials. Yes, temporal rebinning / resampling is applied by interpolation from native timestamps to this new grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
```

```python
"time_bin_size": float(TIME_BIN_SIZE_MS),
"resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
```

iii. In Step 5 of the notes, the agent explicitly says it chose a common 30 Hz time base because behavior and eye tracking are naturally 30 Hz and the local ophys data mix ~31 Hz single-plane and ~11 Hz multi-plane sampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the chosen stimulus-presentation interval table: `start_time`, `stop_time`, `image_name`, and `omitted`. The code does not use trial-table `initial_image_name` / `change_image_name`.

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

```python
def stimulus_identity_codes(
    stimulus: Dict[str, np.ndarray], query_t: np.ndarray, image_to_code: Dict[str, int]
) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]
```

iii. The notes justify this by saying the output should reflect the actual flashed-image stream. They also add an explicit `gray` class for gray-screen and omission periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code first builds a global image vocabulary from all non-omitted stimulus presentations and prepends a `"gray"` category. Then, for each trial time bin, it finds the active stimulus interval by `searchsorted`; bins outside a stimulus interval, or inside omitted intervals, are labeled as `gray`, otherwise they are assigned the corresponding `image_name` code.

ii.
```python
image_names.update(
    str(x)
    for x, omitted in zip(stim["image_name"], stim["omitted"])
    if (not omitted) and str(x) not in ("", "None", "nan")
)
...
image_values = ["gray"] + sorted(image_names)
```

```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. Step 5 of the notes says the agent intentionally used `gray` as an explicit class because the task includes 500 ms gray periods and omission periods, so a complete time-varying image-identity signal needed more than initial/change image labels.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the exact same per-trial 30 Hz grid used for the neural interpolation. Every time bin gets a category based on which stimulus interval contains that bin’s absolute timestamp.

ii.
```python
grid = session_grid(start, stop)
...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
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

iii. The agent’s notes and Step 10 sanity checks say the image-identity output was reconstructed from raw stimulus intervals on the same trial grid as the neural data and matched direct raw-data reconstructions.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table fields `is_change`, `start_time`, `stop_time`, and `omitted`, rather than from the trial table’s `change_time`.

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
```

```python
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The trajectory and notes say the agent switched from an initial one-bin impulse at `change_time` to the changed-image presentation window because the impulse target was “too sparse” and `is_change` better reflected the changed flash.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each trial bin, the code finds the containing stimulus interval and emits `1` when that interval is both `is_change` and not omitted; otherwise it emits `0`.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
...
codes[assign] = changed.astype(np.int64)
```

iii. The justification in the trajectory is explicit: the agent says the original impulse label was too sparse for decoding, and that marking the changed-image presentation window was still faithful to the task’s “right after a change” wording.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is applied beyond converting the boolean changed-flash indicator into binary categories `0` and `1`, named `["no_change", "change"]`.

ii.
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    outcome_values,
]
```

```python
codes = np.zeros(query_t.shape, dtype=np.int64)
...
codes[assign] = changed.astype(np.int64)
```

iii. No separate justification was needed in the notes; the agent treats image change as an already-binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned on the same 30 Hz per-trial grid as neural, running, pupil, and image identity. The absolute time of each bin determines whether it falls inside a changed-image stimulus interval.

ii.
```python
grid = session_grid(start, stop)
...
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The notes describe all outputs as being reconstructed on a common trial grid tied to raw timestamps, and Step 10 says direct raw-data sanity checks matched the converted output row exactly.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` and its timestamps from `processing/running/speed/timestamps` in each NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes say this mirrors the SDK’s running-speed stream while avoiding SDK object construction.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz grid, pooled across all kept trials to compute global quintile edges, then digitized into five bins.

ii.
```python
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
...
running_edges = robust_quintile_edges(running_all)
```

```python
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. In the notes, the agent justifies global discretization so categories are shared across sessions, and the common 30 Hz grid so all streams share one bin size.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global percentile bins using the 20th, 40th, 60th, and 80th percentiles of all interpolated running-speed samples from kept trials. A small epsilon is added if percentile edges are non-monotonic.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles
```

```python
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The notes explicitly say continuous outputs should be discretized globally, not per session, so that class meanings are consistent across the dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same 30 Hz trial grid that is used for the neural event traces.

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

iii. The notes justify the shared grid as the common alignment reference for all trial-wise signals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the raw eye-tracking ellipse widths and heights in `acquisition/EyeTracking/pupil_tracking/{width,height}` and timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

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
```

iii. In Step 5, the notes say the agent planned to compute pupil diameter as `max(width, height)` and to use eye-tracking-derived values rather than a higher-level SDK column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes pupil diameter as `max(width, height)`, interpolates finite values onto each trial’s 30 Hz grid, pools those values across all kept trials to get global quintile edges, and digitizes the result into five bins.

ii.
```python
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

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
```

iii. The notes say the intended rationale was to use blink-masked pupil values and a global quintile discretization on the shared 30 Hz grid. The trajectory/notes do not contain an explicit defense of using `max(width, height)` instead of the SDK’s `pupil_width`; that appears to be the agent’s own interpretation of “diameter.”

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins, using the same `robust_quintile_edges` and `digitize_with_edges` functions as running speed.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes justify using dataset-wide quintile bins so all sessions share one categorical definition of pupil size.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same per-trial 30 Hz grid used for the neural traces.

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

iii. The notes frame the common 30 Hz grid as the shared alignment reference for neural and behavioral streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `intervals/trials`.

ii.
```python
trials = read_interval_table(
    trial_group,
    [
        "go", "catch", "aborted", "auto_rewarded",
        "hit", "miss", "false_alarm", "correct_reject",
        "change_time", "start_time", "stop_time",
    ],
)
```

```python
def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
```

iii. The notes say these are the task’s canonical trial-outcome labels and are required for the requested decoder target.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to a fixed integer code order `["hit", "miss", "false_alarm", "correct_reject"]` and then broadcast across all time bins in a trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. In Step 5, the notes explicitly say all outputs, including trial outcome, will be represented as time-varying by repeating static labels across the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles some edge cases but not many of the same ones as the reference solution. It excludes sessions lacking required eye-tracking groups, raises an error if no finite pupil samples exist, interpolates over finite pupil samples, optionally filters invalid ROIs, and skips sessions with fewer than 2 kept trials. It does not explicitly clip trials to available ophys frames, does not map NaNs to a fallback category, and does not use `change_time.notna()` in its keep mask.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

```python
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
```

```python
if int(raw["keep_mask"].sum()) < 2:
    ...
    continue
```

iii. The notes justify the eye-tracking exclusion as necessary for the pupil output and describe sparse-event warnings as genuine properties of the released data rather than conversion bugs. There is little explicit justification for the missing `change_time` filter or for not using an explicit missing-data category.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is repeated NWB loading plus per-trial interpolation, especially neural interpolation onto the 30 Hz grid. The code is structured as a two-pass conversion, so session files are read once to collect global statistics and again to build the final dataset.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
        running_values.append(...)
        pupil_values.append(...)
```

```python
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
        neural_trial = interpolate_matrix(
            raw["ophys_timestamps"], raw["events"], grid
        ).T.astype(np.float32)
```

iii. In Step 6 and Step 7 of the notes, the agent explicitly says neural interpolation is the dominant expected cost and provides timing estimates for pass 1 and pass 2.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes interpolation across neurons within a trial, but several Python loops remain: iterating over sessions twice, iterating over trials within each session, and iterating over stimulus bins inside `stimulus_identity_codes` when assigning image labels for omitted vs non-omitted presentations. These are the clearest remaining vectorization opportunities.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
```

```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes claim trial-level interpolation is already vectorized across neurons. They do not explicitly discuss the remaining loops, so this is mostly reconstructed from the code itself.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats full-session file reads across two passes. It also repeats trial-grid construction and interpolation of running speed and pupil once in pass 1 for global binning and again in pass 2 for final trial assembly. Session-level summary quantities such as native dt and valid-trial counts are also computed only for logging / QA and then discarded.

ii.
```python
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(
    sessions
)
...
data = convert_sessions(
    sessions=sessions,
    running_edges=running_edges,
    pupil_edges=pupil_edges,
    image_values=image_values,
    show_processing=args.show_processing,
)
```

```python
grid = session_grid(start, stop)
running_values.append(interpolate_vector(..., grid))
pupil_values.append(interpolate_pupil(..., grid))
```

```python
grid = session_grid(start, stop)
running_cont = interpolate_vector(..., grid)
pupil_cont = interpolate_pupil(..., grid)
```

iii. Step 6 of the notes openly describes the two-pass design as a deliberate memory/speed tradeoff: avoid storing all neural arrays up front, at the cost of re-reading sessions and recomputing some interpolations.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra QA / bookkeeping work that is not used downstream by the decoder: optional plotting, collection of `valid_counts` and `native_dt` only for console summaries, and reading several stimulus-table fields (`trials_id`, `active`, `flashes_since_change`) that are not used in the final outputs. The pass-1 per-trial running/pupil interpolation is also discarded after only the global bin edges are kept.

ii.
```python
stim = read_interval_table(
    stim_group,
    [
        "start_time", "stop_time", "image_name", "is_change",
        "omitted", "trials_id", "active", "flashes_since_change",
    ],
)
```

```python
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(
    sessions
)
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
```

```python
if show_processing and (sess_num <= 2) and not plotted:
    make_processing_plot(...)
```

iii. The notes justify most of this as validation and sanity checking rather than as part of the actual decoder dataset. There is no claim that these extra computations are required by the output format itself.
