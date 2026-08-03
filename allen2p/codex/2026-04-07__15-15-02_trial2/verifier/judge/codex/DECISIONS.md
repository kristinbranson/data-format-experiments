# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads local release metadata from `project_metadata/ophys_experiment_table.csv`, matches rows to local NWB files in `behavior_ophys_experiments/`, filters out passive sessions and sessions missing eye-tracking groups, and then opens each NWB directly with `h5py`. Within each kept NWB file it reads trials, stimulus presentations, ophys event timestamps/data, running speed, and pupil measurements.

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
    ophys_timestamps = np.asarray(
        h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
    )
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose direct `h5py` loading because the local `pynwb/hdmf` stack could not instantiate these NWB files through the SDK, so it mirrored SDK field semantics by reading the NWB fields directly. It also chose to exclude sessions without usable eye tracking because pupil diameter was a required decoder output.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from `ophys_experiment_table.csv`, stored as strings. Each kept session is assigned a subject index using that `mouse_id`.

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

iii. The AI justification in the notes is that mouse identifiers are the natural subject identifiers exposed by the SDK/metadata and should populate `subjects`/`subject_idx`.

## 1-c. How are the data split into sessions?

i. Each local NWB file / `ophys_experiment_id` is treated as one decoder session. The AI does not regroup multiple experiments by shared `ophys_session_id`; instead it processes experiment files independently after filtering metadata rows.

ii.
```python
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. In the notes, the AI explicitly resolved the session-definition discrepancy by treating each NWB experiment file as one decoder session because neural traces were experiment-specific in the local subset and SDK object construction was unavailable.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in each NWB file. For each kept trial, the AI uses the trial `start_time` and `stop_time` and creates a trial-specific regular time grid with 30 Hz spacing, then resamples all streams onto that grid.

ii.
```python
trial_group = h5f["intervals"]["trials"]
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The notes say the AI wanted variable-length trials from the SDK/NWB trial table, but because it imposed one common bin size across all sessions, it represented each trial on a new 30 Hz grid rather than native ophys frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials satisfying `(go or catch) and not aborted and not auto_rewarded`. It also drops sessions with fewer than 2 kept trials. It does not explicitly require non-null `change_time`.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    ...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
if len(session_neural) < 2:
    ...
```

iii. The notes justify this as following the task’s go/catch inclusion rule while excluding aborted and auto-rewarded trials. The AI also says at least two valid trials are required for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural activity from `processing/ophys/event_detection/data` with timestamps from `processing/ophys/event_detection/timestamps`, not from dF/F.

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

iii. The notes say this was chosen because the strategy paper discussed detected calcium events, and the AI considered events a better match than dF/F.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from native ophys timestamps onto each trial’s 30 Hz grid, then transposed to `(n_neurons, n_timepoints)`. The AI does not merge multiple imaging planes into one session because each experiment file is treated as a separate session.

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

iii. The AI’s notes justify this by saying the target format needed one shared bin size and that a common 30 Hz grid was “most defensible” because behavior and eye tracking were naturally 30 Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. If the raw event matrix still appears to include all ROI rows from the cell table, the AI filters to `valid_roi == True`; otherwise it leaves the event matrix as-is. No other neuron-level QC is applied.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The notes say this was intended to mimic the SDK’s default valid-ROI filtering when bypassing SDK object construction, without adding extra ad hoc filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start because each trial grid begins at the raw trial `start_time`. The samples themselves come from interpolation of event traces using absolute ophys timestamps.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
"temporal_alignment_event": "trial start",
```

iii. The AI notes describe this as alignment on absolute experiment time using ophys timestamps, but represented on a common trial-start-centered 30 Hz grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid (`1/30 s`, `33.33 ms`) for every trial in every session. The AI rebins/resamples all streams, including neural data, by interpolation to this grid.

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

iii. The AI’s notes say this was a deliberate decision to force a common bin size across mixed native frame rates, and it cites the paper’s use of common 30 Hz interpolation in a different analysis as justification.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the chosen stimulus-presentation table’s `image_name`, `start_time`, `stop_time`, and `omitted` columns, not from trial-table `initial_image_name`/`change_image_name`.

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
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]
```

iii. The AI’s notes say stimulus presentations were used because they directly describe the flashed-image intervals and gray periods, which the AI wanted for a fully time-varying label.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI constructs a global vocabulary `["gray"] + sorted(non-omitted image names)`. For each trial time bin, it looks up which stimulus-presentation interval contains that bin; omitted intervals and gaps are labeled `"gray"`, otherwise the corresponding image name is mapped to an integer code.

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

iii. The notes justify the added `"gray"` class by arguing that gray inter-stimulus periods and omissions are part of the true stimulus stream and therefore need an explicit category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image-identity codes are computed on the exact same per-trial 30 Hz grid that is used for interpolated neural events.

ii.
```python
grid = session_grid(start, stop)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. The notes describe this as cross-stream alignment by putting both stimulus labels and neural activity onto the same common resampled time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` fields. It is not derived from `change_time` in the trials table.

ii.
```python
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The notes say the AI changed this after sample decoding showed a too-sparse target when it used a one-bin impulse at `change_time`; it judged the changed-image presentation interval a better representation of “right after a change.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each query time bin, the AI finds the active stimulus-presentation interval and emits `1` if that interval has `is_change == True` and is not omitted; otherwise it emits `0`.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
assign = np.flatnonzero(valid)[in_interval]
codes[assign] = changed.astype(np.int64)
```

iii. The AI justification in the notes is that this reduces label degeneracy and is still faithful to the task structure because it marks the changed-image presentation window rather than a single change-time impulse.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary with categories `0 = no_change` and `1 = change`.

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

iii. The AI followed the decoder requirement that image change be a binary categorical output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change values are computed on the same 30 Hz per-trial grid used for the interpolated neural events.

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

iii. The notes say all outputs were intentionally resampled onto the same common time grid as neural activity to avoid cross-stream offset.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is read directly from `processing/running/speed/data` with timestamps from `processing/running/speed/timestamps`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes say this mirrors the SDK’s running-speed stream while avoiding SDK object construction.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz grid. Global quintile edges are computed across all kept trials in a first pass, and then each trial’s interpolated running values are digitized into 5 bins in the second pass.

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

iii. The notes justify global discretization because shared output categories were needed across sessions, and running speed was one of the continuous variables explicitly requested for percentile binning.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses 5 global percentile bins (quintiles), encoded as integer bins `0` through `4`.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    ...
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
...
[f"bin_{i}" for i in range(5)]
```

iii. The AI’s notes explicitly say running speed should be discretized globally into five equal-percentile bins to satisfy the decoder specification.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same 30 Hz per-trial grid used for neural events.

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
```

iii. The notes say all streams were resampled onto a common grid so alignment would be by construction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/width` and `pupil_tracking/height`, combined as `max(width, height)`, with timestamps from `eye_tracking/timestamps`.

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

iii. The notes justify this as a direct pupil-diameter proxy from the raw NWB eye-tracking fields, though they also say sessions missing eye tracking had to be excluded entirely.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI takes `max(width, height)` at the raw sampling times, keeps finite samples, linearly interpolates to each trial’s 30 Hz grid, computes global quintile edges in pass 1, and digitizes each trial’s interpolated values in pass 2.

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
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes justify the same global discretization strategy as for running speed and say excluded missing-eye-tracking sessions were necessary because pupil diameter was mandatory.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 global percentile bins, encoded as `0` through `4`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
[f"bin_{i}" for i in range(5)]
```

iii. The AI followed the decoder task instruction to discretize pupil diameter into five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same per-trial 30 Hz grid used for neural events.

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
```

iii. The AI notes repeatedly describe a single common resampled grid as its alignment strategy for neural, running, pupil, and stimulus outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
```

iii. The AI’s notes cite these as the SDK/NWB’s canonical outcome labels for kept go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome labels to integer codes and broadcasts the code across all time bins in a trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The notes say this was done so that even the static per-trial output would fit the time-varying decoder format cleanly.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data mainly by excluding entire sessions that do not have the required eye-tracking groups, skipping sessions with fewer than 2 valid trials, and interpolating pupil using only finite samples. Outside the sampled range, `np.interp` clamps running/pupil to endpoint values rather than inserting NaNs. The script raises an error if a kept session has no valid pupil samples.

ii.
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        ...
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if int(raw["keep_mask"].sum()) < 2:
    ...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. The notes justify explicit exclusion of the three broken eye-tracking sessions because pupil diameter was mandatory, and otherwise treat interpolation-based filling as acceptable for small gaps.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies two expensive steps: the full-session NWB reads in both passes, and especially the per-trial neural interpolation from event timestamps to the 30 Hz grid.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
    ...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
        neural_trial = interpolate_matrix(
            raw["ophys_timestamps"], raw["events"], grid
        ).T.astype(np.float32)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says neural interpolation is the dominant expected cost and that the two-pass design trades repeated reading for lower memory use.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining serial loops are the per-session loop, the per-trial loop inside each session, and the small Python loop inside `stimulus_identity_codes` that assigns image labels interval-by-interval. Neural interpolation is vectorized across neurons within a trial, but trial-level work is not vectorized across trials.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    running_cont = interpolate_vector(...)
    pupil_cont = interpolate_pupil(...)
    image_codes = stimulus_identity_codes(...)
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes say the AI already vectorized the dominant within-trial neural interpolation over neurons, but the overall conversion still retains many trial-by-trial loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats full-session loading twice: pass 1 reads each session without events to collect global running/pupil/image statistics, and pass 2 reads each session again with events to build the converted output. Running and pupil interpolation are also performed in both passes.

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

iii. The notes acknowledge this two-pass structure and justify it as a memory-saving design so global discretization edges can be computed before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and computes several items used only for conversion diagnostics or metadata rather than downstream decoder inputs: `valid_trial_counts`, `native_dt_by_session`, plotting-only example data, and extra stimulus columns such as `active`, `trials_id`, and `flashes_since_change`. It also constructs and stores a `SessionInfo.project_code`/`session_type` path mainly for metadata and plots.

ii.
```python
valid_trial_counts: Dict[int, int] = {}
native_dt_by_session: Dict[int, float] = {}
...
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
if show_processing and (sess_num <= 2) and not plotted:
    make_processing_plot(...)
```

iii. The notes describe these as auditability and sanity-check features rather than parts of the final decoder representation.
