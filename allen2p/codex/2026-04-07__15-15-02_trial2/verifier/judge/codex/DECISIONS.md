# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads local metadata from `/app/data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`, matches rows to NWB files present under `behavior_ophys_experiments/`, drops passive sessions, drops sessions missing eye-tracking groups, and then opens each remaining NWB directly with `h5py`. Trial, stimulus, neural, running, and pupil streams are then read from HDF5 groups inside each file.

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

```python
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
```

iii. The justification is explicit in `CONVERSION_NOTES.md` Steps 4-6 and 10: the agent decided the local disk was a curated subset of the release, chose to work directly from those NWB files, and said it used `h5py` because `BehaviorOphysExperiment.from_nwb_path` was unusable in the environment due to a `pynwb/hdmf` compatibility mismatch. It also justified excluding passive and missing-eye-tracking sessions because the decoder required trial outcomes and pupil output.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are unique `mouse_id` values taken from `ophys_experiment_table.csv` after the session-level filtering. During conversion the code builds a global `subjects` list and assigns each session a `subject_idx`.

ii.
```python
SessionInfo(
    ...
    mouse_id=str(int(row.mouse_id)),
    ...
)
```

```python
subject_to_idx: Dict[str, int] = {}
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The notes justify this only implicitly: Step 5 says to use NWB/session metadata `mouse_id` for `subjects` and `subject_idx`, because that is the subject identifier exposed by the dataset metadata.

## 1-c. How are the data split into sessions?

i. Each remaining NWB file, identified by `ophys_experiment_id`, is treated as one decoder session. The AI does not regroup multiple experiments into a single `ophys_session_id`; instead it sorts and processes experiment files directly.

ii.
```python
file_map = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table.sort_values("ophys_experiment_id")
```

```python
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. The justification is explicit in Step 4 of `CONVERSION_NOTES.md`: the agent saw that one NWB file corresponds to one `ophys_experiment`, noted that local data contained 284 experiment files but 247 unique `ophys_session_id`s, and resolved this by treating each NWB experiment file as one decoder session because the neural traces were experiment-specific.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For every kept trial, the AI uses `start_time` and `stop_time` to define the trial boundaries, then creates a uniform 30 Hz time grid from `start_time` to `stop_time`.

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

iii. The notes justify this in Step 5: “Segment trials from `start_time` to `stop_time`” using the NWB `intervals/trials` table, after filtering to kept go/catch trials. The agent wanted the full behavioral trial window rather than a fixed window around change.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are `go` or `catch`, and are not `aborted` and not `auto_rewarded`. Sessions are also filtered out earlier if they are passive or missing eye tracking, and later if they have fewer than 2 valid trials.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
```

```python
exp_table = exp_table[~exp_table["passive"]].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

```python
if int(raw["keep_mask"].sum()) < 2:
    print(
        f"[pass2] skipping session {session.ophys_experiment_id} because it has "
        f"{int(raw['keep_mask'].sum())} valid trials"
    )
    continue
```

iii. The notes explicitly justify the trial filter in Steps 4-5 as matching the common interpretation across SDK and papers: include `go` and `catch`, exclude `aborted` and `auto_rewarded`. They also justify excluding passive sessions because the decoder required trial outcome, and excluding 3 active sessions with no eye-tracking because pupil output was required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from precomputed calcium event traces in `processing/ophys/event_detection/data`, with timestamps from `processing/ophys/event_detection/timestamps`.

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

iii. The justification is explicit in Step 4 and Step 5 of `CONVERSION_NOTES.md`: the agent noted that the strategy paper used detected calcium events rather than raw dF/F, and chose precomputed `events` to better match that analysis path.

## 2-b. How is the `neural` data processed?

i. The AI does not use native frame slices directly. It linearly interpolates the full event matrix from native ophys timestamps onto each trial’s 30 Hz grid, then transposes the result to `(n_neurons, n_timepoints)` and stores it as `float32`.

ii.
```python
def interpolate_matrix(
    source_t: np.ndarray, source_values: np.ndarray, query_t: np.ndarray
) -> np.ndarray:
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
```

```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify this in Step 5: because sessions mixed native ophys rates, the agent decided to put every stream on a common 30 Hz grid. Step 6 also states that trial-level interpolation over neurons was intentionally vectorized.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI conditionally filters event traces using `valid_roi` from the cell specimen table if the raw event matrix still appears to include invalid ROIs. Otherwise it keeps all loaded event traces.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The justification appears in Steps 1, 4, and 10 of `CONVERSION_NOTES.md`: the agent read that the SDK normally excludes invalid ROIs, and because it bypassed the SDK object loader it tried to mirror that filtering from the NWB `valid_roi` field when needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. Each trial grid begins at `start_time`, ends at `stop_time`, and all neural samples for the trial are resampled onto that grid.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
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

iii. The notes justify this in Step 5 and Step 10: the agent explicitly planned to segment trials from `start_time` to `stop_time` and described the common 30 Hz trial grid as being derived from raw trial timing in absolute experiment time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed bin size of `1/30` s, i.e. `33.333... ms`, for all sessions and all streams. The AI resamples neural, running, and pupil data onto this common 30 Hz grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
```

```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

```python
"metadata": {
    ...
    "time_bin_size": float(TIME_BIN_SIZE_MS),
    "resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
}
```

iii. The justification is explicit in Step 5: the target format required one shared bin size across sessions, local sessions mixed native rates, and the agent believed the paper’s 30 Hz interpolated analyses made a common 30 Hz grid the most defensible choice.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the chosen stimulus-presentation table, specifically `start_time`, `stop_time`, `image_name`, and `omitted`. The code does not use `initial_image_name` / `change_image_name` from the trial table.

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

iii. The justification is explicit in Step 5 and Step 10 of `CONVERSION_NOTES.md`: the agent planned to use stimulus-presentation intervals from the active task block and to represent actual displayed images, with a separate gray class for gray-screen and omission periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a global image vocabulary from all non-omitted stimulus presentations, prepends a special `"gray"` category, and then assigns each trial time bin the current stimulus image code or the gray code if the bin falls in a gray/omitted interval.

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
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The justification is explicit in Step 5: the agent wanted “actual image name during image display” and `"gray"` during gray-screen or omission periods, because the task contains 500 ms gray gaps and omission trials.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same 30 Hz per-trial grid used for neural interpolation. For each bin time, the code finds the most recent stimulus-presentation interval and assigns the corresponding image or gray code.

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

```python
idx = np.searchsorted(starts, query_t, side="right") - 1
...
in_interval = query_t[valid] < stops[idx_valid]
```

iii. The justification is implicit in the 30 Hz common-grid plan from Steps 5-6 and explicit in Step 10, where the agent describes all trial streams as aligned in absolute experiment time and resampled to one common grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag, together with `start_time`, `stop_time`, and `omitted`. The code does not compute it from trial `change_time` plus the `go` flag.

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
def stimulus_change_codes(stimulus: Dict[str, np.ndarray], query_t: np.ndarray) -> np.ndarray:
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
```

iii. The justification is explicit in Step 6 and Step 10 of `CONVERSION_NOTES.md`: the agent revised an earlier `change_time`-based impulse and decided to use raw stimulus `is_change` intervals instead, so that `image_change` would label the changed-image presentation window rather than a single instant.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary time series over the 30 Hz trial grid. Each bin is labeled 1 if it falls inside a non-omitted stimulus-presentation interval whose `is_change` flag is true; otherwise it is 0.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
assign = np.flatnonzero(valid)[in_interval]
codes[assign] = changed.astype(np.int64)
```

iii. The justification is explicit in trajectory steps 155 and 160 and in Step 10 notes: the original one-bin impulse target was considered too sparse, so the AI widened the positive label to the changed-image presentation window while claiming this was still faithful to “right after a change.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric thresholding is applied. `image_change` is already treated as a binary categorical variable, with 0 for `no_change` and 1 for `change`.

ii.
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    outcome_values,
],
```

```python
codes = np.zeros(query_t.shape, dtype=np.int64)
...
codes[assign] = changed.astype(np.int64)
```

iii. The justification is implicit in the task framing and explicit in the metadata/output schema: this output was treated as inherently binary, so the code simply used the Boolean `is_change` flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is evaluated on exactly the same 30 Hz grid as the neural trial data, using the same `grid` vector and the same trial boundaries.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
...
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The justification is the same common-grid argument stated in Steps 5, 6, and 10: all streams are aligned in absolute experiment time and resampled onto the shared trial grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in each NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The justification is implicit and matches the notes: Step 1 identified SDK running-speed processing, and Step 5 mapped the decoded running-speed output to the NWB running-speed timeseries.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The running-speed signal is linearly interpolated onto each trial’s 30 Hz grid, then discretized into 5 global quintile bins computed from all interpolated running-speed values across all kept trials and sessions.

ii.
```python
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
...
running_all = np.concatenate(running_values).astype(np.float32)
running_edges = robust_quintile_edges(running_all)
```

```python
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The justification is explicit in Step 5: the agent planned to interpolate onto the common 30 Hz grid and to define discrete running-speed classes globally so categories were consistent across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 bins using global quintile edges at the 20th, 40th, 60th, and 80th percentiles. The helper also enforces strictly increasing edges by adding a tiny epsilon if percentile ties occur.

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

iii. The justification is explicit in Step 5: the agent wanted five equal-percentile bins defined globally, not separately per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolating it to the same 30 Hz trial grid that is used for neural interpolation, so every running-speed bin corresponds to a neural time bin.

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

iii. The justification is the same shared-grid decision stated in Steps 5 and 10.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from raw eye-tracking ellipse widths and heights in `acquisition/EyeTracking/pupil_tracking/width` and `.../height`, together with eye-tracking timestamps. The code defines pupil diameter as `max(width, height)`.

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

iii. The justification is explicit in Step 5: the agent planned to compute pupil diameter as `max(width, height)` and claimed this, together with interpolation across valid timestamps, matched the blink-masked eye-tracking output closely enough for the decoder target.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI takes `max(width, height)` per eye-tracking frame, discards non-finite samples during interpolation, linearly interpolates the resulting signal onto each trial’s 30 Hz grid, and then discretizes it into 5 global quintile bins.

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
```

```python
pupil_values.append(
    interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
)
...
pupil_all = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
```

iii. The justification is partly explicit and partly implicit. Step 5 says the agent would “use blink-masked values” and interpolate them onto the common grid, but the actual code relies only on finite-value filtering, not an explicit blink mask. The notes later justify excluding sessions that lacked eye tracking entirely.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 percentile bins using the same `robust_quintile_edges` / `digitize_with_edges` scheme used for running speed.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The justification is explicit in Step 5: global quintile bins were chosen so the categories would be consistent across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 30 Hz trial grid used for neural data, so pupil and neural arrays are time-aligned bin by bin.

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

iii. The justification is the same common-grid decision described in Steps 5 and 10.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the Boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `intervals/trials`.

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

iii. The justification is implicit in the trial-table mapping and explicit in Step 10, where the agent states that trial outcome comes from NWB trial outcome flags.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome labels to integer codes and broadcasts the selected code across all time bins in the trial, making it time-varying only by repetition.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The justification is explicit in Step 5: the agent planned to “represent all outputs as time-varying,” so static trial outcome would be repeated across the whole trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues by exclusion or interpolation rather than by full error recovery. It excludes sessions missing eye-tracking groups, skips sessions with fewer than 2 valid trials, filters out non-finite pupil samples during interpolation, and raises an error if a kept session has no valid pupil samples. It does not include the reference code’s per-session `try/except` fallback.

ii.
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
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

iii. The justification is explicit in Steps 9-10 of `CONVERSION_NOTES.md`: the agent found 3 active sessions with no eye tracking and excluded them so the pupil output would stay consistent, and it treated all-zero event trials as genuine sparsity rather than conversion errors. There is no explicit justification for not using a broader per-session exception handler.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies neural interpolation and full-session NWB reads as the main runtime costs, with neural interpolation described as the dominant expected bottleneck.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    neural_trial = interpolate_matrix(
        raw["ophys_timestamps"], raw["events"], grid
    ).T.astype(np.float32)
```

```python
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. The justification is explicit in Step 6 and the sample runtime notes: the agent wrote that neural interpolation was still the dominant expected cost because every kept trial required resampling of event traces, while direct HDF5 reads were chosen to reduce loader overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes interpolation across neurons within a trial, but it still loops in Python over sessions and over every kept trial. Within image-identity assignment it also loops over interval names/omission flags for bins inside the trial.

ii.
```python
for sess_num, session in enumerate(sessions, start=1):
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...
```

```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The justification is mostly implicit. Step 6 explicitly says trial-level interpolation was vectorized across neurons; beyond that, the notes do not defend the remaining Python loops, so this is mainly inferred from the implementation.

## 9-c. What processing does the code repeat multiple times?

i. The code is intentionally two-pass. It rereads every session twice and recomputes per-trial running and pupil interpolation twice: once in `collect_global_statistics()` to build global bin edges and image categories, and again in `convert_sessions()` to build the saved trial arrays.

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
raw = read_session_raw(session, load_events=False)
...
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
...
raw = read_session_raw(session)
...
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
```

iii. The justification is explicit in Step 6: the agent deliberately used a two-pass design to avoid holding all neural arrays in memory while still computing global discretization statistics.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints some statistics that are not used in the saved dataset, such as `native_dt_by_session` and `valid_trial_counts`. It also reads some stimulus columns that are not subsequently used (`trials_id`, `active`, `flashes_since_change`). Optional processing plots are likewise diagnostic only.

ii.
```python
valid_trial_counts: Dict[int, int] = {}
native_dt_by_session: Dict[int, float] = {}
...
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
```

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

iii. The justification is only partial. The notes justify these as sanity-check or diagnostic aids, especially in Steps 5, 7, and 10, but they are not needed by the downstream decoder data structure itself.
