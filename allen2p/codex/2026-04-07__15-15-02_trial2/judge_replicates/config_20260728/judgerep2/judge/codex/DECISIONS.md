# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads the local metadata CSV `ophys_experiment_table.csv`, matches rows to local NWB files, filters out passive sessions and sessions missing eye-tracking data, and then loads each kept NWB directly with `h5py`. Trial and stimulus tables are then read from each NWB file.

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

def read_session_raw(session: SessionInfo, load_events: bool = True) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        ...
        stim_group = choose_task_presentation_group(h5f)
        ...
```

iii. In Step 5 and Step 10 of `CONVERSION_NOTES.md`, the AI justified this as a practical substitute for the SDK because the local `pynwb/hdmf` stack could not instantiate these NWB files, so it mirrored the SDK’s field semantics with direct HDF5 reads instead.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from `ophys_experiment_table.csv`. The script stores subject ids as strings and assigns each session a `subject_idx`.

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

iii. The notes explicitly say to use mouse identifiers from metadata for `subjects` and `subject_idx` because `mouse_id` is the natural per-animal key in the dataset.

## 1-c. How are the data split into sessions?

i. The AI treats each local NWB file, i.e. each `ophys_experiment_id`, as one decoder session. It does not group multiple experiments by shared `ophys_session_id`.

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

iii. In Step 4 and Step 5 notes, the AI said the local data are one NWB per experiment and decided to use each experiment file as one session because neural traces are experiment-specific and the local subset should be treated at that granularity.

## 1-d. How are the data split into trials?

i. Trials are split with the NWB `intervals/trials` table. For each kept trial, the script uses `start_time` and `stop_time` to build a per-trial time grid and then extracts or interpolates all streams onto that grid.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The notes say trial boundaries come directly from the SDK/NWB trial definition and should run from `start_time` to `stop_time`, after trial-type filtering.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are `go` or `catch`, and are not `aborted` and not `auto_rewarded`. Sessions with fewer than 2 kept trials are dropped.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session {session.ophys_experiment_id} because it has "
          f"{int(raw['keep_mask'].sum())} valid trials")
    continue
```

iii. The Step 4 and Step 5 notes justify this as matching the task instruction to include go and catch trials while excluding aborted and auto-rewarded trials; the notes also mention a minimum of two valid trials per session for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/data`, with timestamps from `processing/ophys/event_detection/timestamps`.

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

iii. In Step 4 and Step 5 notes, the AI explicitly chose precomputed calcium `events` instead of `dff_traces`, arguing that the strategy paper’s analyses used detected calcium events.

## 2-b. How is the `neural` data processed?

i. The AI linearly interpolates full-session event traces onto each trial’s common 30 Hz grid, then transposes the result to `(n_neurons, n_timepoints)`. It does not combine multiple imaging planes into one session because each NWB experiment is treated as its own session.

ii.
```python
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes justify this as using a single common 30 Hz time base across sessions because behavior and eye tracking are naturally 30 Hz and because the AI wanted one shared bin size for decoder input/output alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI conditionally filters neurons by `valid_roi` from the cell specimen table when the raw event matrix appears to include the unfiltered ROI table.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. In Step 4, Step 5, and Step 10 notes, the AI says this is meant to mirror the SDK’s `exclude_invalid_rois=True` behavior when reading the NWB directly rather than through the SDK.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start: each per-trial grid begins at `start_time` and ends at `stop_time`, and neural activity is resampled onto that trial-relative grid.

ii.
```python
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The notes say trial start was chosen as the alignment event because the whole trial window is being modeled and all outputs are built on the same start-to-stop trial grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `1/30` s or `33.33 ms` bins. Yes: all streams, including neural activity, are resampled onto that grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The Step 5 notes explicitly justify 30 Hz as a common time base across sessions and modalities; the Step 6 notes repeat that all trial streams are aligned by interpolation onto this common grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the selected stimulus-presentation table’s `image_name`, `start_time`, `stop_time`, and `omitted` columns, rather than from trial-level `initial_image_name` / `change_image_name`.

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

iii. The notes say the AI wanted the time-varying identity signal to come from stimulus-presentation intervals directly, with explicit handling of gray periods and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a global vocabulary of image names, prepends a `gray` category, maps names to integer codes, and then assigns the code active at each 30 Hz trial bin. Omitted flashes and time outside a flash interval are labeled `gray`.

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

iii. The Step 5 notes justify adding `gray` because the trial contains 500 ms gray intervals and omission periods, so the AI wanted a complete time-varying label across the entire trial.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same per-trial 30 Hz `grid` used for neural resampling.

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

iii. The notes repeatedly emphasize that all outputs are generated on the same trial grid as neural activity so that alignment is exact after resampling.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` columns.

ii.
```python
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
```

iii. In Step 5 and Step 10 notes, the AI says it switched to using the stimulus `is_change` presentation window because a more impulse-like definition was too sparse and because this better matched the changed-image flash itself.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 30 Hz time bin, the script finds the active stimulus interval and sets the label to 1 if that interval is a non-omitted change flash; otherwise 0.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
...
codes[assign] = changed.astype(np.int64)
```

iii. The notes say this was chosen to produce a less degenerate decoder target while staying tied to the task’s actual change-stimulus presentation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary category with `0 = no_change` and `1 = change`. There is no further thresholding.

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

iii. The notes describe image change as a binary time-varying target, so the only categorization step is mapping non-change vs change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like the other outputs, image change is computed on the exact same 30 Hz trial grid as the neural data.

ii.
```python
grid = session_grid(start, stop)
...
image_change = stimulus_change_codes(raw["stimulus"], grid)
...
output_trial = np.vstack([image_codes.astype(np.int64), image_change, ...])
```

iii. The Step 10 notes explicitly say all streams are aligned in absolute experiment time and then resampled to the common 30 Hz trial grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes cite the SDK running-speed stream as the intended source and say the direct NWB read mirrors that source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to each trial’s 30 Hz grid, global quintile edges are computed across all kept trials in pass 1, and each bin is digitized in pass 2.

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

iii. The Step 5 notes justify global discretization so the meaning of the running-speed categories is consistent across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five percentile bins using global quintile edges and `np.digitize`, producing integer bins `0` through `4`.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    ...

def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The notes explicitly say the task-required discretization should use five equal-percentile bins computed globally over the converted dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same per-trial 30 Hz grid used for the neural traces.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
```

iii. The notes justify a shared grid across streams so the discrete running label and neural matrix have the same number of time bins in each trial.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width`, `.../height`, and eye-tracking timestamps. The AI defines diameter as `max(width, height)`.

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

iii. The Step 5 notes justify this as a direct pupil-diameter proxy available in the NWB file, and state that blink-related invalidity is handled indirectly through finite-value filtering during interpolation rather than by using the SDK’s blink mask.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI interpolates the per-session pupil-diameter trace onto the 30 Hz trial grid using only finite samples, computes global quintile edges in pass 1, and digitizes into bins in pass 2.

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

iii. The notes justify global discretization for consistency across sessions, and frame the interpolation as filling small gaps after masking invalid samples.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five global percentile bins using the same quintile-edge and digitization logic as running speed.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
"output_values": [
    list(image_values),
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    outcome_values,
],
```

iii. The notes say pupil diameter, like running speed, must be converted to five equal-percentile bins for the decoder task.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid used for neural activity.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
```

iii. The notes treat pupil, running, stimulus labels, and neural activity as co-registered only after resampling them onto the common trial grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trials = read_interval_table(
    trial_group,
    [
        "go",
        "catch",
        "aborted",
        "auto_rewarded",
        "hit",
        "miss",
        "false_alarm",
        "correct_reject",
        ...
    ],
)
...
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. The notes describe these as the canonical trial-outcome flags defined by the dataset’s trial logic.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The boolean outcome flags are mapped to integer codes in the fixed order `hit`, `miss`, `false_alarm`, `correct_reject`, and then broadcast across all time bins in the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The Step 5 notes explicitly say static trial outcome should be repeated across time so that every output variable has a time-varying row in the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some missing-data cases by excluding sessions with missing eye-tracking groups, filtering to finite pupil samples before interpolation, forcing quintile edges to be strictly increasing, and dropping sessions with fewer than 2 valid trials. It does not use a broad `try/except` around session conversion.

ii.
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
...
valid = np.isfinite(pupil_diameter)
...
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
...
if len(session_neural) < 2:
    ...
```

iii. The notes say the AI excluded three sessions that entirely lacked pupil data, treated sparse-event warnings as genuine data rather than bugs, and used finite-sample interpolation plus session exclusion instead of broader imputation.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify two expensive parts: reading each NWB session in both passes and interpolating the neural event matrix for every kept trial. The notes treat neural interpolation as the dominant compute cost.

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

iii. Step 6 and Step 7 of `CONVERSION_NOTES.md` explicitly say neural interpolation is the expected bottleneck and that the full run estimate is driven by neuron-bin interpolation work.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest remaining Python loops are the per-session loop, the per-trial loop within each session, and the small loop inside `stimulus_identity_codes` that assigns image-name codes one interval at a time. Neural interpolation itself is already vectorized across neurons within a trial.

ii.
```python
for sess_num, session in enumerate(sessions, start=1):
    ...
    for trial_idx in np.flatnonzero(raw["keep_mask"]):
        ...

for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The notes mention that trial-level interpolation was vectorized across neurons, implying these higher-level Python loops were left in place as the remaining serial structure.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats full-session loading and some interpolation work across two passes. Pass 1 loads every session to compute global running/pupil edges and image categories; pass 2 loads the same sessions again to build final neural and output arrays. Running and pupil interpolation are therefore done twice for every kept trial.

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

iii. The Step 6 notes justify this repetition as a memory-saving two-pass design: compute global statistics first without retaining all neural arrays, then do the final conversion once the global bin edges are known.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads some stimulus columns that are never used downstream (`trials_id`, `active`, `flashes_since_change`), computes `valid_counts` and `native_dt` only for logging, and imports/defines some optional plotting machinery even when `--show-processing` is not used.

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

iii. The notes frame these as diagnostics and sanity-check support rather than core decoder inputs, so they are retained mainly for validation and documentation rather than downstream modeling.
