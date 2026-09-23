# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not enumerate `.mat` files. It hard-codes the seven animal IDs, then loads one extensionless `joblib` bundle per animal from `/app/data`. From each bundle it extracts `trace`, `position`, `blocked`, and `envs`, and then iterates through recording days and derived trials.

ii.
```python
DATA_DIR = Path("/app/data")
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]
...
for subject_index, animal in enumerate(ANIMALS):
    ...
    bundle = joblib.load(DATA_DIR / animal)
    ...
    record = bundle[animal]
    trace = record["trace"]
    position = record["position"]
    blocked = record["blocked"]
    envs = np.asarray(record["envs"]).reshape(-1)
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose the joblib files because they contain "the same already-converted arrays" as the MATLAB files and "load substantially faster." It also notes that the reference code can load either joblib or MATLAB dictionaries.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the fixed `ANIMALS` list. The output `subjects` field is that list, and `subject_idx` is assigned by enumerating through it.

ii.
```python
data = {
    ...
    "subjects": ANIMALS.copy(),
    "subject_idx": [],
    ...
}
...
for subject_index, animal in enumerate(ANIMALS):
    ...
    data["subject_idx"].append(subject_index)
```

iii. In the notes, the AI says the repository README and `main.py` identify the same seven animals, and that session ordering should be "animal-list order, then zero-based day."

## 1-c. How are the data split into sessions?

i. Each animal-day is treated as a separate session. After loading one animal, the code loops over `day` along the first axis of `trace`/`position`/`blocked`, and each day becomes one session in the output.

ii.
```python
if not (trace.shape[0] == position.shape[0] == len(blocked) == len(envs)):
    raise ValueError(f"{animal}: day counts disagree across source fields")
for day in range(trace.shape[0]):
    ...
    result = convert_session(
        trace[day],
        position[day],
        blocked[day],
        animal,
        subject_index,
        day,
        str(envs[day]),
    )
```

iii. The AI’s notes explicitly state: "each source animal-day is one target session." It justifies this by saying CellReg registration varies across days, so each day is the natural session unit.

## 1-d. How are the data split into trials?

i. Within each session, the continuous recording is split from frame 0 into consecutive non-overlapping 60-second chunks. The number of trials is `n_source_frames // 1800`, so any remainder at the end is dropped. After later pooling, each trial contains 600 decoder bins.

ii.
```python
SOURCE_FPS = 30
TRIAL_SECONDS = 60
SOURCE_FRAMES_PER_TRIAL = SOURCE_FPS * TRIAL_SECONDS
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES
...
n_source_frames = trace_day.shape[1]
n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: fewer than two complete trials")
n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL
...
for trial in range(n_trials):
    start = trial * SOURCE_FRAMES_PER_TRIAL
    stop = start + SOURCE_FRAMES_PER_TRIAL
```

iii. In the notes, the AI says the task requires non-overlapping 1-minute trials, so it starts at source frame 0, excludes only the incomplete tail, and keeps trial boundaries fixed.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply behavioral trial-quality filtering. The only effective filtering is structural: sessions with fewer than two complete 60-second trials are rejected, and incomplete tail frames are discarded instead of forming a partial trial.

ii.
```python
n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: fewer than two complete trials")
n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL
...
if len(data["neural"][s]) < 2:
    raise AssertionError(f"Session {s} has fewer than two trials")
```

iii. The AI justifies not using the paper decoder’s speed-based frame filtering by saying it would destroy contiguous fixed-duration trials. It also cites the target-format requirement that every session must contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data comes from the `trace` field in each joblib record, indexed by session/day.

ii.
```python
record = bundle[animal]
trace = record["trace"]
...
result = convert_session(
    trace[day],
    position[day],
    blocked[day],
    animal,
    subject_index,
    day,
    str(envs[day]),
)
```

iii. The AI’s notes say `trace` already contains the native binary rising-phase calcium-event representation used by the paper, so no dF/F recomputation or deconvolution should be added.

## 2-b. How is the `neural` data processed?

i. The AI filters to valid cells, casts to `float32`, reshapes each session into `(cell, trial, frame)`, applies Gaussian smoothing with sigma 3 source frames independently within each trial, and then mean-pools non-overlapping groups of 3 frames. The stored trial matrices are `(neurons, 600)` float32 arrays.

ii.
```python
selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
smoothed = gaussian_filter1d(
    selected,
    sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
    axis=2,
    mode="reflect",
)
pooled_neural = smoothed.reshape(
    valid_cells.size, n_trials, POOLED_BINS_PER_TRIAL, POOL_FRAMES
).mean(axis=3, dtype=np.float32)
...
neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
```

iii. The AI repeatedly justifies this as matching the paper’s position decoder: in the docstring it says the "reference position decoder's three-frame Gaussian smoothing / average pooling is retained," and the notes say the mapping was fixed to 100-ms bins to match the decoder’s 3-frame smoothing/pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI retains only cells considered valid for that session/day and performs no place-cell or activity-threshold filtering. In code, validity is tested by checking whether the cell has a finite value at the first frame; cells failing that test are dropped. The code also errors if no valid cells remain or if pooled neural values are non-finite.

ii.
```python
# Native data represent an absent CellReg cell as NaN for its entire day.
valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
if valid_cells.size == 0:
    raise ValueError(f"{animal} day {day}: no registered cells")
...
if not np.isfinite(neural_trial).all():
    raise ValueError(f"{animal} day {day}: retained neural data contain NaN/Inf")
```

iii. The AI’s notes say missing/unregistered day-cell combinations are NaN and that all manually curated cells should be kept; it explicitly rejects place-cell filtering and the paper decoder’s `>5` event feature-selection rule as decoder-specific rather than upstream curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no native experimental event. The AI defines the alignment event as the start of each artificial 1-minute trial segment, and it smooths within trial boundaries so that temporal processing does not bleed across adjacent trials.

ii.
```python
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
smoothed = gaussian_filter1d(
    selected,
    sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
    axis=2,
    mode="reflect",
)
...
"temporal_alignment_event": "start of each non-overlapping 1-minute session segment",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The notes say there is no stimulus or behavior event to align to, so the converter uses deterministic trial-start alignment instead. The trajectory also says the AI wanted to prevent smoothing leakage across held-out trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100-ms bins. The source streams start at 30 Hz, but the AI rebins them by non-overlapping 3-frame pooling after neural smoothing.

ii.
```python
SOURCE_FPS = 30
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES
...
"time_bin_size": 100.0,
...
"temporal_pool_frames": POOL_FRAMES,
```

iii. The AI says this was chosen to "match the reference decoder’s 3-frame smoothing/pooling" rather than keep the native 33.3-ms frame rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from the raw `blocked` field for each day/session.

ii.
```python
blocked = record["blocked"]
...
geometry, blocked_indices = blocked_vector(blocked_entry)
```

iii. The AI’s notes say it deliberately uses `blocked` directly, rather than reconstructing geometry from environment names, because `blocked` already stores the canonical partition indices needed for the requested decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the nested `blocked` entry into a 9-element float32 binary vector with `1` for blocked partitions and `0` for accessible ones. A value of `-1` means no blocked partitions. The same geometry vector is copied to every trial in the session.

ii.
```python
def blocked_vector(blocked_entry) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    if values.size == 1 and values[0] == -1:
        indices: list[int] = []
    else:
        ...
        indices = [int(v) for v in values]
    ...
    vector = np.zeros(9, dtype=np.float32)
    vector[indices] = 1.0
    return vector, indices
...
input_trials.append(geometry.copy())
```

iii. The notes say the task asks for compositional environment geometry, so nine blocked/not-blocked flags are a more direct representation than a one-hot session-condition label.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw `position` field for each day/session.

ii.
```python
position = record["position"]
...
result = convert_session(
    trace[day],
    position[day],
    blocked[day],
    animal,
    subject_index,
    day,
    str(envs[day]),
)
```

iii. The AI’s notes state that `position` is already synchronized to the calcium-event stream at 30 Hz and should therefore be used directly for the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each 60-second trial, the AI mean-pools x and y coordinates across non-overlapping 3-frame windows, converts each pooled coordinate to a 3-by-3 spatial bin using 25-cm bins, clips indices to stay within `0..2`, and flattens the 2D bin to a single class label.

ii.
```python
def pool_position(position_trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = position_trial.reshape(2, POOLED_BINS_PER_TRIAL, POOL_FRAMES).mean(axis=2)
    xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
    np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
    labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
    return pooled, labels
```

iii. The AI justifies the pooling as part of the same 100-ms decoder binning used for the neural data. It also says an orientation sanity check against blocked partitions motivated flattening in canonical `(y, x)` row-major order.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into 9 categories by dividing the 75-cm arena into three 25-cm bins along each axis and flattening the result with `label = y_bin * 3 + x_bin`. Values are clipped to the valid outer bins at the arena boundary.

ii.
```python
xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
```

iii. The notes say the task explicitly requires a 3 x 3 output, and that exact 75-cm boundary samples should remain synchronized and therefore be clipped into the outermost bin rather than dropped.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned trial-by-trial using the same source-frame boundaries. Within each trial, both streams are converted into the same 600 non-overlapping 3-frame windows, so output labels and neural features share the same time bins.

ii.
```python
for trial in range(n_trials):
    start = trial * SOURCE_FRAMES_PER_TRIAL
    stop = start + SOURCE_FRAMES_PER_TRIAL
    _, labels = pool_position(position_day[:, start:stop])
    neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
    ...
    neural_trials.append(neural_trial)
    output_trials.append(labels)
```

iii. The notes and trajectory say the AI checked raw-versus-pooled overlays and blocked-partition occupancy to confirm that there was no temporal shift and that the global orientation transform was correct.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly fails fast rather than imputing. It rejects inconsistent shapes, non-finite position values, invalid blocked indices, sessions with no valid cells, sessions with fewer than two complete trials, and out-of-range output labels. It handles incomplete tail frames by dropping them and handles exact boundary positions by clipping them into the outer spatial bin.

ii.
```python
if not np.allclose(values, np.round(values)):
    raise ValueError(f"Non-integer blocked indices: {values}")
if any(v < 0 or v > 8 for v in indices):
    raise ValueError(f"Blocked index outside 0..8: {indices}")
...
if trace_day.shape[1] != position_day.shape[1]:
    raise ValueError(f"{animal} day {day}: trace and position frame counts differ")
if not np.isfinite(position_day).all():
    raise ValueError(f"{animal} day {day}: position contains NaN/Inf")
...
n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
...
np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
```

iii. The notes explicitly discuss two edge-case policies: keep exact boundary samples aligned by clipping 75-cm coordinates into the outer bin, and preserve the tiny number of pooled samples that fall in blocked partitions rather than silently editing the behavior stream.

## 6-a. What are the most time-consuming steps of the code?

i. According to the AI’s notes, the slowest steps are loading each animal bundle, Gaussian smoothing and pooling across all retained cells and trials, optional plotting, and writing the very large final pickle.

ii.
```python
bundle = joblib.load(DATA_DIR / animal)
...
smoothed = gaussian_filter1d(
    selected,
    sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
    axis=2,
    mode="reflect",
)
...
with args.outpicklefile.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 6 and Step 7 of `CONVERSION_NOTES.md`, the AI calls out native bundle loading, batched filtering, and writing a multi-gigabyte pickle as the dominant costs; it also notes that sample-mode plots "dominate" per-session time when enabled.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI explicitly says it avoided slow per-neuron, per-frame, and per-trial filtering loops by vectorizing smoothing and pooling across the full `(neuron, trial, frame)` tensor. The remaining explicit trial loop, which calls `pool_position` and appends trial outputs one at a time, is a piece that still could be vectorized further.

ii.
```python
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
smoothed = gaussian_filter1d(
    selected,
    sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
    axis=2,
    mode="reflect",
)
pooled_neural = smoothed.reshape(
    valid_cells.size, n_trials, POOLED_BINS_PER_TRIAL, POOL_FRAMES
).mean(axis=3, dtype=np.float32)
...
for trial in range(n_trials):
    start = trial * SOURCE_FRAMES_PER_TRIAL
    stop = start + SOURCE_FRAMES_PER_TRIAL
    _, labels = pool_position(position_day[:, start:stop])
    ...
```

iii. The Step 6 notes say "naive per-neuron/per-frame Python loops and per-trial SciPy calls would add substantial overhead," and describe the batched filter call as an intentional speedup.

## 6-c. What processing does the code repeat multiple times?

i. The core conversion pipeline does not intentionally repeat much heavy computation beyond the necessary per-trial output-label generation. The main repeated extra work is optional: `plot_processing` recomputes smoothing/pooling for visualization, and geometry is derived again when plots are requested.

ii.
```python
if args.show_processing and n_plotted < 2:
    geometry, _ = blocked_vector(blocked[day])
    plot_path = plot_processing(
        trace[day],
        position[day],
        valid_cells,
        neural_trials,
        output_trials,
        geometry,
        session_info,
    )
```

iii. The notes frame this repeated work as validation and visualization rather than part of the essential converter. They do not identify a large repeated step in the final production path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional plotting path and some of the extensive provenance/validation machinery are not used by downstream decoder training. `plot_processing` recomputes smoothed traces and pooled positions solely to make figures, and the per-session metadata is kept for auditability rather than decoding.

ii.
```python
parser.add_argument(
    "--show-processing",
    action="store_true",
    help="save processing_<session_id>.png for up to two sessions",
)
...
pooled_xy, labels = pool_position(position_day[:, :SOURCE_FRAMES_PER_TRIAL])
...
"session_info": [],
...
data["metadata"]["session_info"].append(session_info)
```

iii. The notes say the figures and spot checks were added as sanity checks, not because they are needed downstream. The metadata is described as provenance for independent verification.
