# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-converted joblib files (one per animal) rather than the raw `.mat` files. Each joblib file is a dictionary keyed by animal ID, containing `trace`, `position`, `blocked`, and `envs` arrays. The seven animal IDs are hardcoded in the script.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
DATA_DIR = Path("/app/data")
...
bundle = joblib.load(DATA_DIR / animal)
record = bundle[animal]
trace = record["trace"]
position = record["position"]
blocked = record["blocked"]
envs = np.asarray(record["envs"]).reshape(-1)
```

iii. The AI noted that the reference code (`load_dat` in `utils.py`) uses joblib as its primary loader. Joblib files contain the same arrays as the `.mat` files but load faster. The AI verified that both formats contain identical data.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject. The seven animal IDs are hardcoded in `ANIMALS` list and iterated over.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for subject_index, animal in enumerate(ANIMALS):
    bundle = joblib.load(DATA_DIR / animal)
```

iii. The repository README identifies seven animals. The AI hardcodes them rather than discovering them via glob.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject becomes a separate session. The `trace` array has shape `(n_days, n_cells, n_frames)`, so iterating over the first axis yields sessions.

ii.
```python
for day in range(trace.shape[0]):
    result = convert_session(
        trace[day], position[day], blocked[day],
        animal, subject_index, day, str(envs[day]),
    )
```

iii. Each animal-day is a natural session unit, with a stable neuron population and single geometry.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive non-overlapping 1800-source-frame (60-second at 30 Hz) segments. Incomplete tails are discarded. After 3-frame pooling, each trial has 600 time bins.

ii.
```python
SOURCE_FRAMES_PER_TRIAL = SOURCE_FPS * TRIAL_SECONDS  # 1800
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES  # 600
...
n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL
...
selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
```

iii. The instructions specify 1-minute trials. The AI splits based on 1800 source frames then pools 3 frames to get 600 bins per trial.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 complete trials are rejected (explicit check). No other trial-level filtering is applied.

ii.
```python
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: fewer than two complete trials")
```

iii. The instructions require at least two trials per session for decoder evaluation. In practice all sessions have 39-40 trials so this never triggers.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary rising-phase calcium events (values exactly 0 or 1, with NaN for unregistered cells).

ii.
```python
trace = record["trace"]
...
trace_day = trace[day]  # shape (n_cells, n_frames)
```

iii. The `trace` variable contains pre-processed binarized calcium events. No delta-F/F recomputation is needed.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps from the reference paper's decoder: (1) Gaussian smoothing with sigma=3 source frames (using `scipy.ndimage.gaussian_filter1d` with reflect mode), and (2) non-overlapping 3-frame mean pooling. This produces 600 time bins per trial at 100ms resolution. Processing is done per-trial independently to avoid train/validation leakage.

ii.
```python
NEURAL_SMOOTH_SIGMA_FRAMES = 3.0
POOL_FRAMES = 3
...
selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
smoothed = gaussian_filter1d(
    selected, sigma=NEURAL_SMOOTH_SIGMA_FRAMES, axis=2, mode="reflect",
)
pooled_neural = smoothed.reshape(
    valid_cells.size, n_trials, POOLED_BINS_PER_TRIAL, POOL_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES state this matches the reference decoder's `fit_decoder`/`test_decoder` functions which apply sigma-3 Gaussian smoothing and 3-frame average pooling (`AvgPool1d(3)`). The AI chose to apply this to create a processed dataset matching the reference paper's decoder pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with NaN at the first frame (indicating unregistered cells for that day) are excluded. Only finite cells are retained.

ii.
```python
valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
if valid_cells.size == 0:
    raise ValueError(f"{animal} day {day}: no registered cells")
```

iii. The AI notes that absent CellReg cells are NaN for their entire day, so checking the first frame is sufficient. No additional place-cell or activity-threshold filtering is applied; the AI argues these are decoder-specific feature selection, not upstream QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous free exploration, and trials are artificial 60-second segments starting from frame 0. Neural and position data share the same 30 Hz frame axis.

ii. N/A (alignment is implicit through shared frame indices)

iii. The AI notes that position and trace are synchronized at 30 Hz with identical frame axes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 3-frame mean pooling, changing the resolution from 33.33ms (30 Hz) to 100ms (10 Hz). Each trial has 600 time bins instead of 1800.

ii.
```python
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES  # 600
...
"time_bin_size": 100.0,  # ms
```

iii. The AI justifies this by matching the reference paper's decoder which uses 3-frame temporal pooling (`temporal_bin_size=3` in `fit_decoder`).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable, which contains indices of blocked reward positions for each recording day.

ii.
```python
blocked = record["blocked"]
...
blocked_entry = blocked[day]
```

iii. The `blocked` variable stores which of the 9 possible positions were blocked during each session.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector (1=blocked, 0=accessible). If no positions are blocked (indicated by `[-1]`), the vector is all zeros. The vector is static per trial.

ii.
```python
def blocked_vector(blocked_entry) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    if values.size == 1 and values[0] == -1:
        indices: list[int] = []
    else:
        indices = [int(v) for v in values]
    vector = np.zeros(9, dtype=np.float32)
    vector[indices] = 1.0
    return vector, indices
```

iii. One-hot encoding allows each blocked position to be treated independently. The encoding is per-session (constant across trials).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates of the animal in the arena at each frame.

ii.
```python
position = record["position"]
...
position_day = position[day]  # shape (2, n_frames)
```

iii. The position variable records the animal's location in a 75x75 cm open field arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first mean-pools position over the same 3-frame windows used for neural data, then discretizes the pooled coordinates into a 3x3 grid using `floor(coord/25cm)` with clipping to [0, 2].

ii.
```python
def pool_position(position_trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = position_trial.reshape(2, POOLED_BINS_PER_TRIAL, POOL_FRAMES).mean(axis=2)
    xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
    np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
    labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
    return pooled, labels
```

iii. The AI pools position to match the 3-frame temporal pooling applied to neural data, ensuring synchronized time bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is discretized into a 3x3 grid (9 classes) using 25cm bin edges. Each axis is divided into 3 equal bins (0-25, 25-50, 50-75 cm). The grid label is computed as `y_bin * 3 + x_bin`.

ii.
```python
SPATIAL_BIN_CM = 25.0
N_SPATIAL_BINS = 3
...
xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
```

iii. The AI uses `floor(coord/25)` with clipping, while validating orientation with an exhaustive 8-transform analysis to confirm `y*3+x` matches the blocked partition ordering.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is mean-pooled over the same 3-frame windows as neural data, ensuring 1:1 temporal correspondence. Both produce 600 bins per trial.

ii.
```python
for trial in range(n_trials):
    start = trial * SOURCE_FRAMES_PER_TRIAL
    stop = start + SOURCE_FRAMES_PER_TRIAL
    _, labels = pool_position(position_day[:, start:stop])
    neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
```

iii. Both neural and position data use identical 3-frame pooling windows, maintaining exact temporal alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons absent for a day (all-NaN) are excluded by checking the first frame. Incomplete session tails are discarded. Position values at exactly 75cm are clipped to bin 2 (the outer bin). Sessions with fewer than 2 trials are rejected. The AI also validates that position contains no NaN/Inf and that neural data has no NaN after filtering.

ii.
```python
valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
...
if not np.isfinite(position_day).all():
    raise ValueError(...)
...
n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL
# Only used frames are processed; tail is discarded
...
np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
```

iii. The AI investigated 152 pooled samples (0.003%) that appeared in blocked bins and determined they were valid boundary/pose observations in the raw data, not conversion errors.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step (~9s per animal). The Gaussian smoothing and pooling operations are fast due to vectorized batch processing. Full conversion takes ~168 seconds.

ii.
```python
load_start = time.perf_counter()
bundle = joblib.load(DATA_DIR / animal)
print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f}s: ...")
```

iii. CONVERSION_NOTES report per-animal load times and total conversion time of 167.71s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for position pooling and output construction could potentially be vectorized. However, the neural smoothing/pooling is already vectorized across all trials in a single SciPy call.

ii.
```python
# This loop processes position trial-by-trial
for trial in range(n_trials):
    start = trial * SOURCE_FRAMES_PER_TRIAL
    stop = start + SOURCE_FRAMES_PER_TRIAL
    _, labels = pool_position(position_day[:, start:stop])
```

iii. The AI notes that the main bottleneck is I/O, not computation. The neural processing is batched across all trials.

## 6-c. What processing does the code repeat multiple times?

i. The `blocked_vector` function is called twice for sessions where processing plots are generated (once in `convert_session` and once in `plot_processing`). This is minimal overhead.

ii.
```python
# In main loop:
result = convert_session(trace[day], position[day], blocked[day], ...)
...
if args.show_processing and n_plotted < 2:
    geometry, _ = blocked_vector(blocked[day])  # duplicate call
```

iii. The duplication only occurs for at most 2 sessions when `--show-processing` is enabled.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing (sigma=3 frames) and 3-frame mean pooling of both neural and position data is additional processing not present in the reference solution. This changes the temporal resolution from 1800 to 600 bins per trial. The position mean-pooling before discretization is also extra processing compared to direct frame-by-frame discretization.

ii.
```python
smoothed = gaussian_filter1d(selected, sigma=NEURAL_SMOOTH_SIGMA_FRAMES, axis=2, mode="reflect")
pooled_neural = smoothed.reshape(...).mean(axis=3, dtype=np.float32)
```

iii. The AI justifies this as matching the reference paper's decoder pipeline. However, the instructions say to match reference processing "when applicable" and the downstream decoder is a different neural network, not the paper's Bayesian decoder.
