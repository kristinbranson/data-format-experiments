# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the pre-processed joblib files (not the `.mat` files). Each subject has a corresponding joblib file in the data directory (e.g., `QLAK-CA1-08`). These are loaded using `joblib.load()`, which returns a dictionary keyed by subject name containing `trace`, `position`, `envs`, and `blocked` fields.

ii.
```python
source_path = data_dir / subject
wrapped = joblib.load(source_path)
source = wrapped[subject]
traces = np.asarray(source["trace"])
positions = np.asarray(source["position"])
environments = np.asarray(source["envs"]).reshape(-1)
```

iii. The AI noted: "The source data already contains the paper's final binary rising-phase events and aligned x-y tracking, so the key choices are session-level cell registration, temporal downsampling, one-minute segmentation, and translating each environment's blocked sectors into decoder inputs." The AI chose joblib files because they contain the same data in a more convenient format.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as a list of 7 animal IDs. Each subject corresponds to one joblib file. The subject index is tracked by enumerating over the SUBJECTS list.

ii.
```python
SUBJECTS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
for subject_number, subject in enumerate(SUBJECTS):
    source_path = data_dir / subject
```

iii. The AI identified all 7 subjects from the data directory and hardcoded their names in sorted order.

## 1-c. How are the data split into sessions?

i. Each joblib file contains a 3D array for traces with shape `(n_days, n_cells, n_frames)`. Each day (index along axis 0) becomes a separate session.

ii.
```python
traces = np.asarray(source["trace"])
for day in range(traces.shape[0]):
    position = positions[day]
    ...
    subject_idx.append(subject_number)
```

iii. Each recording day is one session. The AI iterates over the first axis of the traces array, which indexes days/sessions.

## 1-d. How are the data split into trials?

i. Trials are 60-second non-overlapping segments. At 30 Hz, each trial is 1800 raw frames. After rebinning to 1-second bins, each trial has 60 time bins. The remainder frames that don't fill a complete trial are discarded.

ii.
```python
RAW_FRAMES_PER_TRIAL = TRIAL_SECONDS * SOURCE_FPS  # 1800
n_trials = total_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
...
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. Per the instructions, trials are defined as 60-second non-overlapping segments of the continuous recording. Incomplete tails are discarded.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 complete trials are rejected (raises an error). No other trial-level quality filtering is performed.

ii.
```python
if n_trials < 2:
    raise ValueError(f"Fewer than two complete trials for {subject}, day {day}")
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib file, which contains binary rising-phase calcium event traces with shape `(n_days, n_cells, n_frames)`.

ii.
```python
traces = np.asarray(source["trace"])
selected = traces[day, cell_mask, :].astype(np.float32, copy=True)
```

iii. The AI identified these as the paper's pre-processed binary rising-phase events (1 = significant calcium event, 0 otherwise).

## 2-b. How is the `neural` data processed?

i. The neural data undergoes three processing steps:
1. Cell filtering based on activity (see 2-c)
2. Gaussian temporal smoothing with sigma=3 frames
3. Temporal rebinning via average pooling into 1-second (30-frame) non-overlapping bins

ii.
```python
selected = traces[day, cell_mask, :].astype(np.float32, copy=True)
smoothed = np.empty_like(selected)
gaussian_filter1d(selected, sigma=3, axis=1, output=smoothed)
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The AI stated this follows the repository's `decode_position_within` function which applies `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` with `temporal_bin_size=3` before average pooling (via `AvgPool1d`). The AI explicitly chose to replicate this smoothing and pool to 1-second bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells must have more than 5 events while the animal is moving faster than 5 cm/s. Speed is estimated using a 5-frame Gaussian-filtered instantaneous velocity. Cells not registered on a given day (NaN) are implicitly excluded because NaN comparisons fail.

ii.
```python
def moving_mask(position: np.ndarray) -> np.ndarray:
    speed = np.zeros(position.shape[1], dtype=np.float64)
    instantaneous = np.linalg.norm(np.diff(position, axis=1) * SOURCE_FPS, axis=0)
    speed[1:] = gaussian_filter1d(instantaneous, sigma=5)
    return speed > 5.0

moving = moving_mask(position)
events_while_moving = np.sum(traces[day][:, moving], axis=1)
cell_mask = events_while_moving > 5
```

iii. The AI noted: "As in `decode_position_within` in the paper repository, cells must have more than five events while the animal is moving faster than 5 cm/s. The speed estimate uses the repository's 5-frame Gaussian filter." This matches the reference code's `cell_threshold=5` and `v_thresh=5` parameters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The recording is continuous and trials are consecutive 60-second segments starting from the beginning of the session. Neural data is split into trials after smoothing and rebinning.

ii.
```python
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
neural.append(
    [np.ascontiguousarray(binned_neural[:, trial, :]) for trial in range(n_trials)]
)
```

iii. There is no stimulus event to align to. The metadata records `off_start=0.0` and `off_end=60.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from native 30 Hz (33.33 ms per frame) to 1-second bins (1000 ms). This is done by averaging 30 consecutive frames. Each trial has 60 time bins.

ii.
```python
SOURCE_FPS = 30
TIME_BIN_FRAMES = 30
TIME_BIN_MS = 1000.0
TIME_BINS_PER_TRIAL = int(TRIAL_SECONDS * SOURCE_FPS / TIME_BIN_FRAMES)  # 60

binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The AI stated: "We retain that smoothing and pool to non-overlapping 1 s bins. The coarser downstream bin is appropriate for these 40-minute recordings and still preserves the paper's event-rate representation while making the common neural decoder tractable."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the joblib file, which contains a list of arrays indicating which of the 9 positions are blocked for each session.

ii.
```python
geometry, blocked_bins = blocked_geometry(source["blocked"][day])
```

iii. The `blocked` field stores indices of blocked reward locations for each recording day.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are placed into a 3x3 matrix in y-major order (matching the source data), then the matrix is transposed to x-major order to match the position class convention (`3*x_bin + y_bin`). The result is flattened to a 9-element float32 vector where 1=blocked and 0=open. If no positions are blocked (value -1), the vector is all zeros. The vector is static per trial.

ii.
```python
def blocked_geometry(blocked_for_day: object) -> tuple[np.ndarray, list[int]]:
    raw = np.asarray(blocked_for_day[0]).reshape(-1)
    blocked_yx = np.zeros((N_SPATIAL_BINS, N_SPATIAL_BINS), dtype=np.float32)
    for value in raw:
        idx = int(value)
        if idx >= 0:
            blocked_yx.flat[idx] = 1.0
    blocked_xy = blocked_yx.T
    vector = blocked_xy.reshape(-1)
    return vector, np.flatnonzero(vector).astype(int).tolist()
```

iii. The AI verified that "The geometry field's indexing is transposed relative to the stored (x, y) position arrays; I confirmed this from bins with zero occupancy." The transpose ensures blocked indices align with the position class labeling convention.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the joblib file, which contains 2D coordinates `(x, y)` of the animal in the 75x75 cm arena. Shape is `(n_days, 2, n_frames)`.

ii.
```python
positions = np.asarray(source["position"])
position = positions[day]  # (2, n_frames)
```

iii. The position field records the animal's head location tracked by DeepLabCut.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first averaged within each 1-second (30-frame) temporal bin, then discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal 25 cm bins. The class label is `3*x_bin + y_bin`. Boundary values (exactly 75 cm) are clipped to the outermost bin.

ii.
```python
def position_classes(position: np.ndarray, used_frames: int) -> np.ndarray:
    mean_position = position[:, :used_frames].reshape(
        2, -1, TIME_BIN_FRAMES
    ).mean(axis=2)
    xy_bin = np.floor(mean_position / SPATIAL_BIN_CM).astype(np.int64)
    xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. The AI chose `floor(position / 25)` for bin assignment, which gives bins [0, 25), [25, 50), [50, 75+]. Clipping handles the boundary case at 75 cm.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). Each spatial dimension is divided into 3 equal bins of 25 cm each using `floor(position / 25)`, then clipped to [0, 2]. The final class is `3 * x_bin + y_bin`.

ii. (Same as 4-b code snippet above)

iii. This creates an x-major ordering for the 9 position classes, matching the transposed blocked geometry encoding.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is averaged within the same 1-second temporal bins used for neural data, ensuring frame-level alignment. Both are split into trials using the same frame indices.

ii.
```python
# Position is binned to 1-second bins matching neural
mean_position = position[:, :used_frames].reshape(2, -1, TIME_BIN_FRAMES).mean(axis=2)
# Then reshaped into trials
classes = position_classes(position, used_frames).reshape(n_trials, TIME_BINS_PER_TRIAL)
```

iii. Both neural and position data use the same `used_frames` and are split identically into trials of `TIME_BINS_PER_TRIAL` bins.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Cells not registered on a given day appear as NaN in the trace array. These are implicitly excluded by the activity filter (NaN comparisons return False, so `np.sum` of NaN values doesn't exceed the threshold of 5). Remainder frames that don't fill a complete 60-second trial are discarded. Sessions with zero qualifying cells or fewer than 2 trials raise errors.

ii.
```python
# NaN cells implicitly excluded:
events_while_moving = np.sum(traces[day][:, moving], axis=1)
cell_mask = events_while_moving > 5  # NaN sums don't exceed 5

# Remainder frames discarded:
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
```

iii. The AI commented: "np.sum deliberately matches the reference implementation: a NaN from an unregistered cell makes the comparison false."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large joblib files (each 68-134 MB), which involves deserialization and decompression. The Gaussian smoothing on full trace arrays is also relatively expensive.

ii. N/A

iii. The full conversion took roughly 3 minutes across 7 animals, dominated by I/O and the smoothing step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `blocked_geometry` function loops over blocked indices to set values in a matrix, which could be vectorized with fancy indexing. However, the loop is over at most 9 values so the impact is negligible.

ii.
```python
for value in raw:
    idx = int(value)
    if idx >= 0:
        blocked_yx.flat[idx] = 1.0
```

iii. This is a minor loop with negligible performance impact.

## 6-c. What processing does the code repeat multiple times?

i. The code loads each subject's data only once. The `geometry.copy()` is called per trial but this is a 9-element array copy, which is trivial.

ii.
```python
decoder_input.append([geometry.copy() for _ in range(n_trials)])
```

iii. No significant repeated processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `blocked_bins` (list of which bins are blocked) in the `blocked_geometry` function, but this is only stored in `session_info` metadata and not used by the decoder. The Gaussian smoothing and temporal rebinning add processing that the reference solution does not perform, though the AI justified these as matching the paper's decoder pipeline.

ii.
```python
geometry, blocked_bins = blocked_geometry(source["blocked"][day])
# blocked_bins is only stored in session_info metadata
```

iii. The `blocked_bins` list is informational metadata only.
