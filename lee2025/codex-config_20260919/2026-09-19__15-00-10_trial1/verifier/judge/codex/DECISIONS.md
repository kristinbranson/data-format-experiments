# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob `.mat` files. It uses a hard-coded list of seven animal IDs, loads one extensionless joblib file per animal with `joblib.load`, extracts the top-level animal dictionary, and then iterates through all days/sessions in `source["trace"]`. Trial splitting happens later inside the per-session loop.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_index, animal in enumerate(ANIMALS):
    ...
    source = joblib.load(DATA_DIR / animal)[animal]
    ...
    ndays = source["trace"].shape[0]
    for day in range(ndays):
        ...
```

iii. In `CONVERSION_NOTES.md`, the AI says the reference code defaults to joblib loading and that sampled MATLAB/joblib values matched, so it treated sequential joblib loading as equivalent while reducing memory pressure.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs in `ANIMALS`. Each loaded animal contributes one subject label, and `subject_idx` stores the animal index for each session.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_index, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(animal_index)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The notes say the seven animals are explicitly listed in the reference project and that session order should follow that canonical animal order.

## 1-c. How are the data split into sessions?

i. Each recording day for an animal is treated as one session. The AI uses the first dimension of `source["trace"]` as the number of days/sessions and iterates over it.

ii.
```python
ndays = source["trace"].shape[0]
for day in range(ndays):
    trace = source["trace"][day]
    position = source["position"][day]
    ...
    session_id = f"{animal}_day{day:02d}"
```

iii. The notes say this matches the paper/code interpretation that each recording day is one continuous recording session.

## 1-d. How are the data split into trials?

i. Within each session, trials are artificial consecutive 60-second, non-overlapping chunks. The number of trials is `floor(n_frames / 1800)` in the raw 30 Hz stream. After the AI’s 3-frame pooling, each trial contains 600 time bins.

ii.
```python
SOURCE_HZ = 30
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = SOURCE_HZ * TRIAL_SECONDS
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)
...
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
...
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
    input_trials.append(blocked.copy())
    output_trials.append(position_classes[None, start:end].copy())
```

iii. The notes justify this as matching the instruction to build one-minute trials while dropping only the terminal incomplete segment.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies no per-trial quality rejection beyond omitting the terminal incomplete minute. All complete consecutive 60-second segments are retained.

ii.
```python
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
...
"trial_filter": "all complete consecutive 60-s segments; terminal partial minute omitted",
```

iii. The notes explicitly state that the dataset has continuous free-exploration sessions with no native trial rejection, so only incomplete tail segments are excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the per-day `trace` array in the loaded source dictionary.

ii.
```python
trace = source["trace"][day]
...
"neural_signal": "binarized significant calcium-transient rising phases",
```

iii. The notes describe `trace` as already-processed binary rise-event calcium activity, so the AI treated it as the neural source variable.

## 2-b. How is the `neural` data processed?

i. The AI selects registered cells, casts to `float32`, applies a Gaussian filter with sigma 3 source frames along time, then averages non-overlapping 3-frame bins, producing a pooled `float32` neural matrix.

ii.
```python
def pool_neural(trace: np.ndarray, registered: np.ndarray, n_raw_keep: int) -> np.ndarray:
    source = np.asarray(trace[registered], dtype=np.float32)
    smoothed = np.empty_like(source, dtype=np.float32)
    gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
    pooled = smoothed[:, :n_raw_keep].reshape(
        smoothed.shape[0], -1, POOL_FRAMES
    ).mean(axis=2, dtype=np.float32)
    return np.ascontiguousarray(pooled, dtype=np.float32)
```

iii. The notes justify this as matching the reference decoder preprocessing (`fit_decoder` / `test_decoder`) and as reducing output size versus preserving 30 Hz traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells registered in that day/session, detected by finiteness of the first timepoint, and asserts that all values for those cells are finite binary values. Unregistered all-NaN rows are removed.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()
assert np.isin(trace[registered], (0.0, 1.0)).all()
...
source = np.asarray(trace[registered], dtype=np.float32)
```

iii. The notes say the raw dataset uses NaN rows for cells not registered on a given day, and the AI intentionally kept all registered cells without additional place-cell or running-based filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. There is no biological event alignment. The AI defines the alignment event as the start of each consecutive 60-second recording segment and slices trials from that origin.

ii.
```python
"temporal_alignment_event": "start of each consecutive 60-second recording segment",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The notes say the sessions are continuous free exploration with no stimulus event, so segment start was used as the artificial alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins at 10 Hz, not the native 30 Hz sampling. The AI rebins by averaging each non-overlapping group of 3 source frames after neural smoothing, and it applies the same 3-frame pooling to position.

ii.
```python
SOURCE_HZ = 30
POOL_FRAMES = 3
TARGET_HZ = SOURCE_HZ / POOL_FRAMES
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)
...
"time_bin_size": 100.0,
"target_sampling_rate_hz": TARGET_HZ,
...
pooled = smoothed[:, :n_raw_keep].reshape(
    smoothed.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
```

iii. The notes explicitly justify 100 ms bins as reproducing the reference decoder’s preprocessing and lowering memory/runtime relative to 30 Hz export.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the per-day `blocked` entry in the source dictionary.

ii.
```python
blocked = blocked_mask_x_y(source["blocked"][day])
```

iii. The notes say `blocked` stores the blocked 3x3 partitions directly, while environment names in `envs` are kept only as metadata.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the raw blocked indices into a 9-element binary vector with `1` meaning blocked. It builds a 3x3 `(y, x)` mask, transposes it into `(x, y)` order to match its position class convention, flattens it, and reuses the same static vector for every trial in the session. The `-1` sentinel produces an all-zero vector.

ii.
```python
def blocked_mask_x_y(blocked_entry: object) -> np.ndarray:
    source_indices = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    source_y_x = np.zeros((N_POSITION_BINS, N_POSITION_BINS), dtype=np.float32)
    valid = source_indices[source_indices >= 0].astype(np.int64)
    if valid.size:
        source_y_x.reshape(-1)[valid] = 1.0
    return source_y_x.T.reshape(-1).copy()
...
blocked = blocked_mask_x_y(source["blocked"][day])
...
input_trials.append(blocked.copy())
```

iii. The notes justify the transpose as necessary to make input dimension order match the output class coordinate system, and they say occupancy checks showed blocked classes had zero occupancy under this convention.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the per-day `position` array in the source dictionary.

ii.
```python
position = source["position"][day]
...
pooled_xy, position_classes = pool_position(position, n_raw_keep)
```

iii. The notes describe `position` as the aligned 2D head-position stream acquired at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first averages position in non-overlapping 3-frame bins, then discretizes the pooled x and y coordinates into three equal 25 cm bins each, and finally converts each pooled timepoint into a single class index.

ii.
```python
def pool_position(position: np.ndarray, n_raw_keep: int) -> tuple[np.ndarray, np.ndarray]:
    trimmed = np.asarray(position[:, :n_raw_keep], dtype=np.float64)
    pooled_xy = trimmed.reshape(2, -1, POOL_FRAMES).mean(axis=2)
    xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
    np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
    classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
    return pooled_xy, classes
```

iii. The notes justify this as using the same temporal pooling as the neural stream and as implementing the task’s required 3x3 spatial discretization.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded at 25 cm and 50 cm, giving bins `[0,25)`, `[25,50)`, and `[50,75]` after clipping. The AI’s class ordering is `3 * x_bin + y_bin`, with output value names `x0_y0` through `x2_y2`.

ii.
```python
POSITION_BIN_CM = ARENA_SIZE_CM / N_POSITION_BINS
...
xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
...
"output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
```

iii. The notes say this ordering was chosen so the output class order matches the transposed blocked-mask input order.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI keeps position aligned with neural data by trimming both streams to the same retained raw frames, applying the same 3-frame non-overlapping pooling, and then slicing both pooled streams into the same 60-second trial boundaries.

ii.
```python
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
pooled_neural = pool_neural(trace, registered, n_raw_keep)
pooled_xy, position_classes = pool_position(position, n_raw_keep)
assert pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS
...
neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
output_trials.append(position_classes[None, start:end].copy())
```

iii. The notes repeatedly justify this as preserving exact neural/behavioral alignment while matching the AI’s chosen reference-decoder pooling.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes unregistered all-NaN cells by selecting only finite rows, asserts that retained neural and position values are finite, clips positions that would otherwise exceed the top spatial bin, interprets blocked value `-1` as “no blocked bins,” and drops only the terminal incomplete 60-second segment instead of padding.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert np.isfinite(trace[registered]).all()
assert np.isfinite(position).all()
...
valid = source_indices[source_indices >= 0].astype(np.int64)
...
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
...
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
```

iii. The notes explicitly mention NaN-based unregistered cells, safe clipping of rare 75 cm boundary values, the `-1` square sentinel, and dropping only partial tail segments.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies sequential source-file loading, full-session Gaussian filtering plus 3-frame pooling, and writing the multi-gigabyte pickle as the main time costs.

ii.
```python
source = joblib.load(DATA_DIR / animal)[animal]
...
gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
...
with temporary.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. `CONVERSION_NOTES.md` says source loading is I/O-heavy, session-wide filtering/pooling dominates processing, and serialization of a ~6.2 GiB pickle is another significant cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the heavy numeric work (`gaussian_filter1d`, reshape, mean), but it still uses Python loops for per-trial list construction and nested validation over every trial. Those loops could be reduced with batched reshaping or by deferring some validation to vectorized whole-session checks.

ii.
```python
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
    input_trials.append(blocked.copy())
    output_trials.append(position_classes[None, start:end].copy())
...
for neural, inputs, outputs in zip(neural_trials, input_trials, output_trials):
    assert neural.shape == (nneurons, TRIAL_TIMEPOINTS)
    ...
    output_counts += np.bincount(outputs[0], minlength=9)
```

iii. The notes emphasize that the AI intentionally vectorized filtering/pooling; it did not separately justify the remaining Python loops.

## 6-c. What processing does the code repeat multiple times?

i. The code avoids repeating the expensive smoothing step by processing full sessions once, but it still repeats some lighter work: it copies the same static blocked vector into every trial, slices pooled arrays trial-by-trial into lists, and then re-iterates over all trials during validation.

ii.
```python
for trial in range(ntrials):
    ...
    input_trials.append(blocked.copy())
...
for s, (neural_trials, input_trials, output_trials) in enumerate(
    zip(data["neural"], data["input"], data["output"])
):
    ...
    for neural, inputs, outputs in zip(neural_trials, input_trials, output_trials):
        ...
```

iii. The notes explicitly say the AI chose full-session processing once to avoid repeated per-trial convolution; they do not separately justify the smaller repeated bookkeeping work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded computation is `pooled_xy`: it is computed for every session even though downstream saved outputs only keep the categorical class labels. Optional plotting code and rich audit metadata are also outside the core decoder inputs/outputs.

ii.
```python
pooled_xy, position_classes = pool_position(position, n_raw_keep)
...
if show_processing and processing_plots < 2:
    plot_path = plot_processing(
        session_id, trace, registered, position, pooled_neural,
        pooled_xy, position_classes, blocked,
    )
...
session_info.append({
    "session_id": session_id,
    "subject": animal,
    "source_day_index": day,
    "environment": environment,
    ...
})
```

iii. The notes justify these extras as auditability and visual validation, not as necessary decoder features.
