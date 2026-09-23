# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from extensionless joblib-format files in `/app/data/`. Each file contains a dictionary keyed by animal ID, with sub-fields `trace`, `position`, `blocked`, `envs`, etc. The seven animals are hard-coded in an `ANIMALS` list. Data is loaded one animal at a time to manage memory.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
# ...
source = joblib.load(DATA_DIR / animal)[animal]
```

iii. The AI noted that both `.mat` and joblib files are present in the data directory, and that `joblib` is the reference code's default loading method (used in `load_dat` in `utils.py`). Loading one animal at a time bounds memory usage.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to a separate file in the data directory. The seven animal IDs are hard-coded in the `ANIMALS` list and used as subject identifiers.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
# ...
for animal_index, animal in enumerate(ANIMALS):
    source = joblib.load(DATA_DIR / animal)[animal]
```

iii. Each data file contains all recording sessions for one animal. The filename/key serves as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple recording days. The AI iterates over the day index within each animal's data. Each day becomes a separate session in the output.

ii.
```python
ndays = source["trace"].shape[0]
for day in range(ndays):
    trace = source["trace"][day]
    position = source["position"][day]
```

iii. The trace and position arrays have a day dimension (first axis). Each day's recording is a separate continuous session.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into non-overlapping 60-second segments. After 3-frame pooling to 10 Hz, each trial is 600 time bins. Remainder frames that don't fill a complete 60-second segment are discarded.

ii.
```python
RAW_TRIAL_FRAMES = SOURCE_HZ * TRIAL_SECONDS  # 1800
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)  # 600
# ...
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
# ...
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
```

iii. Per the task instructions, trials are 1-minute segments. The AI computes the number of complete trials from raw frame count (dividing by 1800), discards the terminal partial minute, then slices the pooled arrays at 600-bin boundaries.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are retained. Only the terminal partial minute (if any) is dropped.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES state: "No trial rejection is described. Each session is a continuous 40-minute free-exploration recording." The paper does not describe trial-level quality criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binarized calcium-transient rising phase events (0/1 values, with NaN for unregistered cells).

ii.
```python
trace = source["trace"][day]
registered = np.isfinite(trace[:, 0])
```

iii. The CONVERSION_NOTES explain: "Native neural data are already processed rise-extracted calcium event traces, with 1 denoting a significant event and NaN denoting a cell not registered that day."

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps matching the reference paper's decoder pipeline: (1) Gaussian smoothing with sigma=3 source frames along the time axis, and (2) non-overlapping 3-frame mean pooling, reducing from 30 Hz to 10 Hz (100 ms bins). This is applied to the full session before trial slicing to avoid edge artifacts.

ii.
```python
SMOOTH_SIGMA_FRAMES = 3.0
POOL_FRAMES = 3
# ...
def pool_neural(trace, registered, n_raw_keep):
    source = np.asarray(trace[registered], dtype=np.float32)
    smoothed = np.empty_like(source, dtype=np.float32)
    gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
    del source
    pooled = smoothed[:, :n_raw_keep].reshape(
        smoothed.shape[0], -1, POOL_FRAMES
    ).mean(axis=2, dtype=np.float32)
    return np.ascontiguousarray(pooled, dtype=np.float32)
```

iii. The AI justified this by citing the reference code's `fit_decoder`/`test_decoder` functions, which Gaussian-smooth traces (sigma=3 frames) and apply non-overlapping 3-frame average pooling before decoding. The CONVERSION_NOTES say: "Temporal bins are 100 ms: The reference position decoder Gaussian-smooths trace events with sigma equal to 3 frames and applies non-overlapping 3-frame average pooling."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that are registered (finite/non-NaN) in each session are retained. Registration is checked at the first frame: if a neuron's first value is finite, it is included. All-NaN neurons (not recorded that day) are excluded.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()
# ...
source = np.asarray(trace[registered], dtype=np.float32)
```

iii. The CONVERSION_NOTES state that registered cells are defined as those with finite traces, and that NaN denotes cells not recorded on a given day. Place-cell labels and running/activity filters from the reference decoder are NOT applied, because they would break contiguous one-minute trial timing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous free exploration with no stimulus events. Trials are consecutive 60-second segments starting from the beginning of the session.

ii. N/A (no alignment code; trials are simply consecutive segments)

iii. The CONVERSION_NOTES confirm: "Sessions were continuous free exploration; one-minute trials are required downstream." There is no stimulus onset or behavioral event to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 100 ms time bins (10 Hz), obtained by 3-frame mean pooling from the native 30 Hz. Each trial has 600 time bins (60 seconds / 100 ms).

ii.
```python
SOURCE_HZ = 30
POOL_FRAMES = 3
TARGET_HZ = SOURCE_HZ / POOL_FRAMES  # 10 Hz
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)  # 600
```

iii. The AI chose 100 ms bins to match the reference decoder's temporal preprocessing. The CONVERSION_NOTES state: "100 ms bins exactly reproduce reference decoder temporal preprocessing."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the source data, which contains indices of blocked reward locations for each recording session.

ii.
```python
blocked = blocked_mask_x_y(source["blocked"][day])
```

iii. The `blocked` variable stores which of the 9 possible positions in a 3x3 grid are blocked (unavailable). A sentinel value of -1 indicates no positions are blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are placed into a 3x3 matrix in (y,x) row-major order, then transposed to (x,y) order to match the position class convention, then flattened to a 9-element binary vector. Positions with index -1 (sentinel for "none blocked") are filtered out. The result is 1 for blocked and 0 for accessible. The vector is static (same for all trials in a session).

ii.
```python
def blocked_mask_x_y(blocked_entry):
    source_indices = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    source_y_x = np.zeros((N_POSITION_BINS, N_POSITION_BINS), dtype=np.float32)
    valid = source_indices[source_indices >= 0].astype(np.int64)
    if valid.size:
        source_y_x.reshape(-1)[valid] = 1.0
    return source_y_x.T.reshape(-1).copy()  # transpose (y,x) -> (x,y)
```

iii. The CONVERSION_NOTES explain: "Transpose source blocked matrices to `(x_bin,y_bin)` before flattening, so input dimension `x*3+y` matches output class." This ensures the blocked mask dimensions align with the position class encoding (`3*x_bin + y_bin`).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint.

ii.
```python
position = source["position"][day]
# shape: (2, n_timepoints) where [0] is x, [1] is y
```

iii. The position variable records the animal's head location via DeepLabCut tracking at 30 Hz in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The raw position is pooled using non-overlapping 3-frame mean bins (same as neural data), then each axis (x, y) is discretized into 3 equal 25-cm bins spanning 0-75 cm. Values are clipped to valid range [0, 2].

ii.
```python
def pool_position(position, n_raw_keep):
    trimmed = np.asarray(position[:, :n_raw_keep], dtype=np.float64)
    pooled_xy = trimmed.reshape(2, -1, POOL_FRAMES).mean(axis=2)
    xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
    np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
    classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
    return pooled_xy, classes
```

iii. Position is pooled to match the neural temporal resolution (10 Hz). Each axis is divided into 3 bins of 25 cm each. The class label is computed as `3*x_bin + y_bin`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each position axis is divided into 3 equal bins: [0, 25), [25, 50), [50, 75] cm. The bin index is computed via `floor(position / 25)` and clipped to [0, 2]. The final class label is `3*x_bin + y_bin`, giving 9 classes (0-8) in x-major order.

ii.
```python
POSITION_BIN_CM = ARENA_SIZE_CM / N_POSITION_BINS  # 25.0
# ...
xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
```

iii. The 3x3 grid with 25 cm bins gives a coarse spatial discretization as required by the task instructions (3x3 = 9 spatial bins). The x-major ordering (`3*x + y`) was chosen for consistency with the transposed blocked mask.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are acquired at the same 30 Hz rate and share frame indices. Both are pooled using the same non-overlapping 3-frame bins before trial slicing, ensuring identical temporal alignment.

ii.
```python
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
pooled_neural = pool_neural(trace, registered, n_raw_keep)
pooled_xy, position_classes = pool_position(position, n_raw_keep)
assert pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS
```

iii. Both streams are trimmed to the same number of raw frames (`n_raw_keep`), pooled identically, and sliced at the same trial boundaries. The assertion verifies alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with NaN values (not registered in a session) are excluded. Terminal frames that don't complete a 60-second trial are dropped. The number of dropped frames is recorded per session in metadata. Position values at the arena boundary (75 cm) are clipped into the last bin.

ii.
```python
registered = np.isfinite(trace[:, 0])
# ...
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
# dropped frames recorded:
"dropped_terminal_frames": int(trace.shape[1] - n_raw_keep),
```

iii. The CONVERSION_NOTES document: "Nominal vs acquired duration: Paper says 40 min, while three subjects have 39 min 55.5 s and others slightly exceed 40 min. Resolution: preserve every exact 60 s segment, drop only incomplete tails, and record dropped frames."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading the joblib files (I/O bound), and (2) Gaussian smoothing of neural traces per session. The AI reported full conversion completed in ~194 seconds before serialization.

ii.
```python
load_start = time.perf_counter()
source = joblib.load(DATA_DIR / animal)[animal]
print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
```

iii. The CONVERSION_NOTES detail timing: loading takes ~10s per animal, processing ~0.84s/session. The pickle write adds ~6s for the 6.2 GiB output.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials to slice arrays, but this is a lightweight indexing operation. The Gaussian smoothing and pooling are already vectorized over neurons and time.

ii.
```python
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
```

iii. The AI chose not to vectorize the trial loop since it's just array slicing. The expensive operations (smoothing, pooling) are already vectorized using scipy and numpy reshape/mean.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session's data is processed once (smooth, pool, discretize, split). The blocked mask is computed once per session and copied for each trial.

ii.
```python
input_trials.append(blocked.copy())  # same vector copied per trial
```

iii. The AI processes full sessions before trial slicing to avoid repeating convolution and to prevent edge artifacts at trial boundaries.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `pooled_xy` (pooled continuous position coordinates) which is used for plotting (`--show-processing` mode) but not included in the final output. The code also stores extensive session metadata (registered cell indices, environment names, frame counts) that is not used by the decoder.

ii.
```python
pooled_xy, position_classes = pool_position(position, n_raw_keep)
# pooled_xy is only used in plot_processing(), not in the output data
```

iii. The pooled continuous position is needed for the processing visualization but adds minimal computational cost. The metadata provides auditability.
