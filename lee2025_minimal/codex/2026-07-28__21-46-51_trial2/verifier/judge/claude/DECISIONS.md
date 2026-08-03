# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-serialized files (one per animal) rather than `.mat` files. It uses a hardcoded list of 7 animal IDs (`ANIMALS`) and loads each with `joblib.load()`. Each file contains a dictionary with keys `trace`, `position`, `envs`, and `blocked`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    envs = flatten_envs(dat["envs"])
    trace = np.asarray(dat["trace"], dtype=np.float32)
    position = np.asarray(dat["position"], dtype=np.float32)
```

iii. The AI chose joblib files because the data directory contains both `.mat` files and joblib-serialized directories. The AI explored the code repository and found that the paper's code used joblib format, so it chose to load from that format directly.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 7 animal IDs. Each animal's data is loaded from a separate joblib file.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
```

iii. The AI inspected the data directory and the paper's code to identify all 7 animals. Using a hardcoded list ensures consistent ordering.

## 1-c. How are the data split into sessions?

i. Each animal's data contains a `trace` array with shape `(n_days, n_cells, T)`. Each day (first axis) becomes a separate session.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
# trace.shape = (n_days, n_cells, T)
for day in range(trace.shape[0]):
    day_trace = trace[day]
    day_pos = position[day]
```

iii. The AI confirmed from the paper code that each recording day is treated as one session. 207 total sessions were found, matching the paper's reported count.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into non-overlapping 60-second windows (1800 frames at 30 Hz). Trailing incomplete remainders are discarded. Sessions produce either 39 or 40 complete trials.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800
n_trials = T // TRIAL_FRAMES
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. Per the decoder task instructions, trials are defined as 1-minute segments. The AI confirmed this matches the instruction requirements.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second windows are kept. The AI does verify that each session has at least 2 complete trials (raising an error otherwise).

ii.
```python
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
```

iii. No trial filtering criteria were specified in the instructions or paper code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib files, which contains binary rising-phase event traces (binarized calcium transients) with shape `(n_days, n_cells, T)`.

ii.
```python
dat = load_animal(data_dir, animal)
trace = np.asarray(dat["trace"], dtype=np.float32)
day_trace = trace[day]  # (n_cells, T)
```

iii. The AI identified from the paper methods that all analyses use binarized rising phase of calcium transients. The trace data in the files already contains these binary events.

## 2-b. How is the `neural` data processed?

i. For each session/day, only neurons registered on that day are kept (unregistered neurons are all-NaN). The registered neuron data is cast to float32. No smoothing, re-deconvolution, or other processing is applied.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The AI noted that the source data already contains the binarized rising phase events, so no further signal processing was needed. Only NaN-neuron removal was necessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered on a given day (all-NaN) are removed. The AI also validates that registered neurons do not contain any NaNs, and that unregistered neurons are consistently NaN across all timepoints.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The AI used the first timepoint's NaN status to determine registration, then validated consistency. This is slightly different from the reference which checks all-NaN across all timepoints, but functionally equivalent given the data structure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of each session.

ii. N/A (no alignment code)

iii. There is no stimulus onset or behavioral event to align to. The temporal alignment event is described as "trial start of each contiguous 60 second window within a recording session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz frame rate from the miniscope recording. No resampling was needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the data files, which contains indices of blocked reward/arena positions for each recording session.

ii.
```python
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_mask = mask_from_blocked(blocked_entry)
input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. The AI explicitly chose to derive the geometry mask from the `blocked` field rather than the environment name string, noting that "the env string alone does not fully specify orientation in this dataset, while blocked does."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` indices are converted to a 3x3 binary mask where 1 = open/accessible and 0 = blocked. This mask is then flattened to a 9-element vector. This is the **inverted** representation compared to the reference (which uses 1 = blocked).

ii.
```python
def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask

input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES describe this as: "Input is a static 9D vector per trial: a flattened 3 x 3 open-bin mask. 1 means the arena partition is open, 0 means blocked." The input_names are `geometry_x0_y0` through `geometry_x2_y2`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` field in the data files, which contains 2D (x, y) coordinates with shape `(n_days, 2, T)`.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
day_pos = position[day]  # (2, T)
```

iii. The `position` field records the animal's location tracked by DeepLabCut at each timepoint.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI normalizes positions by dividing by per-axis maximum values (plus epsilon), then applies `floor` to bin into a 3x3 grid. This matches the paper code's binning approach but differs from the reference solution which uses fixed arena size (75 cm) with `np.linspace` edges.

ii.
```python
def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)
```

iii. From CONVERSION_NOTES: "I matched the paper code's binning style: no min subtraction, divide raw positions by per-axis maxima, then floor."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 3x3 grid bins are combined into a single categorical label using `x_bin * 3 + y_bin` (row-major with x as the first axis). This differs from the reference which uses `y_bin * 3 + x_bin`. Additionally, the AI performs a "blocked-bin cleanup" step: positions landing in geometry-blocked bins are reassigned to the nearest valid open bin.

ii.
```python
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)

def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    valid = np.argwhere(env_mask > 0)
    lookup = np.zeros((3, 3, 2), dtype=np.int64)
    for x in range(3):
        for y in range(3):
            if env_mask[x, y] > 0:
                lookup[x, y] = np.array([x, y], dtype=np.int64)
                continue
            dists = np.sum((valid - np.array([x, y])) ** 2, axis=1)
            lookup[x, y] = valid[np.argmin(dists)]
    return lookup
```

iii. From CONVERSION_NOTES: "After 3x3 binning, some time points fell into blocked coarse bins. For those frames only, I reassigned the label to the nearest valid open bin... this is in the spirit of the paper's within-session decoder code."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are sampled at the same 30 Hz frame rate and are aligned frame-for-frame. Both are split into trials using the same indices.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. Both arrays share the same time axis and are sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI performs several data validation checks: (1) Unregistered neurons (all-NaN) are removed. (2) Position data is checked for NaNs and an error is raised if found. (3) Registered neurons are validated to have no NaN values. (4) Unregistered neurons are validated to be consistently NaN. (5) Positions in blocked bins are reassigned to nearest valid bin.

ii.
```python
if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
```

iii. The AI took a more defensive approach with explicit validation and error raising, compared to the reference which silently filters.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the data files via `joblib.load()`, which is I/O bound. The trace arrays are large (n_days x n_cells x T). The AI also performs per-session position reassignment via the blocked-bin lookup, which adds computation.

ii. N/A

iii. The data files contain large neural trace arrays. Processing steps like NaN filtering, position discretization, and trial splitting are fast by comparison.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_nearest_valid_lookup` function uses a nested loop over the 3x3 grid to compute nearest valid bins. This is a small 9-iteration loop so vectorization would have negligible impact. The `normalize_blocked_entry` function also does some per-element processing.

ii.
```python
def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    for x in range(3):
        for y in range(3):
            ...
```

iii. These loops are over very small arrays (3x3), so vectorization would not meaningfully improve performance.

## 6-c. What processing does the code repeat multiple times?

i. The `mask_from_blocked` and `build_nearest_valid_lookup` functions are called once per session (207 times), even though many sessions share the same environment geometry. The lookup tables could be cached by blocked pattern.

ii.
```python
for day in range(trace.shape[0]):
    blocked_entry = normalize_blocked_entry(dat["blocked"][day])
    env_mask = mask_from_blocked(blocked_entry)
    lookup = build_nearest_valid_lookup(env_mask)
```

iii. Since there are only ~10 unique environment geometries across 207 sessions, caching would avoid redundant computation, though the per-call cost is trivial.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI defines an `ENV_TO_MASK` dictionary mapping environment names to 3x3 masks, but this is never used in the conversion. The mask is always derived from the `blocked` field instead. The AI also computes extensive sanity statistics (environment counts, frame lengths, reassignment counts) that are stored in metadata but not used by the decoder.

ii.
```python
ENV_TO_MASK = {
    "square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
    "o": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32),
    ...
}
```

iii. The `ENV_TO_MASK` was likely defined during exploration but replaced by the `blocked`-based approach. The sanity statistics are useful for validation but are not consumed by downstream code.
